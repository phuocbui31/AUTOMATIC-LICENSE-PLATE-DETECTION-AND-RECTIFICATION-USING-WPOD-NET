import numpy as np
import cv2
import tensorflow
import keras
import os
import matplotlib.pyplot as plt
from custom_utils import *
import seaborn as sns
import multiprocessing
from joblib import Parallel, delayed
from train import *

# arr = np.array([1, 2, 3, 4, 5, 6])
# print(np.max(arr))

# image = cv2.imread('samples/test/03009car.png')
# plt.imshow(image, cmap='Greys')
# plt.show()
# def rearrange_coordinates(input_file, output_file):
#     with open(input_file, 'r') as file:
#         lines = file.readlines()

#     with open(output_file, 'w') as file:
#         for line in lines:
#             # Tách các giá trị từ dòng
#             values = line.strip().split()
#             if len(values) != 9:
#                 print("Dữ liệu không đúng định dạng 9 giá trị trên một dòng.")
#                 continue
            
#             # Lấy các giá trị từ dòng
#             N, tlx, tly, trx, try_, brx, bry, blx, bly = values
#             N=4
            
#             # Sắp xếp lại thứ tự
#             rearranged_line = f"{N},{tlx},{trx},{brx},{blx},{tly},{try_},{bry},{bly},,\n"
            
#             # Ghi vào file đầu ra
#             file.write(rearranged_line)

# # Đường dẫn file
# input_file = 'samples/test/Dieu_0006.txt'  # Đổi tên file nếu cần

# # Gọi hàm để chuyển đổi định dạng
# folder_dir = 'samples/test'
# labels_paths = glob('%s/*.txt' % folder_dir)
# for label in labels_paths:
#     rearrange_coordinates(label, label)

def calculate_ap(targets, preds, iou_thresholds=np.arange(0.5, 1, 0.05)):
    """
    Tính toán Average Precision (AP) dựa trên giá trị mục tiêu và giá trị dự đoán.
    """
    ious = []
    confidences = []
    aps = []

    # Lặp qua từng cặp giá trị dự đoán và mục tiêu
    for target, pred in zip(targets, preds):
        gt_polygon = target['polygon']
        pred_polygon = pred['polygon']
        score = pred['score']
        
        if isinstance(pred_polygon, int) and pred_polygon == -1:
            # Không tìm thấy kết quả dự đoán
            # Chỉ tính cho Recall, không thêm vào danh sách dự đoán để tránh ảnh hưởng đến Precision
            continue
        
        # Tính toán IoU giữa đa giác dự đoán và đa giác mục tiêu
        iou = calculate_iou(pred_polygon, gt_polygon)
        
        # Lưu giá trị IoU, điểm số (confidence) và cờ đánh dấu match
        ious.append(iou)
        confidences.append(score)
        # matches.append(iou >= iou_threshold)
    
    # Chuyển sang mảng numpy
    confidences = np.array(confidences)
    # matches = np.array(matches, dtype=np.float32)
    
    # print("IoUs:", ious)
    # print("Confidences:", confidences)
    # print("Matches:", matches)

    # Sắp xếp theo confidence (theo thứ tự giảm dần)
    sorted_indices = np.argsort(-confidences)
    # matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]
    
    def get_ap_each_threshold(iou_threshold):
        matches = ious >= iou_threshold
        matches = np.array(matches, dtype=np.float32)
        matches = matches[sorted_indices]
            
        # Tính toán đường cong precision-recall
        tp_cumsum = np.cumsum(matches)
        fp_cumsum = np.cumsum(1 - matches)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recalls = tp_cumsum / (len(targets) + 1e-7)
        # print(precisions)
        # print(recalls)
            

        # Sử dụng quy tắc AUC: duyệt từ phải qua trái và chọn giá trị precision lớn nhất cho mỗi mức recall
        # Nếu giá trị recall tiếp theo có max precision nhỏ hơn max precision trước đó, bỏ qua giá trị precision đó
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

        # print(precisions)
        # Tính diện tích dưới đường cong precision-recall (AP) theo phương pháp hình thang
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        return ap
    
    # def process_pair(target, pred):
    #     gt_polygon = target['polygon']
    #     pred_polygon = pred['polygon']
    #     score = pred['score']

    #     if isinstance(pred_polygon, int) and pred_polygon == -1:
    #         # Không có dự đoán, bỏ qua
    #         return None, None

    #     # Tính toán IoU giữa đa giác dự đoán và đa giác mục tiêu
    #     iou = calculate_iou(pred_polygon, gt_polygon)
    #     return iou, score

    # # Sử dụng đa luồng để tính toán nhanh hơn cho các cặp giá trị mục tiêu và dự đoán
    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(process_pair)(t, p) for t, p in zip(targets, preds))

    # # Lưu trữ kết quả vào danh sách ious và confidences
    # for iou, score in results:
    #     if iou is not None and score is not None:
    #         ious.append(iou)
    #         confidences.append(score)

    # # Chuyển sang mảng numpy để sử dụng tính toán vector hóa
    # ious = np.array(ious)
    # confidences = np.array(confidences)

    # # Sắp xếp theo confidence (theo thứ tự giảm dần)
    # sorted_indices = np.argsort(-confidences)
    # ious = ious[sorted_indices]
    # confidences = confidences[sorted_indices]

    # def get_ap_each_threshold(iou_threshold):
    #     # Tính toán các cờ match dựa trên ngưỡng IoU
    #     matches = ious >= iou_threshold
    #     matches = matches.astype(np.float32)

    #     # Tính toán đường cong precision-recall
    #     tp_cumsum = np.cumsum(matches)
    #     fp_cumsum = np.cumsum(1 - matches)
    #     precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    #     recalls = tp_cumsum / (len(targets) + 1e-7)

    #     # Sử dụng quy tắc AUC để tối ưu giá trị precision
    #     precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    #     # Tính diện tích dưới đường cong precision-recall (AP) theo phương pháp hình thang
    #     ap = np.trapz(precisions, recalls)
    #     return ap

    # # Sử dụng vector hóa để tính toán AP cho tất cả các ngưỡng IoU
    # aps = Parallel(n_jobs=num_cores)(delayed(get_ap_each_threshold)(p) for p in iou_thresholds)
    aps = [get_ap_each_threshold(p) for p in iou_thresholds]
    return aps

# wpod_net_path = 'models/eccv-model-scracth'
# preds, targets = get_result_on_val_set(wpod_net_path)

# # Tính toán AP
# average_precisions = calculate_ap(targets, preds)

# print(f'AP@0.5: {average_precisions[0]:.4f}')
# print(f'AP@[0.5-0.95]: {np.mean(average_precisions)}')

# train_losses = np.load('output_training/data_plot/train_losses.npy').tolist()
# aps_05 = np.load('output_training/data_plot/aps_05.npy').tolist()
# aps_05_095 = np.load('output_training/data_plot/aps_05_095.npy').tolist()
# print(train_losses)
# print(aps_05)
# print(aps_05_095)

images, labels = get_val_images_labels('validation_path')
wpod_net_path = 'models/my-trained-model/my-trained-model_lastest'
ap_05, ap_05_095 = validate(wpod_net_path=wpod_net_path)