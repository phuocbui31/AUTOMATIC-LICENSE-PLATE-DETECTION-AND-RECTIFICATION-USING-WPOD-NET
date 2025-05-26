from glob import glob
from src.utils import image_files_from_folder
from os.path import isfile, splitext
import cv2
from src.label import readShapes

def rearrange_coordinates(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            # Tách các giá trị từ dòng
            values = line.strip().split()
            if len(values) != 9:
                print("Dữ liệu không đúng định dạng 9 giá trị trên một dòng.")
                continue
            
            # Lấy các giá trị từ dòng
            N, tlx, tly, trx, try_, brx, bry, blx, bly = values
            N=4
            
            # Sắp xếp lại thứ tự
            rearranged_line = f"{N},{tlx},{trx},{brx},{blx},{tly},{try_},{bry},{bly},,\n"
            
            # Ghi vào file đầu ra
            file.write(rearranged_line)

# Gọi hàm để chuyển đổi định dạng
# folder_dir = '/home/phuocbui3102/wpod-net/labels_tmp'
# labels_paths = glob('%s/*.txt' % folder_dir)
# for label in labels_paths:
#     rearrange_coordinates(label, label)
    
Files = image_files_from_folder('samples/train-data')
    
labels_paths = []
for file in Files:
    label_path = splitext(file)[0] + '.txt'
    labels_paths.append(label_path)
print(len(labels_paths))