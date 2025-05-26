
import sys
import numpy as np
import cv2
import argparse
import keras

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder
from src.sampler import augment_sample, labels2output_map

from src.data_generator import DataGenerator
from custom_utils import *
import tensorflow as tf


def load_network(modelpath,input_dim):

	model = load_model(modelpath)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple([s.value for s in outputs.shape[1:]])
	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

def process_data_item(data_item,dim,model_stride):
	XX,llp,pts = augment_sample(data_item[0],data_item[1].pts,dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY, pts.reshape(8)

def calculate_ap(targets, preds, iou_thresholds=np.arange(0.5, 1, 0.05)):
    """
    Tính toán Average Precision (AP) dựa trên giá trị mục tiêu và giá trị dự đoán.
    """
    ious = []
    confidences = []
    aps = []

    for target, pred in zip(targets, preds):
        gt_polygon = target['polygon']
        pred_polygon = pred['polygon']
        score = pred['score']
        
        if isinstance(pred_polygon, int) and pred_polygon == -1:
            continue
        
        # Tính toán IoU giữa đa giác dự đoán và đa giác mục tiêu
        iou = calculate_iou(pred_polygon, gt_polygon)
        
        # Lưu giá trị IoU, điểm số (confidence) và cờ đánh dấu match
        ious.append(iou)
        confidences.append(score)
    
    # Chuyển sang mảng numpy
    confidences = np.array(confidences)

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
            

        # Sử dụng quy tắc AUC: duyệt từ phải qua trái và chọn giá trị precision lớn nhất cho mỗi mức recall
        # Nếu giá trị recall tiếp theo có max precision nhỏ hơn max precision trước đó, bỏ qua giá trị precision đó
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

        # Tính diện tích dưới đường cong precision-recall (AP) theo phương pháp hình thang
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        return ap
    
    aps = [get_ap_each_threshold(p) for p in iou_thresholds]
    return aps

def validate(wpod_net_path, images, labels):
    preds, targets = get_result_on_val_set(wpod_net_path=wpod_net_path, images=images, labels=labels)
    aps = calculate_ap(targets=targets, preds=preds)
    ap_05 = aps[0]
    ap_05_095 = np.mean(aps)
    return ap_05, ap_05_095
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m' 		,'--model'			,type=str   , required=True		,help='Path to previous model')
    parser.add_argument('-n' 		,'--name'			,type=str   , required=True		,help='Model name')
    parser.add_argument('-tr'		,'--train-dir'		,type=str   , required=True		,help='Input data directory for training')
    parser.add_argument('-nps'		,'--num-epochs'		,type=int   , default=200	    ,help='Number of epochs (default = 200)')
    parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
    parser.add_argument('-nb'		,'--num-batches'	,type=int   , default=45		,help='Number of batches (default = 45)')
    parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./'		,help='Output directory (default = ./)')
    parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
    parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.01		,help='Optmizer (default = 0.01)')
    args = parser.parse_args()

    netname 	= basename(args.name)
    train_dir 	= args.train_dir
    outdir 		= args.output_dir

    num_epochs 	= args.num_epochs
    batch_size 	= args.batch_size
    num_batches = args.num_batches
    dim 		= 208

    if not isdir(outdir):
        makedirs(outdir)
    
    model,model_stride,xshape,yshape = load_network(args.model,dim)
        
    opt = getattr(keras.optimizers,args.optimizer)(lr=args.learning_rate)
    model.compile(loss=loss, optimizer=opt)
        
    print('Checking input directory...')
    Files = image_files_from_folder(train_dir)
        
    Data = []
    for file in Files:
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            L = readShapes(labfile)
            I = cv2.imread(file)
            Data.append([I,L[0]])
                
    print('%d images with labels found' % len(Data))
        
    images, labels = get_val_images_labels('samples/validation-data')
        
    dg = DataGenerator(	data=Data, \
                        process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
                        xshape=xshape, \
                        yshape=(yshape[0],yshape[1],yshape[2]+1), \
                        nthreads=2, \
                        pool_size=1000, \
                        min_nsamples=100 )
    dg.start()
        
    Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
    Ytrain = np.empty((batch_size,int(dim/model_stride),int(dim/model_stride),2*4+1))
    pts = np.empty((batch_size, 8))
        
    train_losses = np.load('output_training/data_plot/train_losses.npy').tolist()
    aps_05 = np.load('output_training/data_plot/aps_05.npy').tolist()
    aps_05_095 = np.load('output_training/data_plot/aps_05_095.npy').tolist()
        
    model_path_backup = '%s/%s_backup' % (outdir,netname)
    model_path_final  = '%s/%s_final'  % (outdir,netname)
    model_path_lastest  = '%s/%s_lastest'  % (outdir,netname)
    model_path_best = '%s/%s_best' % (outdir,netname)
    
    for epoch in range(num_epochs):
        
        print('Epoch. %d (of %d)' % (epoch+1,num_epochs))
        epoch_loss = 0
        for batch in tqdm(range(num_batches), desc=f'Training on batches of epoch {epoch+1}'):    
            Xtrain,Ytrain, pts = dg.get_batch(batch_size)
            train_loss = model.train_on_batch(Xtrain,Ytrain)
            epoch_loss += train_loss
            
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch+1}\t\tLoss {avg_epoch_loss}')
        
        print(f'Save latest model {model_path_lastest}')
        save_model(model, model_path_lastest)
        ap_05, ap_05_095 = validate(wpod_net_path=model_path_lastest, images=images, labels=labels)
                
        if ap_05 > max(aps_05) or ap_05_095 > max(aps_05_095):
            save_model(model, model_path_best)
            print(f'Save best model {model_path_best}')
                
        aps_05.append(ap_05)
        aps_05_095.append(ap_05_095)
                
        print(f'AP@0.5: {ap_05}')
        print(f'AP@[0.5-0.95]: {ap_05_095}')
        save_loss_plot('output_training', train_losses)
        save_AP('output_training', aps_05, aps_05_095)
                
        # Save model every 5 epochs
        if (epoch+1) % 5 == 0:
            print('Saving model (%s)' % model_path_backup)
            save_model(model,model_path_backup)
                
        train_losses_array = np.array(train_losses)
        aps_05_array = np.array(aps_05)
        aps_05_095_array = np.array(aps_05_095)
        np.save('output_training/data_plot/train_losses.npy', train_losses_array)
        np.save('output_training/data_plot/aps_05.npy', aps_05_array)
        np.save('output_training/data_plot/aps_05_095.npy', aps_05_095_array)
            
    print('Stopping data generator')
    dg.stop()
        
    print('Saving model (%s)' % model_path_final)
    save_model(model,model_path_final)