import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import isfile, splitext

from src.keras_utils import load_model
from src.label import readShapes
from src.utils import im2single
from glob 						import glob
from shapely.geometry import Polygon
from tqdm import tqdm
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix
from src.keras_utils import DLabel

def get_results(I,Y,threshold=.9):

	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	Affines = Y[...,2:]
	rx,ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)

	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(np.dot(A, base(vxx,vyy))) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))

		labels.append(DLabel(0,pts_prop,prob))

	final_labels = nms(labels,.1)

	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
	return final_labels
	

def detect_val_lp(model,I,max_dim,net_step,threshold):

	min_dim_img = min(I.shape[:2])
	factor 		= float(max_dim)/min_dim_img

	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	Iresized = cv2.resize(I,(w,h))

	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)

	L = get_results(Iresized,Yr,threshold)

	return L

def get_val_label(label_path):
    
    if isfile(label_path):
        L = readShapes(label_path)
    return L[0]

def calculate_iou(polygon_1, polygon_2):
    points1 = [(polygon_1[0][i], polygon_1[1][i]) for i in range(polygon_1.shape[1])]
    points2 = [(polygon_2[0][i], polygon_2[1][i]) for i in range(polygon_2.shape[1])]
    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)
    
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def get_val_images_labels(input_dir):
    images = []
    labels = []
    imgs_paths = glob('%s/*.png' % input_dir)
    for img_path in tqdm(imgs_paths, desc='Loading images and labels in validation'):
        label_path = splitext(img_path)[0] + '.txt'
        label = get_val_label(label_path=label_path)
        labels.append(label)
        
        Ivehicle = cv2.imread(img_path)
        images.append(Ivehicle)
    return images, labels
    
def get_result_on_val_set(wpod_net_path, images, labels):
    wpod_net = load_model(wpod_net_path)
    
    pred_polygons = []
    scores = []
    
    for Ivehicle in images:
        
        lp_threshold = 0.5
        
        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        
        Llp = detect_val_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,lp_threshold)
        
        if len(Llp):
            probability_of_lp = Llp[0].prob()
            loc_of_lp = Llp[0].pts
            
            scores.append(probability_of_lp)
            pred_polygons.append(loc_of_lp)
        else:
            scores.append(-1)
            pred_polygons.append(-1)
            
    preds = [{'polygon': pred_polygon, 'score': score} 
             for pred_polygon, score in zip(pred_polygons, scores)]

    targets = [{'polygon': label.pts} 
               for label in labels]
    return preds, targets

def save_loss_plot(
    OUT_DIR,
    train_loss_list, 
    x_label='iterations',
    y_label='train loss',
    save_name='train_loss'):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')

def save_AP(OUT_DIR, ap_05, ap):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        ap_05[1:], color='tab:orange', linestyle='-', 
        label='AP@0.5'
    )
    ax.plot(
        ap[1:], color='tab:red', linestyle='-', 
        label='AP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/ap.png")