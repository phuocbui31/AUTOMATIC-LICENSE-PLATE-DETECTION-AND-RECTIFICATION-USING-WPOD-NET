import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label                 import Label, lwrite
from os.path                 import splitext, basename, isdir
from os                     import makedirs
from src.utils                 import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

    try:
    
        input_dir  = sys.argv[1]
        output_dir = sys.argv[2]

        vehicle_threshold = 0.3

        # Vehicle model paths
        vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
        vehicle_netcfg  = b'data/vehicle-detector/yolo-voc.cfg'
        vehicle_dataset = b'data/vehicle-detector/voc.data'

        # Load YOLO model for vehicle detection
        vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
        vehicle_meta = dn.load_meta(vehicle_dataset)

        # Get list of image paths from input directory
        imgs_paths = image_files_from_folder(input_dir)
        imgs_paths.sort()

        # Create output directory if it does not exist
        if not isdir(output_dir):
            makedirs(output_dir)

        print('Searching for vehicles using YOLO...')

        for i, img_path in enumerate(imgs_paths):

            print(f'\tScanning {img_path}')

            bname = basename(splitext(img_path)[0])

            img_path_bytes = img_path.encode('utf-8')  # Convert to bytes

            # Detect vehicles in the image
            R, _ = detect(vehicle_net, vehicle_meta, img_path_bytes, thresh=vehicle_threshold)

            # Filter results to only keep cars and buses
            R = [r for r in R if r[0] in [b'car', b'bus']]

            print(f'\t\t{len(R)} cars found')

            if len(R):
                # Read the original image
                Iorig = cv2.imread(img_path)
                WH = np.array(Iorig.shape[1::-1], dtype=float)
                Lcars = []

                # Process each detected vehicle
                for i, r in enumerate(R):
                    cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
                    tl = np.array([cx - w / 2., cy - h / 2.])
                    br = np.array([cx + w / 2., cy + h / 2.])
                    label = Label(0, tl, br)
                    Icar = crop_region(Iorig, label)

                    Lcars.append(label)

                    # Save the cropped car image
                    cv2.imwrite(f'{output_dir}/{bname}_{i}car.png', Icar)

                # Write the labels to a text file
                lwrite(f'{output_dir}/{bname}_cars.txt', Lcars)

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)