import glob
import cv2
import os
import numpy as np
img_files = glob.glob("./blood_cell/*/*.tif")
for file_path in img_files:
    img = cv2.imread(file_path,0)
    img = 255 -img
    basename,kakuchousi = os.path.basename(file_path).split(".")
    print(basename,kakuchousi)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_internal = np.zeros(img.shape)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 25 or 2000*2< area:
            continue
        print(i)
        if hierarchy[0][i][2] == -1: 

            image_internal = cv2.drawContours(image_internal, contours, i, 255, -1)
    cv2.imwrite('./output/'+basename+".tif",image_internal)
    