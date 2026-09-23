# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:40:02 2026

@author: utsav
"""

from ultralytics import YOLO
import os
import cv2


trained_model = YOLO("runs/detect/yolo8s_100epochs_imgsz8002/weights/best.pt")


test_image_dir = r'C:\Users\utsav\OneDrive\Desktop\license plate detector\images\val'
test_label_dir = r'C:\Users\utsav\OneDrive\Desktop\license plate detector\labels\val'
test_images = os.listdir(test_image_dir)
test_labels = os.listdir(test_label_dir)

test_image_num = 23
test_image = os.path.join(test_image_dir, test_images[test_image_num])

test_label = os.path.join(test_label_dir, test_labels[test_image_num])
label_content = open(test_label).read()

test_result = trained_model(test_image, show=True, conf=0.5, iou=0.8)

annotated = test_result[0].plot()

cv2.imshow("YOLO Output", annotated)
cv2.waitKey(0)   # waits indefinitely for a key press
cv2.destroyAllWindows()
