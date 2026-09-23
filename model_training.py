# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:40:02 2026

@author: utsav
"""

from ultralytics import YOLO

def main():

    model = YOLO("yolov8s.pt") #pre-trained base model
    
    model.train(data = "C:/Users/utsav/OneDrive/Desktop/license plate detector/dataset.yaml", 
                epochs=100, imgsz=800, batch = 8,
                name="yolo8s_100epochs_imgsz800")

if __name__ == "__main__":
    main()
    