# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:40:02 2026

@author: utsav
"""
from ultralytics import YOLO

import os
import cv2

#contains the definition of the functions used in this code
import utils
from paddleocr import PaddleOCR

# directory containing validation/test images
test_image_dir = r'C:\Users\utsav\OneDrive\Desktop\license plate detector\images\val'
# directory containing the YOLO label files corresponding to the validation/test images
test_label_dir = r'C:\Users\utsav\OneDrive\Desktop\license plate detector\labels\val'

# list of all the files
# test_images = os.listdir(test_image_dir)
# list of all the labels
# test_labels = os.listdir(test_label_dir)

# Get only image files
test_images = sorted(
    [
        f for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

# Select image number
test_image_num = 53

image_filename = test_images[test_image_num]

# Construct full image path
test_image = os.path.join(test_image_dir, image_filename)

# Replace image extension with .txt
label_filename = os.path.splitext(image_filename)[0] + ".txt"

# Construct corresponding label path
test_label = os.path.join(test_label_dir, label_filename)


ground_truth = utils.read_yolo_labels(test_label)

print(ground_truth)

# load previously trained model
trained_model = YOLO(r'C:\Users\utsav\OneDrive\Desktop\license plate detector\runs\detect\yolo8s_100epochs_imgsz8002\weights\best.pt')
# initialize OCR
# consider the orientation of the text lines: use_textline_orientation=True
# primarily interested in English: lang = 'en'
# use gpu
reader = PaddleOCR(use_textline_orientation=True, lang = 'en', use_gpu = True)
# load test image
image = cv2.imread(test_image)
#if cv2.imread() cannot find or read the image, it returns None
if image is None:
    
    raise FileNotFoundError(f"Could not load image: {test_image}")
# print the image filename and its dimensions
print("Test image:", test_images[test_image_num])
print("Image dimensions:", image.shape)

# Run YOLO license-plate detection
results = trained_model(image)

# Process YOLO detection results
for r in results:
    # Extract the bounding-box coordinates from the YOLO result. 
    # 
    # Each bounding box has the format: 
    # 
    # [x1, y1, x2, y2] 
    # 
    # where: 
    # 
    # x1 = left edge 
    # y1 = top edge 
    # x2 = right edge 
    # y2 = bottom edge 
    # 
    # The coordinates are measured in pixels.
    boxes = r.boxes.xyxy
    # There may be more than one license plate in an image. 
    # Therefore, process each detected bounding box individually.
    for box in boxes:
        # Use try/except so that one problematic detection does not terminate the entire program.
        try:
            # Convert the four bounding-box coordinates to integers. 
            # 
            # YOLO returns numerical values that may be floating-point numbers. 
            # OpenCV image coordinates need integer pixel values.

            x1, y1, x2, y2 = map(int, box)
    
            # Crop the license plate
            plate = image[y1:y2, x1:x2]
            
            # Make sure the detected bounding box produced a non-empty image. 
            # 
            # If the crop has zero pixels, something went wrong with the bounding-box coordinates. 
            if plate.size == 0: 
                print("Warning: Empty license-plate crop.") 
                continue
            # License plates can be relatively small inside the original image. 
            #Enlarging the crop gives the OCR model more pixels with which to analyze the characters. 
            # 
            # fx=3 means: 
            # 
            # width -> 3 times larger 
            #
            # fy=3 means:
            #
            #height -> 3 times larger 
            # 
            # INTER_CUBIC uses cubic interpolation when creating the additional pixels.
            plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            # Send the enlarged license-plate image to PaddleOCR. 
            # 
            # PaddleOCR attempts to: 
            # 
            # 1. Locate text within the plate. 
            # 2. Recognize the characters. 
            # 3. Return the recognized text and confidence information.    
            
            result = reader.ocr(plate)
            # PaddleOCR returns a nested result structure. 
            # 
            # In the version/configuration being used here, the recognized 
            # text is located at: 
            # 
            # result[0][0][1][0] 
            # 
            # The indexing represents the different levels of the OCR result structure.
            plate_text_raw = result[0][0][1][0]
            plate_text, confidence = utils.extract_ocr_result(result)            
            
            # Print both the raw OCR output and the cleaned output. 
            # 
            # Seeing both is extremely useful while debugging OCR.
            print("Raw OCR result:      ", plate_text_raw)
            print("Cleaned plate text:  ", plate_text)
            print("Confidence:", confidence)
            # Draw a green rectangle around the detected license plate. 
            # 
            # The four arguments are: 
            # 
            # image -> image to modify 
            # (x1, y1) -> top-left corner 
            # (x2, y2) -> bottom-right corner 
            # (0,255,0) -> green in OpenCV's BGR format 
            # 2 -> line thickness
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        # If something goes wrong while processing this particular plate, 
        # print the error and continue processing the remaining detections.
        except Exception as e:
            # Print the Python error message.
            print("Error while processing detection:", e)
            # Move to the next detected plate.
            continue
# Display the results
cv2.imshow("License Plate Detection", image)
cv2.imshow("Plate crop used for OCR", plate)

# Wait for user input
# Keep the OpenCV windows open until the user presses a key
# The argument 0 means 'wait indefinitely'
cv2.waitKey(0)  

# Close the windows
# Close all OpenCV windows that were opened by this program
cv2.destroyAllWindows()
