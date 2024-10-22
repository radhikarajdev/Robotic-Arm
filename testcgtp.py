# import cv2
# import serial
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time
# import tensorflow as tf

# ser = serial.Serial('COM3', 9600)

# # Initialize webcam and hand detector
# cap = cv2.VideoCapture(1)  # Change to 0 if you only have one camera
# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# counter = 0

# # Function to load labels from the labels.txt file
# def load_labels(label_path):
#     with open(label_path, 'r') as f:
#         labels = f.read().strip().split('\n')
#     return labels

# # Load Teachable Machine model and labels
# interpreter = tf.lite.Interpreter(model_path="/Users/radhikarajdev/Desktop/converted_tflite/model_unquant.tflite")
# interpreter.allocate_tensors()
# labels = load_labels("/Users/radhikarajdev/Desktop/converted_tflite/labels.txt")

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#         imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
#         imgCropShape = imgCrop.shape

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap: wCal + wGap] = imgResize

#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap: hCal + hGap, :] = imgResize

#         # Preprocess the image for the model
#         imgWhite_resized = cv2.resize(imgWhite, (224, 224))
#         imgWhite_resized = imgWhite_resized.astype(np.float32)
#         imgWhite_resized = np.expand_dims(imgWhite_resized, axis=0)
#         imgWhite_resized = (imgWhite_resized / 127.5) - 1  # Normalize the image to [-1, 1]

#         # Set the tensor to the image
#         interpreter.set_tensor(input_details[0]['index'], imgWhite_resized)

#         # Perform inference
#         interpreter.invoke()

#         # Get the prediction result
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         prediction = np.argmax(output_data)
#         label = labels[prediction]

#         if label == "0 Claw Open":
#             print("Detected label Claw Open")
#         elif label == "1 Claw Closed":
#             print("Detected label Claw Closed")
#         elif label == "2 Thumbs Up":
#             print("Detected label Thumbs Up")
#         elif label == "3 Thumbs Down":
#             print("Detected label Thumbs closed")
#         else:
#             print("Detected label")

#         # Display the result
#         cv2.rectangle(img, (x - offset, y - offset - 50), (x + w + offset, y - offset), (255, 0, 255), cv2.FILLED)
#         cv2.putText(img, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

#         cv2.imshow('ImageCrop', imgCrop)
#         cv2.imshow('ImageWhite', imgWhite)

#     cv2.imshow('Image', img)

#     key = cv2.waitKey(1)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()










import cv2
import serial
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

ser = serial.Serial('COM3', 9600)

# Initialize webcam and hand detector
cap = cv2.VideoCapture(1)  # Change to 0 if you only have one camera
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Function to load labels from the labels.txt file
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

# Load Teachable Machine model and labels
interpreter = tf.lite.Interpreter(model_path="/Users/radhikarajdev/Desktop/converted_tflite/model_unquant.tflite")
interpreter.allocate_tensors()
labels = load_labels("/Users/radhikarajdev/Desktop/converted_tflite/labels.txt")

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Preprocess the image for the model
        imgWhite_resized = cv2.resize(imgWhite, (224, 224))
        imgWhite_resized = imgWhite_resized.astype(np.float32)
        imgWhite_resized = np.expand_dims(imgWhite_resized, axis=0)
        imgWhite_resized = (imgWhite_resized / 127.5) - 1  # Normalize the image to [-1, 1]

        # Set the tensor to the image
        interpreter.set_tensor(input_details[0]['index'], imgWhite_resized)

        # Perform inference
        interpreter.invoke()

        # Get the prediction result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        label = labels[prediction]

        # Define angles based on the label
        angles = [0, 0, 0, 0, 0]  # Placeholder for five angles

        if label == "0 Claw Open":
            print("Detected label Claw Open")
            angles = [90, 90, 90, 90, 90]  # Example angles for Claw Open
        elif label == "1 Claw Closed":
            print("Detected label Claw Closed")
            angles = [0, 0, 0, 0, 0]  # Example angles for Claw Closed
        elif label == "2 Thumbs Up":
            print("Detected label Thumbs Up")
            angles = [45, 45, 45, 45, 45]  # Example angles for Thumbs Up
        elif label == "3 Thumbs Down":
            print("Detected label Thumbs Down")
            angles = [135, 135, 135, 135, 135]  # Example angles for Thumbs Down
        else:
            print("Detected label")

        # Send the angles to Arduino
        angle_str = ','.join(map(str, angles)) + '\n'
        ser.write(angle_str.encode())

        # Display the result
        cv2.rectangle(img, (x - offset, y - offset - 50), (x + w + offset, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
