import cv2
import serial
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
import time

# Initialize serial communication
ser = serial.Serial('/dev/tty.HC-05', 9600)
time.sleep(2)

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

# Define default servo angles
ang1 = 90
ang2 = 90
ang3 = 90
ang4 = 90
ang5 = 90
ang6 = 90
servo_angles = [ang1, ang2, ang3, ang4, ang5, ang6]  # Example default angles for 5 servos

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

        # Update servo angles based on detected label
        if label == "0 Claw Open":
            ang1 = 90  # Update with specific angles
        elif label == "1 Claw Closed":
            ang1 = 30  # Update with specific angles
        elif label == "2 Thumbs Up":
            ang1 = 30
            ang2 = 50
            ang3 = 30
            ang4 = 90 # Update with specific angles
        elif label == "3 Thumbs Down":
            ang1 = 90
            ang2 = 50
            ang3 = 30
            ang4 = 150 # Update with specific angles
        else:
            servo_angles = [90, 90, 90, 90, 90, 90]  # Default angles for other labels

        servo_angles = [ang1, ang2, ang3, ang4, ang5, ang6]

        # Send servo angles to Arduino
        # ser.write(f"{servo_angles[0]}\n".encode())
        # Convert the list of angles to a comma-separated string
        angles_str = ','.join(map(str, servo_angles))

        # Send the angles array to Arduino
        ser.write(f"{angles_str}\n".encode())  # Encode the string with newline character to bytes
        print(label)
        print(f"Sent angles: {angles_str}")

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
ser.close()









# import cv2
# import serial
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import tensorflow as tf

# # Initialize serial communication
# try:
#     ser = serial.Serial('/dev/tty.usbmodem101', 9600)
# except serial.SerialException as e:
#     print(f"Error opening serial port: {e}")
#     exit(1)

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

# # Define default servo angles
# servo_angles = [90, 90, 90, 90, 90]  # Example default angles for 5 servos

# try:
#     while True:
#         success, img = cap.read()
#         hands, img = detector.findHands(img)
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#             imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
#             imgCropShape = imgCrop.shape

#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap: wCal + wGap] = imgResize

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap: hCal + hGap, :] = imgResize

#             # Preprocess the image for the model
#             imgWhite_resized = cv2.resize(imgWhite, (224, 224))
#             imgWhite_resized = imgWhite_resized.astype(np.float32)
#             imgWhite_resized = np.expand_dims(imgWhite_resized, axis=0)
#             imgWhite_resized = (imgWhite_resized / 127.5) - 1  # Normalize the image to [-1, 1]

#             # Set the tensor to the image
#             interpreter.set_tensor(input_details[0]['index'], imgWhite_resized)

#             # Perform inference
#             interpreter.invoke()

#             # Get the prediction result
#             output_data = interpreter.get_tensor(output_details[0]['index'])
#             prediction = np.argmax(output_data)
#             label = labels[prediction]

#             # Update servo angles based on detected label
#             if label == "0 Claw Open":
#                 servo_angles = [90, 90, 90, 90, 90]  # Update with specific angles
#             elif label == "1 Claw Closed":
#                 servo_angles = [0, 0, 0, 0, 0]  # Update with specific angles
#             elif label == "2 Thumbs Up":
#                 servo_angles = [45, 45, 45, 45, 45]  # Update with specific angles
#             elif label == "3 Thumbs Down":
#                 servo_angles = [135, 135, 135, 135, 135]  # Update with specific angles
#             else:
#                 servo_angles = [90, 90, 90, 90, 90]  # Default angles for other labels

#             # Send servo angles to Arduino
#             ser.write(bytearray(servo_angles))

#             # Display the result
#             cv2.rectangle(img, (x - offset, y - offset - 50), (x + w + offset, y - offset), (255, 0, 255), cv2.FILLED)
#             cv2.putText(img, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#             cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

#             cv2.imshow('ImageCrop', imgCrop)
#             cv2.imshow('ImageWhite', imgWhite)

#         cv2.imshow('Image', img)

#         key = cv2.waitKey(1)

#         if key == ord('q'):
#             break
# finally:
#     cap.release()
#     ser.close()
#     cv2.destroyAllWindows()










# import serial
# import time

# # Initialize serial communication
# ser = serial.Serial('/dev/cu.HC-05', 9600)
# time.sleep(2)  # Wait for the serial connection to initialize

# # Define servo angles to test
# servo_angle = 60

# # Send servo angles to Arduino
# ser.write(bytes([servo_angle]))
# print(f"Sent angles: {servo_angle}")

# # Close the serial connection
# ser.close()








# import serial
# import time

# # Initialize serial communication
# ser = serial.Serial('/dev/tty.HC-05', 9600)
# time.sleep(2)  # Wait for the serial connection to initialize

# # Define the servo angle to test
# servo_angle = 150

# # Send the servo angle to Arduino
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")

# # Close the serial connection
# ser.close()








# import serial
# import time

# # Initialize serial communication
# ser = serial.Serial('/dev/tty.HC-05', 9600)
# time.sleep(2)  # Wait for the serial connection to initialize

# # Define the servo angles to test
# servo_angles = [30, 60, 90, 120, 150, 180]

# # Send each servo angle to Arduino with a delay in between
# for angle in servo_angles:
#     ser.write(f"{angle}\n".encode())  # Encode the string with newline character to bytes
#     print(f"Sent angle: {angle}")
#     time.sleep(1)  # Wait for 1 second before sending the next angle

# # Close the serial connection
# ser.close()








# Array of angles
# import serial
# import time

# # Initialize serial communication
# ser = serial.Serial('/dev/tty.HC-05', 9600)
# time.sleep(2)  # Wait for the serial connection to initialize

# # Define the servo angles to test
# servo_angles = [20, 30, 90, 120, 150, 180]

# # Convert the list of angles to a comma-separated string
# angles_str = ','.join(map(str, servo_angles))

# # Send the angles array to Arduino
# ser.write(f"{angles_str}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angles: {angles_str}")

# # Close the serial connection
# ser.close()




# import serial
# import time

# # Initialize serial communication
# ser = serial.Serial('/dev/tty.HC-05', 9600)
# time.sleep(2)  # Wait for the serial connection to initialize

# # Define the servo angle to test
# servo_angle = 0
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 30
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 60
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 90
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 120
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 150
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")
# time.sleep(3)

# servo_angle = 180
# ser.write(f"{servo_angle}\n".encode())  # Encode the string with newline character to bytes
# print(f"Sent angle: {servo_angle}")

# # Close the serial connection
# ser.close()