import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
# import serial

# Config
write_video = True
debug = True

# ser = serial.Serial('COM3', 9600)

x_min = 0
x_mid = 75
x_max = 150
palm_angle_min = -50
palm_angle_mid = 20

servo_angle = [x_mid, 90, 90, 60]  # [x, y, z, claw]
prev_servo_angle = servo_angle
fist_threshold = 7

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

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

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hands_detected, img = detector.findHands(image)
        label = "none"
        if hands_detected:
            hand = hands_detected[0]
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

            imgWhite_resized = cv2.resize(imgWhite, (224, 224))
            imgWhite_resized = imgWhite_resized.astype(np.float32)
            imgWhite_resized = np.expand_dims(imgWhite_resized, axis=0)
            imgWhite_resized = (imgWhite_resized / 127.5) - 1

            interpreter.set_tensor(input_details[0]['index'], imgWhite_resized)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data)
            label = labels[prediction]

            if label == "0 Claw Open":
                print("Detected label Claw Open")
            elif label == "1 Claw Closed":
                print("Detected label Claw Closed")
            elif label == "2 Thumbs Up":
                print("Detected label Thumbs Up")
            elif label == "3 Thumbs Down":
                print("Detected label Thumbs closed")
            else:
                print("Detected label")

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                WRIST = hand_landmarks.landmark[0]
                INDEX_FINGER_MCP = hand_landmarks.landmark[5]
                palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + (WRIST.y - INDEX_FINGER_MCP.y)**2 + (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

                distance = palm_size
                angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance
                angle = int(angle * 180 / 3.1415926)
                angle = clamp(angle, palm_angle_min, palm_angle_mid)
                servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)

                if servo_angle != prev_servo_angle:
                    print("Servo angle: ", servo_angle)
                    prev_servo_angle = servo_angle
                    if not debug:
                        print(servo_angle[0])
                        # ser.write(bytearray(servo_angle))

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
        image = cv2.flip(image, 1)
        combined_text = f"{label}, Servo: {servo_angle}"
        cv2.putText(image, combined_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image)

        if write_video:
            out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            if write_video:
                out.release()
            break

cap.release()
cv2.destroyAllWindows()









# import cv2
# import serial
# import mediapipe as mp
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time
# import tensorflow as tf

# # Config
# write_video = True
# debug = True

# # if not debug:
# #     ser = serial.Serial('COM4', 115200)

# x_min = 0
# x_mid = 75
# x_max = 150
# palm_angle_min = -50
# palm_angle_mid = 20

# servo_angle = [x_mid, 90, 90, 60]  # [x, y, z, claw]
# prev_servo_angle = servo_angle
# fist_threshold = 7

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
# cap = cv2.VideoCapture(1)
# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# counter = 0

# # Video writer
# if write_video:
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

# clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
# map_range = lambda x, in_min, in_max, out_min, out_max: abs((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

# # Initialize hand detector
# detector = HandDetector(maxHands=1)

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

# with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue

#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



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

#             if label == "0 Claw Open":
#                 print("Detected label Claw Open")
#             elif label == "1 Claw Closed":
#                 print("Detected label Claw Closed")
#             elif label == "2 Thumbs Up":
#                 print("Detected label Thumbs Up")
#             elif label == "3 Thumbs Down":
#                 print("Detected label Thumbs closed")
#             else:
#                 print("Detected label")




#         if results.multi_hand_landmarks:
#             if len(results.multi_hand_landmarks) == 1:
#                 hand_landmarks = results.multi_hand_landmarks[0]
#                 WRIST = hand_landmarks.landmark[0]
#                 INDEX_FINGER_MCP = hand_landmarks.landmark[5]
#                 palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + (WRIST.y - INDEX_FINGER_MCP.y)**2 + (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

#                 # Calculate x angle
#                 distance = palm_size
#                 angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance
#                 angle = int(angle * 180 / 3.1415926)
#                 angle = clamp(angle, palm_angle_min, palm_angle_mid)
#                 servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)

#                 if servo_angle != prev_servo_angle:
#                     print("Servo angle: ", servo_angle)
#                     prev_servo_angle = servo_angle
#                     if not debug:
#                         print(servo_angle[0])
#                         # ser.write(bytearray(servo_angle))
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())
#                 # Flip the image horizontally for a selfie-view display.
#         image = cv2.flip(image, 1)
#         combined_text = f"{label}, Servo: {servo_angle}"
#         cv2.putText(image, combined_text, (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # cv2.putText(image, str(servo_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.imshow('MediaPipe Hands', image)

#         if write_video:
#             out.write(image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             if write_video:
#                 out.release()
#             break
# cap.release()
# cv2.destroyAllWindows()












# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("/Users/radhikarajdev/Desktop/converted_keras/keras_model.h5" , "/Users/radhikarajdev/Desktop/converted_keras/labels.txt")
# offset = 20
# imgSize = 300
# counter = 0

# labels = ["Claw Open","Claw Closed"]


# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

#         imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
#         imgCropShape = imgCrop.shape

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWhite[:, wGap: wCal + wGap] = imgResize
#             prediction , index = classifier.getPrediction(imgWhite, draw= False)
#             print(prediction, index)

#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap: hCal + hGap, :] = imgResize
#             prediction , index = classifier.getPrediction(imgWhite, draw= False)

       
#         cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

#         cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
#         cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

#         cv2.imshow('ImageCrop', imgCrop)
#         cv2.imshow('ImageWhite', imgWhite)

#     cv2.imshow('Image', imgOutput)
#     cv2.waitKey(1)
    