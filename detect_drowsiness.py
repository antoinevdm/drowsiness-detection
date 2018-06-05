from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame
import argparse
import imutils
import time
import dlib
import cv2

# Play an alarm sound
def soundAlarm(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

# Compute the distances between the two sets of vertical landmarks
def eyeAspectRation(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

pygame.init()
# Handle arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
    help="path alarm .WAV file")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
    help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())


# Load labels' signification
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Load Caffe model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
fps = FPS().start()

# Define the eye aspect ratio
EYE_AR_THRESH = 0.3
# Define the number of closed eyes' consecutive frames
EYE_AR_CONSEC_FRAMES = 48

# Initialize the frame counter and alarm on flags
COUNTER = 0
COUNTER_FACE = 0
COUNTER_PHONE = 0
ALARM_ON = False
ALARM_ON_FACE = False
ALARM_ON_PHONE = False

# Load the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)

# Loop over video stream's frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    (h, w) = frame.shape[:2]

    # Face detection
    if len(rects) == 0:
        COUNTER_FACE +=1
        if COUNTER_FACE >= EYE_AR_CONSEC_FRAMES:
            if (not ALARM_ON_PHONE) & (not ALARM_ON_FACE) & (not ALARM_ON):
                print("No face detected")
                ALARM_ON_FACE = True

                if args["alarm"] != "":
                    t = Thread(target=soundAlarm, args=(args["alarm"],))
                    t.deamon = True
                    t.start()
            # Draw on the frame
            cv2.putText(frame, "No face alert!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        COUNTER_FACE = 0
        ALARM_ON_FACE = False

    # Closed eyes detection
    if (COUNTER_FACE == 0) & (COUNTER_PHONE == 0):
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eyeAspectRation(leftEye)
            rightEAR = eyeAspectRation(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # If the eyes were closed for a sufficient time
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if (not ALARM_ON_PHONE) & (not ALARM_ON_FACE) & (not ALARM_ON):
                        print("Open your eyes")
                        ALARM_ON = True

                        if args["alarm"] != "":
                            t = Thread(target=soundAlarm, args=(args["alarm"],))
                            t.deamon = True
                            t.start()

                    # Draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            # Draw the computed eye aspect ratio on the frame
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Handel cellphone detection
    if (COUNTER_FACE == 0) & (COUNTER == 0):
        # Grab the frame from the threaded video file stream, resize it, and convert it to grayscale
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            1, (224, 224), (104, 117, 123))

        net.setInput(blob)
        detections = net.forward()

        # Sort the indexes of the probabilities in descending probability order and grab the top-10 predictions
        idxs = np.argsort(detections[0])[::-1][:7]
        found = False
        #  Loop over the top-10 predictions, looking for cellular phone
        for (i, idx) in enumerate(idxs):
                if classes[idx] == "cellular telephone":
                    found = True
                    COUNTER_PHONE += 1

        if not found:
            COUNTER_PHONE = 0
            ALARM_ON_PHONE = False

        if COUNTER_PHONE >= 5:
            if (not ALARM_ON_PHONE) & (not ALARM_ON_FACE) & (not ALARM_ON):
                print("Hang-up")
                ALARM_ON_PHONE = True

                if args["alarm"] != "":
                    t = Thread(target=soundAlarm, args=(args["alarm"],))
                    t.deamon = True
                    t.start()

            # Draw on image
            cv2.putText(frame, "Cellphone detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fps.update()
    else:
        COUNTER_PHONE = 0

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit by pressing q
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
