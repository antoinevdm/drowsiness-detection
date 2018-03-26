# USAGE
# python deep_learning_with_opencv.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import numpy as np
import argparse
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

COUNTER_PHONE = 0

# load the input image from disk
image = cv2.imread(args["image"])

# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
start = time.time()

# loop over the frames from the video stream
while True:
    now = time.time()
    if now > (start + 3):
            start = time.time()
            COUNTER_PHONE = 0
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            1, (224, 224), (104, 117, 123))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)

    detections = net.forward()

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    idxs = np.argsort(detections[0])[::-1][:5]

    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
            # draw the top prediction on the input image
            if i == 0:
                    text = "Label: {}, {:.2f}%".format(classes[idx],
                            detections[0][idx] * 100)
                    cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            # display the predicted label + associated probability to the
            # console
            if classes[idx] == "cellular telephone":
                print("Phone detected")
                COUNTER_PHONE += 1


    if COUNTER_PHONE == 5:
        print("RACROHE CONNASSE!!!")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

        # stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

#####


# # our CNN requires fixed spatial dimensions for our input image(s)
# # so we need to ensure it is resized to 224x224 pixels while
# # performing mean subtraction (104, 117, 123) to normalize the input;
# # after executing this command our "blob" now has the shape:
# # (1, 3, 224, 224)
# print("[INFO] classification took {:.5} seconds".format(end - start))

# # sort the indexes of the probabilities in descending order (higher
# # probabilitiy first) and grab the top-5 predictions
# idxs = np.argsort(preds[0])[::-1][:5]

# # loop over the top-5 predictions and display them
# for (i, idx) in enumerate(idxs):
	# # draw the top prediction on the input image
	# if i == 0:
		# text = "Label: {}, {:.2f}%".format(classes[idx],
			# preds[0][idx] * 100)
		# cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			# 0.7, (0, 0, 255), 2)

	# # display the predicted label + associated probability to the
	# # console
	# print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		# classes[idx], preds[0][idx]))

# # display the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
