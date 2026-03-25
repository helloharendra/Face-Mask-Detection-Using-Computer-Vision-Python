# # import the necessary packages
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
# import numpy as np

# import imutils
# import time
# import cv2
# import os
# def detect_and_predict_mask(frame, faceNet, maskNet,cnf=.5):
# 	# grab the dimensions of the frame and then construct a blob
# 	# from it
# 	(h, w) = frame.shape[:2]
# 	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
# 	# pass the blob through the network and obtain the face detections
# 	faceNet.setInput(blob)
# 	detections = faceNet.forward()
# 	# initialize our list of faces, their corresponding locations,
# 	# and the list of predictions from our face mask network
# 	faces = []
# 	locs = []
# 	preds = []
#     # loop over the detections
# 	for i in range(0, detections.shape[2]):
# 		# extract the confidence (i.e., probability) associated with
# 		# the detection
# 		confidence = detections[0, 0, i, 2]
# 		# filter out weak detections by ensuring the confidence is
# 		# greater than the minimum confidence
# 		if confidence > cnf:
# 			# compute the (x, y)-coordinates of the bounding box for
# 			# the object
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			(startX, startY, endX, endY) = box.astype("int")
# 			# ensure the bounding boxes fall within the dimensions of
# 			# the frame
# 			(startX, startY) = (max(0, startX), max(0, startY))
# 			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
#             # extract the face ROI, convert it from BGR to RGB channel
# 			# ordering, resize it to 224x224, and preprocess it
# 			face = frame[startY:endY, startX:endX]
# 			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# 			face = cv2.resize(face, (224, 224))
# 			face = img_to_array(face)
# 			face = preprocess_input(face)
# 			# add the face and bounding boxes to their respective
# 			# lists
# 			faces.append(face)
# 			locs.append((startX, startY, endX, endY))
#     # only make a predictions if at least one face was detected
# 	if len(faces) > 0:
# 		# for faster inference we'll make batch predictions on *all*
# 		# faces at the same time rather than one-by-one predictions
# 		# in the above `for` loop
# 		faces = np.array(faces, dtype="float32")
# 		preds = maskNet.predict(faces, batch_size=32)
# 	# return a 2-tuple of the face locations and their corresponding
# 	# locations
# 	return (locs, preds)


# def video(face ='face_detector',model = 'mask_detector.model',cnf=.5):
# 	print("[INFO] loading face detector model...")
# 	prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
# 	weightsPath = os.path.sep.join([face,"res10_300x300_ssd_iter_140000.caffemodel"])
# 	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# 	# load the face mask detector model from disk
# 	print("[INFO] loading face mask detector model...")
# 	maskNet = load_model(model)
# 	# initialize the video stream and allow the camera sensor to warm up
# 	print("[INFO] starting video stream...")
# 	vs = VideoStream(src=0).start()
# 	time.sleep(2.0)
# 	# loop over the frames from the video stream
# 	while True:
# 		frame = vs.read()
# 		frame = imutils.resize(frame, width=400)
# 		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
# 		for (box, pred) in zip(locs, preds):
# 			(startX, startY, endX, endY) = box
# 			(mask, withoutMask) = pred
# 			label = "Mask" if mask > withoutMask else "No Mask"
# 			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
# 			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
# 			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
# 			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
# 		cv2.imshow("Frame", frame)
# 		key = cv2.waitKey(1)
# 		if key == 27:
# 			break
# 	# do a bit of cleanup
# 	cv2.destroyAllWindows()
# 	vs.stop()
# if __name__  =='__main__':
# 	video()

import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet, cnf=0.5):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > cnf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


def video(face='face_detector', model='mask_detector.model', cnf=0.5):
    st.warning("Starting camera... Click STOP to end")

    prototxtPath = os.path.join(face, "deploy.prototxt")
    weightsPath = os.path.join(face, "res10_300x300_ssd_iter_140000.caffemodel")

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(model, compile=False)

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    stop_btn = st.button("🛑 Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("❌ Camera not working")
            break

        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, cnf)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")

        if stop_btn:
            break

    cap.release()