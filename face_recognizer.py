#!/usr/bin/python

import cv2, os
import numpy as np
from PIL import Image
#from time import sleep

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_name(a):
	if a == 1:
		return 'AC'
	elif a == 2:
		return 'AM'
	elif a == 3:
		return 'AY'
	else :
		return 'Anonymous'

def get_images_and_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []

    for image_path in image_paths:

        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0])
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels

def train(image_loc) :
	
	global recognizer
	path = image_loc
	images, labels = get_images_and_labels(path)
	cv2.destroyAllWindows()
	# Perform the tranining
	recognizer.train(images, np.array(labels))
	print "RECOGNIZER TRAINED\n"
	recognizer.save("AAA.yml")

def recog():
	
	global recognizer
	video_capture = cv2.VideoCapture(0)

	while True:
		if not video_capture.isOpened():
			print 'Unable to load camera.'
#        	sleep(5)
        	pass
	    # Capture frame-by-frame
   		ret, frame = video_capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		predict_image = np.array(gray, 'uint8')
		faces = faceCascade.detectMultiScale(
        	predict_image,
        	scaleFactor=1.1,
        	minNeighbors=5,
        	minSize=(30, 30)
   		)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


		if cv2.waitKey(1) & 0xFF == ord('s'):
			print "Recognize...\n"
		
			for (x, y, w, h) in faces:
				nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
				print "FACE recognized as %s with conf %d\n" % (get_name(nbr_predicted), conf)
	
  	  	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		cv2.imshow('Video', frame)

	video_capture.release()
	cv2.destroyAllWindows()

print "Enter 't' for train model or 'r' to recognize\n"
action = str(raw_input('-->\n'))
image_path = './image-samples'

while True:
	if action == 't':
		train()
	elif action == 'r':
		recognizer.load("AAA.yml")
		print "Model loaded\n"
		recog()
	else:
		sys.exit()

