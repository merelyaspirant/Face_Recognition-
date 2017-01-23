import cv2
import sys

video_capture = cv2.VideoCapture(0)

max_captures = 50
m = 1
while m <= max_captures :
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
	
    if cv2.waitKey(1) & 0xFF == ord('s'):
		print "Take SelFIE %d\n" % m
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		name = 'image-samples/' + str(1) + '.' + str(m) + '.jpg'
		cv2.imwrite(name, gray)
		m = m + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
