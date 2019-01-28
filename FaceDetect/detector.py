#!/usr/bin/python
import sys
import cv2

class Detector:
    def __init__(self, cascade_path=""):
        #os.path.dirname(os.path.realpath(__file__))
        if not cascade_path:
            print("Detector: no cascade path, trying default location for face cascade")
            cascade_path = '/jevois/share/facedetector/haarcascade_frontalface_alt.xml'
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if(self.classifier.empty()):
            print("Detector: error loading cascade file " + cascade_path)
                
    def detect(self, gray):
        if (len(gray) != 0):
            faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE) # cv2.cv.CV_HAAR_SCALE_IMAGE in older OpenCV
        else:
            print("Empty image...")
        return faces


if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Cascade path required")
        exit()

    # /usr/share/jevois-opencv-3.3.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml
    cascade_path = str(sys.argv[1])
    face_detector = Detector(cascade_path)

    # /home/guido/Pictures/faces.jpg
    img_path = str(sys.argv[2])
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face detection
    faces = face_detector.detect(gray)
    print ("Faces detected: " + str(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)


    
