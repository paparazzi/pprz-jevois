import cv2
from DetectMailbox import MailboxDetector
import time
import numpy as np
import sys

video = None
image = None
if len(sys.argv) == 2:
    if sys.argv[1].isdigit():
        video = cv2.VideoCapture(int(sys.argv[1]))
        time.sleep(2)
        video.set(3,640) #width
        video.set(4,480) #height
        video.set(5,8.3) #fps
    else:
        image = sys.argv[1]
else:
    print("input video number or image file missing")
    exit(1)

mailbox_red = MailboxDetector([[163, 173, 0],[9, 255, 255]], color="RED")
mailbox_blue = MailboxDetector([[109, 176, 0],[145, 241, 255]], color="BLUE")
mailbox_yellow = MailboxDetector([[21, 195, 0],[45, 255, 255]], color="YELLOW")

while True:
    img = None
    if video is not None:
        ret, img = video.read()
    else:
        img = cv2.imread(image)

    res = mailbox_red.detect(img)
    if res is not None:
        box = cv2.cv.BoxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    res = mailbox_blue.detect(img)
    if res is not None:
        box = cv2.cv.BoxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    res = mailbox_yellow.detect(img)
    if res is not None:
        box = cv2.cv.BoxPoints(res)
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)

    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if video is not None:
    video.release()
cv2.destroyAllWindows()

