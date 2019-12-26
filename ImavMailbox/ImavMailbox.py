import libjevois as jevois
import numpy as np
import cv2
import cv2 as cv
from DetectMailbox import MailboxDetector

MARK_RED = 1
MARK_BLUE = 2
MARK_YELLOW = 3
MARK_ORANGE = 4

class ImavMailbox:

    def __init__(self):
        self.alt = 0 # in mm from AP

        # initial color thresholds
        self.mailbox_red = MailboxDetector([[163, 173, 0],[9, 255, 255]], 750) # split box
        self.mailbox_blue = MailboxDetector([[109, 176, 0],[145, 241, 255]], 1200)
        self.mailbox_yellow = MailboxDetector([[21, 195, 0],[45, 255, 255]], 1500)
        self.mailbox_orange = MailboxDetector([[141, 61, 0],[163, 76, 255]], 500)

        # cam params
        self.focal = (770., 770.)
        self.center = (320., 240.)
        self.calib_fisheye = None

        self.save = None # save current image

    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()
        self.processImage(img) # no need to process returned data

    def process(self, inframe, outframe):
        img = inframe.getCvBGR()
        detect = self.processImage(img)
        for mark in detect.values():
            box = cv2.boxPoints(mark)
            ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(img, [ctr], -1, (0, 255, 0), 4)
        outframe.sendCv(img)
        #outframe.sendCv(self.mailbox_yellow.mask)

    def processImage(self, img):
        '''
        process a single image
        return a dict with detected featured
        '''
        if self.save is not None:
            cv2.imwrite(self.save, img)
            jevois.LINFO(self.save)
            self.save = None

        detect = {} # dict of detected objects

        size_factor = None
        if self.alt > 1000:
            size_factor = self.focal[0] * self.focal[1] / (self.alt * self.alt)

        # red
        ret = self.mailbox_red.detect(img, size_factor)
        if ret is not None:
            detect[MARK_RED] = ret
            self.send_message(MARK_RED, ret)

        # blue
        ret = self.mailbox_blue.detect(img, size_factor)
        if ret is not None:
            detect[MARK_BLUE] = ret
            self.send_message(MARK_BLUE, ret)

        # yellow
        ret = self.mailbox_yellow.detect(img, size_factor)
        if ret is not None:
            detect[MARK_YELLOW] = ret
            self.send_message(MARK_YELLOW, ret)

        # orange (assuming only one in image)
        ret = self.mailbox_orange.detect(img, size_factor)
        if ret is not None:
            detect[MARK_ORANGE] = ret
            self.send_message(MARK_ORANGE, ret)

        return detect

    def send_message(self, mark, pos):
        '''
        send message over serial link to AP
        '''
        (u, v), (w, h), _ = pos
        x ,y = 0., 0.
        if self.calib_fisheye is None:
            # pos in "mm" in image frame
            x = 1000. * (u - self.center[0]) / self.focal[0]
            y = 1000. * (v - self.center[1]) / self.focal[1]
        else:
            pts_uv = np.array([[[u, v]]], dtype=np.float32)
            undist = cv2.fisheye.undistortPoints(pts_uv, self.calib_fisheye[0], self.calib_fisheye[1])
            x = 1000. * undist[0][0][0]
            y = 1000. * undist[0][0][1]
        jevois.sendSerial("N2 {} {:.2f} {:.2f} {:.2f} {:.2f}".format(mark, x, y, w, h))

    def parseSerial(self, cmd):
        str_list = cmd.split(' ')
        str_len = len(str_list)
        if str_len == 2 and str_list[0] == "alt" and str_list[1].isdigit():
            self.alt = int(str_list[1])
            return "OK"
        elif str_len == 2 and str_list[0] == "save":
            self.save = "/jevois/data/images/{}.png".format(str_list[1])
            return self.save
        elif str_len == 7 and str_list[0] == "hsv_red":
            h_min = [int(str_list[1]), int(str_list[2]), int(str_list[3])]
            h_max = [int(str_list[4]), int(str_list[5]), int(str_list[6])]
            self.mailbox_red.set_hsv_th(h_min, h_max)
            return "OK"
        elif str_len == 7 and str_list[0] == "hsv_blue":
            h_min = [int(str_list[1]), int(str_list[2]), int(str_list[3])]
            h_max = [int(str_list[4]), int(str_list[5]), int(str_list[6])]
            self.mailbox_blue.set_hsv_th(h_min, h_max)
            return "OK"
        elif str_len == 7 and str_list[0] == "hsv_yellow":
            h_min = [int(str_list[1]), int(str_list[2]), int(str_list[3])]
            h_max = [int(str_list[4]), int(str_list[5]), int(str_list[6])]
            self.mailbox_yellow.set_hsv_th(h_min, h_max)
            return "OK"
        elif str_len == 7 and str_list[0] == "hsv_orange":
            h_min = [int(str_list[1]), int(str_list[2]), int(str_list[3])]
            h_max = [int(str_list[4]), int(str_list[5]), int(str_list[6])]
            self.mailbox_orange.set_hsv_th(h_min, h_max)
            return "OK"
        elif str_len == 5 and str_list[0] == "calib":
            self.focal = (float(str_list[1]), float(str_list[2]))
            self.center = (float(str_list[3]), float(str_list[4]))
            return "OK"
        elif str_len == 9 and str_list[0] == "calib_fisheye":
            K = np.array([[float(str_list[1]), 0., float(str_list[3])], [0., float(str_list[2]), float(str_list[4])], [0., 0., 1.]])
            D = np.array([[float(str_list[5])], [float(str_list[6])], [float(str_list[7])], [float(str_list[8])]])
            self.calib_fisheye = (K, D)
            return "OK"
        return "ERR"

    def supportedCommands(self):
        return "alt - set alt in mm"

