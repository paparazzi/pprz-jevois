import libjevois as jevois
import cv2
import numpy as np
from detector import Detector
 
## Simple face detection using OpenCV in Python on JeVois
#
# This module by default simply converts the input image to a grayscale OpenCV image, and then applies the Canny
# edge detection algorithm. Try to edit it to do something else (note that the videomapping associated with this
# module has grayscale image outputs, so that is what you should output).
#
# @author Guido Manfredi
# 
# @displayname Face Detection
# @videomapping YUYV 640 480 20.0 YUYV 640 480 20.0 JeVois FaceDetect
# @email guido.manfredi\@enac.fr
# @address 7 Avenue Edouard Belin, 31400 Toulouse, France
# @copyright Copyright (C) 2017 by Guido Manfredi, UAS lab at ENAC
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class FaceDetect:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        jevois.LINFO("Init FaceDetect")
        self.timer = jevois.Timer("FaceDetect", 100, jevois.LOG_INFO)
        self.got_face = False

        self.face_detector = Detector('/jevois/share/facedetector/haarcascade_frontalface_alt.xml')
        #self.face_detector = Detector('/jevois/share/facedetector/lbpcascade_frontalface.xml')
        self.faces = np.empty(shape=(0,0))

    # ###################################################################################################
    ## Process function without USB output
    def processNoUSB(self, inframe):
        # call process without outframe
        self.process(inframe)

    # ###################################################################################################
    ## Process function with or without USB output
    def process(self, inframe, outframe = None):
        jevois.LINFO("Processing faces")

        self.timer.start()
        inimg = inframe.getCvGRAY()
        #inimg.require("input", 640, 480, V4L2_PIX_FMT_GREY);
        equimg = cv2.equalizeHist(inimg)
        if outframe is not None:
            outimg = outframe.get()
            jevois.pasteGreyToYUYV(inimg, outimg, 0, 0)
        #inframe.done()

        # eyes will not be used since eyes detection is disabled by default (but here since not returned null)
        self.faces = self.face_detector.detect(equimg)

        self.got_face = False
        (self.x, self.y, self.w, self.h) = (0, 0, 0, 0)
        for (x,y,w,h) in self.faces:
            # store face if first found or bigger
            if (self.got_face == False) or (w * h > self.w * self.h):
                (self.x, self.y, self.w, self.h) = (x, y, w, h)
                self.got_face = True
                jevois.LINFO("Found faces {} {} ({}, {})".format(x, y, w, h))
            # draw rectangle if outframe requested
            if outframe is not None:
                jevois.drawRect(outimg, int(x), int(y), int(w), int(h), 2, 0x80ff)

        fps = self.timer.stop()

        if outframe is not None:
            outframe.send()

        # Communication over serial, sending a string
        if self.got_face:
            height, width = equimg.shape[:2]
            jevois.LINFO("FPS {}".format(fps))
            x = self.x + self.w/2. - width/2.
            y = self.y + self.h/2. - height/2.
            jevois.sendSerial("T2 {} {}".format(int(x * 2000./width), int(-y * 2000./height)))

    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    def parseSerial(self, str):
        return "ERR: Unsupported command"

    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    def supportedCommands(self):
        return ""
