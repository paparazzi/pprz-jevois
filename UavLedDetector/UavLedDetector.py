#
# Copyright (C) 2017 Gautier Hattenberger <gautier.hattenberger@enac.fr>
#
# This file is part of paparazzi.
#
# paparazzi is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# paparazzi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with paparazzi; see the file COPYING.  If not, see
# <http://www.gnu.org/licenses/>.
#

'''
Basic detector for a small UAV equipped with a bright white LED.

This is returning the position and the area of the detected shape in image plane.
It can be used to estimate the relative position and distance.
If their are bright static elements in the scene, they can be masked before the flight.
'''

import libjevois as jevois
import cv2
import numpy as np

class UavLedDetector:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # request to create a mask on first frame
        self.set_mask = True
        self.mask = None
        self.x = 0.
        self.y = 0.
        self.area = 0.
        self.threshold = 200
        
        # A simple frame counter used to demonstrate sendSerial():
        self.frame = 0
        
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sample", 100, jevois.LOG_INFO)

    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Get the next camera image (may block until it is captured) and convert it to OpenCV GRAY:
        img = inframe.getCvBGR()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = gray.shape
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,th = cv2.threshold(blur,self.threshold,255,cv2.THRESH_BINARY)
        if self.set_mask:
            self.mask = cv2.dilate(th, np.ones((10,10), np.uint8), iterations=1)
            self.mask = cv2.bitwise_not(self.mask)
            self.set_mask = False
        th_masked = cv2.bitwise_and(th, th, mask=self.mask)
        image, contours, hierarchy = cv2.findContours(th_masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ## rotated rectangle (min area)
            rect = cv2.minAreaRect(cnt)
            ((x, y), (w, h), _) = rect
            if abs(w - h) > 10: # not square
                continue
            self.x = x-self.width/2
            self.y = -(y-self.height/2)
            self.area = w * h
            jevois.sendSerial("POS {:.2f} {:.2f} {:.2f}".format(self.x, self.y, self.area))

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and convert it to OpenCV GRAY:
        img = inframe.getCvBGR()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

        # Get image width, height, channels in pixels. Beware that if you change this module to get img as a grayscale
        # image, then you should change the line below to: "height, width = img.shape" otherwise numpy will throw. See
        # how it is done in the PythonOpenCv module of jevoisbase:
        self.height, self.width = gray.shape
        jevois.LINFO("{} {}".format(self.height, self.width))

        # Draw red cross
        cv2.line(img,(int(self.width/2),int(self.height*0.1)), (int(self.width/2),int(self.height*0.9)), (0,0,255),2)
        cv2.line(img,(int(self.width*0.1),int(self.height/2)), (int(self.width*0.9),int(self.height/2)), (0,0,255),2)

        #if self.mask is None:
        #    self.mask = np.zeros(gray.shape, np.uint8)

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,th = cv2.threshold(blur,self.threshold,255,cv2.THRESH_BINARY)
        if self.set_mask:
            self.mask = cv2.dilate(th, np.ones((10,10), np.uint8), iterations=1)
            self.mask = cv2.bitwise_not(self.mask)
            self.set_mask = False
        th_masked = cv2.bitwise_and(th, th, mask=self.mask)
        image, contours, hierarchy = cv2.findContours(th_masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ## rotated rectangle (min area)
            rect = cv2.minAreaRect(cnt)
            ((x, y), (w, h), _) = rect
            if abs(w - h) > 10: # not square
                continue
            self.x = x-self.width/2
            self.y = -(y-self.height/2)
            self.area = w * h
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(img, [box], 0, (0,255,0), 3)
            jevois.sendSerial("POS {:.2f} {:.2f} {:.3f} {:.3f} {:.4f} {}".format(self.x, self.y, w, h, self.area, self.frame))
            
        # Write frames/s info from our timer (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        c = (255,255,255)
        if len(contours) > 0:
            c = (0,0,255)
        cv2.putText(img, fps, (3, self.height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    
        # Convert our image to video output format and send to host over USB:
        outframe.sendCv(img)
         
        # Send a string over serial (e.g., to an Arduino). Remember to tell the JeVois Engine to display those messages,
        # as they are turned off by default. For example: 'setpar serout All' in the JeVois console:
        #jevois.sendSerial("DONE frame {}".format(self.frame));
        self.frame += 1

    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    # This function is optional and only needed if you want your module to handle custom commands. Delete if not needed.
    def parseSerial(self, str):
        print("parseserial received command [{}]".format(str))
        str_list = str.split(' ')
        if str == "set_mask":
            self.set_mask = True
            return("Mask set")
        elif str == "clear_mask":
            self.mask = np.zeros((self.height, self.width), np.uint8)
            self.mask = cv2.bitwise_not(self.mask)
            return("Mask cleared")
        elif len(str_list) == 2 and str_list[0] == "set_thres" and str_list[1].isdigit():
            self.threshold = max(0, min(255, int(str_list[1])))
            return("Threshold set")
        return "ERR: Unsupported command"
    
    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    # This function is optional and only needed if you want your module to handle custom commands. Delete if not needed.
    def supportedCommands(self):
        # use \n seperator if your module supports several commands
        return "set_mask - hide visible objects from the scene\nclear_mask - clear all mask (show all objects)"

