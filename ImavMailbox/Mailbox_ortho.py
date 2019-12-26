#!/usr/bin/python3
import cv2
from DetectMailbox import MailboxDetector
import numpy as np

def boxPoints(pts):
    if int(cv2.__version__[0]) >= 3:
        return cv2.boxPoints(pts)
    else:
        return cv2.cv.BoxPoints(pts)

DEFAULT_IMAGE_VIEWER = "gwenview"
DEFAULT_IMAGE_OUTPUT = "out_detect.png"
DEFAULT_SCALE_FACTOR = 4
DEFAULT_RESOLUTION = 20 # pixels per meter

mailbox_red = MailboxDetector([[163, 173, 0],[9, 255, 255]], 750, color="RED")
mailbox_blue = MailboxDetector([[103, 129, 0],[129, 190, 255]], 1200, color="BLUE")
mailbox_yellow = MailboxDetector([[21, 195, 0],[45, 255, 255]], 1500, color="YELLOW")
#mailbox_yellow = MailboxDetector([[0, 0, 0],[179, 6, 255]], 1500, aspect_ratio_th=0.6, color="YELLOW") # for test image only
mailbox_orange = MailboxDetector([[141, 61, 0],[163, 76, 255]], 500, color="ORANGE")

def get_geo_data(filename):
    import os
    if os.path.splitext(filename)[1] == '.tif':
        try:
            from osgeo import gdal
            # Open tif file
            ds = gdal.Open(filename)
            # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
            return ds.GetGeoTransform()
        except:
            print('failed loading gdal')
            return None
    else:
        print('not a tif file')
        return None

def transform_utm_to_wgs84(easting, northing, zone=31):
    try:
        import osr
        utm_coordinate_system = osr.SpatialReference()
        utm_coordinate_system.SetWellKnownGeogCS("WGS84") # Set geographic coordinate system to handle lat/lon
        is_northern = northing > 0    
        utm_coordinate_system.SetUTM(zone, is_northern)
        wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS() # Clone ONLY the geographic coordinate system 
        # create transform component
        utm_to_wgs84_transform = osr.CoordinateTransformation(utm_coordinate_system, wgs84_coordinate_system) # (<from>, <to>)
        return utm_to_wgs84_transform.TransformPoint(easting, northing, 0) # returns lon, lat, altitude
    except:
        return None

def pixel2coord(x, y, geo):
    """Returns global coordinates from pixel x, y coords"""
    if geo is not None:
        xoff, a, b, yoff, d, e = geo
        xp = a * x + b * y + xoff
        yp = d * x + e * y + yoff
        return transform_utm_to_wgs84(xp, yp)
    else:
        return None

def process_result(img, out, res, label, geo=None):
    center = (int(res[0][0]), int(res[0][1]))
    cv2.circle(out, center, 50, (0, 255, 0), 5)
    cv2.putText(out, label, (center[0]+60, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
    coord = pixel2coord(center[0], center[1], geo)
    if coord is not None:
        # print lat lon if available
        cv2.putText(out, '{:.7f} {:.7f}'.format(coord[1], coord[0]), (center[0]+60, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
    box = boxPoints(res)
    ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
    mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.drawContours(mask, [ctr], -1, (255,255,255),-1)
    img = cv2.bitwise_and(img,img,mask = cv2.bitwise_not(mask))
    return img, out

def find_mailboxes(img, output=None, scale=DEFAULT_SCALE_FACTOR, res=DEFAULT_RESOLUTION, geo=None):
    out = img.copy()

    scale_factor = pow(res / 1000., 2)

    res = mailbox_red.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "RED", geo)

    res = mailbox_blue.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "BLUE", geo)

    res = mailbox_yellow.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "YELLOW", geo)

    res = mailbox_orange.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "ORANGE_1", geo)

    res = mailbox_orange.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "ORANGE_2", geo)

    res = mailbox_orange.detect(img, scale_factor)
    if res is not None:
        img, out = process_result(img, out, res, "ORANGE_3", geo)

    if output is None:
        w, h, _ = img.shape
        img_out = cv2.resize(out, (int(h/scale),int(w/scale)))
        cv2.imshow('frame',img_out)
        while True:
            if cv2.waitKey(-1)  & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(output, out)

if __name__ == '__main__':
    '''
    When used as a standalone script
    '''
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Search mailboxes in image")
    parser.add_argument('img', help="image path")
    parser.add_argument("-v", "--viewer", help="program used to open the image", default=DEFAULT_IMAGE_VIEWER)
    parser.add_argument("-nv", "--no_view", help="Do not open image after processing", action='store_true')
    parser.add_argument("-o", "--output", help="output file name", default=None)
    parser.add_argument("-s", "--scale", help="resize scale factor", type=int, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("-r", "--resolution", help="resolution in pixels per meter", type=float, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    img = cv2.imread(args.img)
    geo = get_geo_data(args.img)
    find_mailboxes(img, args.output, args.scale, args.resolution, geo)

    if not args.no_view and args.output is not None:
        subprocess.call([args.viewer, args.output])

