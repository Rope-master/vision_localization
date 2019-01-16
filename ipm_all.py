import numpy
import cv2
from math import sin, cos, pi
import sys
import random
import pdb
import glob
import os.path
import os

'''
src_roi = [x1, y1, x2, y2]
dst_sz = [h, w]
'''

output_w = 448
output_h = 448  #224 #448
output_w2 = 448
output_h2 = 448


def get_ipm_mat(yaw, pitch, src_roi, dst_sz, fx = 700.0, fy = 700.0, cx = 320, cy = 240, roll = 0):

    fx = 1.0*fx
    fy = 1.0*fy
    cx = 1.0*cx
    cy = 1.0*cy
    print "src_roi:", src_roi
    print "dst height:", dst_sz[0]
    print "dst weight:", dst_sz[1]
    print "cx: ", cx
    print "cy: ", cy
    print "fx: ", fx
    print "fy: ", fy
    print "cx/fx: ", 1.0*(cx/fx)

    print "yaw: ", yaw
    print "pitch: ", pitch
    print "roll: ", roll
    #### prepare the matrix
    c1 = cos(pitch * pi / 180)
    s1 = sin(pitch * pi / 180)

    c2 = cos(yaw * pi / 180)
    s2 = sin(yaw * pi / 180)

    c3 = cos(roll * pi / 180)
    s3 = sin(roll * pi / 180)

    pitch_mat = numpy.eye(3, 3)
    print "pitch_mat.dtype: ", pitch_mat.dtype
    pitch_mat[1:3,1:3] = [[c1, s1], [ -s1, c1]]
    print "pitch_mat: ", pitch_mat

    yaw_mat = numpy.eye(3, 3)
    yaw_mat[0:2,:] = [[c2, s2, 0], [-s2, c2, 0]]
    print "yaw_mat: ", yaw_mat

    roll_mat = numpy.eye(3, 3)
    roll_mat[0:3:2,:] = [[c3, 0, -s3], [s3, 0, c3]]
    print "roll_mat: ", roll_mat

    camera_to_world = numpy.eye(3, 3)
    camera_to_world[1:3,1:3] = [[0, -1], [1, 0]]
    print "camera_to_world: ", camera_to_world

    image_to_camera = numpy.eye(3, 3)
    image_to_camera[0, 0:3:2] = [1.0/fx, -cx/fx]
    image_to_camera[1, 1:3] = [1.0/fy, -cy/fy]
    print "image_to_camera: ", image_to_camera

    image_to_world = numpy.dot(numpy.dot(numpy.dot(numpy.dot(roll_mat, yaw_mat), pitch_mat), camera_to_world), image_to_camera)
    print "image_to_world: ", image_to_world

    #### remap the src and dst points
    src_pt = numpy.zeros([4, 2], dtype='float32')
    dst_pt = numpy.zeros([4, 2], dtype='float32')
    src_pt[...] = [[src_roi[0], src_roi[1]], [src_roi[2], src_roi[1]], [src_roi[2], src_roi[3]], [src_roi[0], src_roi[3]]]
    for i in range(src_pt.shape[0]):
        pt = numpy.ones([3, 1])
        pt[0:2, 0] = src_pt[i, :]
        pt = numpy.dot(image_to_world, pt)
        dst_pt[i, :] = [pt[0, 0]/pt[2, 0], pt[1, 0]/pt[2, 0]]

    dst_pt[:, 0] = (dst_pt[:, 0] - numpy.min(dst_pt[:, 0])) / (numpy.max(dst_pt[:, 0]) - numpy.min(dst_pt[:, 0])) * dst_sz[1]
    dst_pt[:, 1] = (dst_pt[:, 1] - numpy.min(dst_pt[:, 1])) / (numpy.max(dst_pt[:, 1]) - numpy.min(dst_pt[:, 1])) * dst_sz[0]

    #### get the final matrix
    trans = cv2.getPerspectiveTransform(src_pt, dst_pt)
    _, trans_inv = cv2.invert(trans)

    return trans, trans_inv


pitch = 36.4
deltax = 180
y1 = 240
y2 = 640
yaw = -19.7
roll = 2.2

fsx = 700.0
cx = 646.0
cy = 482.0


lines = os.listdir('src')
print lines;
f = open('trans.txt', 'r')
paras = f.readlines()
print paras
para = paras[1].split(',')
pitch = float(para[0])
yaw = float(para[1])
roll = float(para[2])
deltax = int(para[3])
y1 = int(para[4])
f.close()

print "%f,%f,%f,%d,%d\n"%(pitch, yaw, roll, deltax, y1)

ipm_flag = 0

for i in range(0, len(lines), 1):
    name_img = lines[i][-3:]
    if name_img != 'jpg':
        continue
    line = lines[i].strip()
    srcImg = cv2.imread('src/' + line)
    if ipm_flag == 1:
        break
    while 1:
        trans, trans_inv = get_ipm_mat(yaw, pitch, [480 - deltax, y1, 480 + deltax, y2], [output_h, output_w], fsx, fsx, cx, cy, roll)
        ipm = cv2.warpPerspective(srcImg, trans, (output_w, output_h))
        print "trans"
        print trans
        print "trans_inv"
        print trans_inv 
       
        text_show = "%f,%f,%f,%d,%d\n"%(pitch, yaw, roll, deltax, y1)
        cv2.putText(ipm, text_show, (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    #       cv2.imshow('source',im)
        cv2.imshow('ipm', ipm)
        cv2.imshow('source', srcImg)
        key = cv2.waitKey(0)
        if key == ord('q'):
            #f.write(text_show)
            #f.close()
            ipm_flag = 1
            break
        elif key == ord('a'):
            break
        elif key == ord('w'):
            pitch += 1
        elif key == ord('s'):
            pitch -= 1
        elif key == ord('e'):
            pitch += 0.1
        elif key == ord('d'):
            pitch -= 0.1
        elif key == ord('j'):
            deltax -= 10
        elif key == ord('l'):
            deltax += 10
        elif key == ord('i'):
            y1 -= 10
        elif key == ord('k'):
            y1 += 10
        elif key == ord('z'):
            break
        elif key == ord('m'):
            yaw -= 0.1
        elif key == ord('n'):
            yaw += 0.1
        elif key == ord('o'):
            roll += 0.1
        elif key == ord('p'):
            roll -= 0.1

for i in range(0, len(lines), 1):
    name_img = lines[i][-3:]
    print name_img
    if name_img != 'jpg':
        continue
    line = lines[i].strip()
    print "read image: ", line
    srcImg = cv2.imread('src/' + line)

    #trans, trans_inv = get_ipm_mat(yaw, pitch, [480 - deltax, y1, 480 + deltax, y2], [output_h, output_w],
    #                               fsx, fsx, cx, cy, roll)
    ipm = cv2.warpPerspective(srcImg, trans, (output_w, output_h))
    cv2.imwrite('ipm/' + line, ipm)
    cv2.waitKey(1)
