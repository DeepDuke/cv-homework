# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:40:05 2019

@author: 28954
"""
import cv2
import random
import numpy as np


# change color
def random_light_color(img):
    B, G, R = cv2.split(img)    
    # change B
    B = B.astype('int')
    b_rand = random.randint(-50, 50)
    B = B + b_rand
    B[B > 255] = 255
    B[B < 0] = 0
    B = B.astype('uint8')
    # change G
    G = G.astype('int')
    g_rand = random.randint(-50, 50)
    G = G + g_rand
    G[G > 255] = 255
    G[G < 0] = 0
    G = G.astype('uint8')
    # change R
    R = R.astype('int')
    r_rand = random.randint(-50, 50)
    R = R + r_rand
    R[R > 255] = 255
    R[R < 0] = 0
    R = R.astype('uint8')
    # merge B, G, R
    img_merge = cv2.merge((B, G, R))
    return img_merge
# rotation transform
def rotate(img):
    angle = random.randint(0, 361)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1) 
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate
# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)
    
    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
# augment using changing color, rotation and perspective transform
def augment(img):
    img = random_light_color(img)
    img = rotate(img)
    M_warp, img =random_warp(img, img.shape[0], img.shape[1])
    return img

if __name__ == '__main__':
    img = cv2.imread('Lena.png')
    # generate 30 aug_img
    for i in range(30):
        aug_img = augment(img)
        cv2.imwrite('aug_img'+str(i)+'.png', aug_img)
        cv2.imshow('aug_img' + str(i), aug_img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()