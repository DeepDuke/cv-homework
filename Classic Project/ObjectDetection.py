"""
Classic Project:
B. Object Detection

I use two figures which both contain my kindle to do sift matching.
The matching result is result.jpg.
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10
img1 = cv2.imread('1.jpg')  # queryImage
img2 = cv2.imread('2.jpg')  # trainImage
# change RGB to BGR channels for opencv uses BGR
img1 = np.array(img1[:, :, ::-1])
img2 = np.array(img2[:, :, ::-1])
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []

for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print('Matrix:\n', M)
    matchesMask = mask.ravel().tolist()
    h, w, d = img1.shape
    # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # draw bounding box in second figure
    points = [(110, 3050), (140, 1100), (2940, 1100), (2970, 3090)]  # four points of my kindle
    box = []  # four angle points of bounding box in in second figure
    for point in points:
        point = list(point)
        point.append(1)
        point = np.array(point)
        new_point = M.dot(point)
        new_point = np.array(list(map(int, new_point)))
        new_point = tuple(new_point[0:2])  # first two element
        box.append(new_point)
        print(new_point)
    # draw bounding box of kindle in the first figure
    points = np.array(points)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(img1, [points], True, (255, 0, 0), 40)
    # draw bounding box of kindle in the second figure
    box = np.array(box)
    box = box.reshape((-1, 1, 2))
    cv2.polylines(img2, [box], True, (255, 0, 0), 40)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3)

plt.show()
img3 = np.array(img3[:, :, ::-1])  # change from BRG to RGB
cv2.imwrite('result.jpg', img3)

