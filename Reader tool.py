import cv2
import numpy as np

image = cv2.imread('barcode.jpg', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0, -1)

scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1, -1)


gradient = cv2.subtract(scharr_x, scharr_y)
gradient = cv2.convertScaleAbs(gradient)

blurred = cv2.blur(gradient, (4,5))
_, thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_TOZERO)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

closed = cv2.erode(closed, None, iterations=5)
closed = cv2.dilate(closed, None, iterations=5)

cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(image, [box], -1, (0, 255, 255), 3)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyWindow('image')