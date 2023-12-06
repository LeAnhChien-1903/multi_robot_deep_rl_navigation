import cv2

image = cv2.imread("empty_20.png")

cv2.imwrite("empty_20.png", cv2.resize(image, (1200, 1200)))