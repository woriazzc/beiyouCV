import cv2
from ImageInfo import ImageInfo
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("test.jpg")
    desc, kp = ImageInfo(img, None).getImgFeature()
    img2 = cv2.drawKeypoints(img, kp, img)
    print(desc)
    cv2.imshow("1", img)
    cv2.waitKey()
    fr = np.hstack((cv2.imread("test.jpg"), img2))
    cv2.imwrite("ttttt.jpg", fr)