'''

'''
import cv2
import numpy as np


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print('Get_gradient_img')

        self.img_origin = self.img.copy()
        for i in range(0, self.y):
            for j in range(0, self.x):
                if j == 0:
                    dx = 1
                else:
                    dx = np.dot(self.img_origin[i, j - 1:j + 1], self.x_kernal.flatten())
                if i == 0:
                    dy = 1
                else:
                    dy = np.dot(self.img_origin[i - 1:i + 1, j], self.y_kernal.flatten())
                self.img[i][j] = np.sqrt(dx * dx + dy * dy)
                self.angle[i][j] = np.arctan2(dy, dx)

        self.angle = np.tan(self.angle)
        return self.img

    def Non_maximum_suppression(self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print('Non_maximum_suppression')

        res = np.zeros([self.y, self.x])
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] <= 4:
                    res[i][j] = 0
                    continue
                if abs(self.angle[i][j]) > 1:
                    g2 = self.img[i - 1][j]
                    g4 = self.img[i + 1][j]
                    if self.angle[i][j] > 0:
                        g1 = self.img[i - 1][j - 1]
                        g3 = self.img[i + 1][j + 1]
                    else:
                        g1 = self.img[i - 1][j + 1]
                        g3 = self.img[i + 1][j - 1]
                else:
                    g2 = self.img[i][j - 1]
                    g4 = self.img[i][j + 1]
                    if self.angle[i][j] > 0:
                        g1 = self.img[i - 1][j - 1]
                        g3 = self.img[i + 1][j + 1]
                    else:
                        g1 = self.img[i + 1][j - 1]
                        g3 = self.img[i - 1][j + 1]
                tmp1 = abs(self.angle[i][j]) * g1 + (1 - abs(self.angle[i][j])) * g2
                tmp2 = abs(self.angle[i][j]) * g3 + (1 - abs(self.angle[i][j])) * g4
                if tmp1 > self.img[i][j] or tmp2 > self.img[i][j]:
                    res[i][j] = 0
                else:
                    res[i][j] = self.img[i][j]
        self.img = res

        return self.img

    def check(self, i, j):
        if self.img_origin[i][j] > self.HT_low_threshold and self.img_origin[i][j] < self.HT_high_threshold:
            self.img[i][j] = self.HT_high_threshold

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print('Hysteresis_thresholding')

        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img_origin[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:
                        self.check(i - 1, j)
                        self.check(i + 1, j)
                        if self.angle[i][j] < 0:
                            self.check(i - 1, j - 1)
                            self.check(i + 1, j + 1)
                        else:
                            self.check(i - 1, j + 1)
                            self.check(i + 1, j - 1)
                    else:
                        self.check(i, j - 1)
                        self.check(i, j + 1)
                        if self.angle[i][j] < 0:
                            self.check(i - 1, j - 1)
                            self.check(i + 1, j + 1)
                        else:
                            self.check(i + 1, j - 1)
                            self.check(i - 1, j + 1)

        return self.img

    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img
