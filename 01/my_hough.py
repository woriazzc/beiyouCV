'''

'''

import numpy as np
import math


class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print('Hough_transform_algorithm')

        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0:
                    r = 0
                    u = i
                    v = j
                    while r < self.radius and u < self.y and v < self.x and u >= 0 and v >= 0:
                        self.vote_matrix[math.floor(u / self.step)][math.floor(v / self.step)][
                            math.floor(r / self.step)] += 1
                        v += self.step
                        u += self.step * self.angle[i][j]
                        r += math.sqrt(self.step ** 2 + (self.step * self.angle[i][j]) ** 2)
                    r = 0
                    u = i
                    v = j
                    while r < self.radius and u < self.y and v < self.x and u >= 0 and v >= 0:
                        if r > 0:
                            self.vote_matrix[math.floor(u / self.step)][math.floor(v / self.step)][
                                math.floor(r / self.step)] += 1
                        v -= self.step
                        u -= self.step * self.angle[i][j]
                        r += math.sqrt(self.step ** 2 + (self.step * self.angle[i][j]) ** 2)

        return self.vote_matrix

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print('Select_Circle')

        cirs = []
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for k in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][k] >= self.threshold:
                        cirs.append([math.ceil(j * self.step + self.step / 2), math.ceil(i * self.step + self.step / 2),
                                     math.ceil(k * self.step + self.step / 2)])
        if len(cirs) == 0:
            print("No Circle")
            return
        tmp = []
        cu = []
        x, y, r = cirs[0]
        for e in cirs:
            if abs(x - e[0]) <= 20 and abs(y - e[1]) <= 20 and abs(r - e[2]) <= 10:
                tmp.append(e)
            else:
                cu.append(np.array(tmp).mean(axis=0))
                x, y, r = e
                tmp.clear()
                tmp.append(e)
        cu.append(np.array(tmp).mean(axis=0))

        tmp.clear()
        x, y, r = cu[0]
        for e in cu:
            if abs(x - e[0]) <= 20 and abs(y - e[1]) <= 20 and abs(r - e[2]) <= 20:
                tmp.append(e)
            else:
                self.circles.append(np.array(tmp).mean(axis=0))
                x, y, r = e
                tmp.clear()
                tmp.append(e)
        self.circles.append(np.array(tmp).mean(axis=0))
        for e in self.circles:
            print("Circle core: (%f, %f)  Radius: %f" % (e[0], e[1], e[2]))

    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles

    # def Hough_transform_algorithm(self):
    #     '''
    #     按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
    #     元进行投票。每个点投出来结果为一折线。
    #     :return:  投票矩阵
    #     '''
    #     print('Hough_transform_algorithm')
    #
    #     self.angle = np.arctan(self.angle)
    #     for i in range(1, self.y - 1):
    #         for j in range(1, self.x - 1):
    #             if self.img[i][j] > 0:
    #                 r = 0
    #                 u = i
    #                 v = j
    #                 while r < self.radius and u < self.y and v < self.x and u >= 0 and v >= 0:
    #                     self.vote_matrix[math.floor(u / self.step)][math.floor(v / self.step)][
    #                         math.floor(r / self.step)] += 1
    #                     tmp = self.step - (r % self.step)
    #                     if np.sin(self.angle[i][j]) > 0:
    #                         tmp = min(tmp, math.floor((self.step - u % self.step) / np.sin(self.angle[i][j])))
    #                     elif np.sin(self.angle[i][j]) < 0:
    #                         tmp = min(tmp, math.floor((u % self.step + 1) / abs(np.sin(self.angle[i][j]))))
    #                     if np.cos(self.angle[i][j]) > 0:
    #                         tmp = min(tmp, math.floor((self.step - v % self.step) / np.cos(self.angle[i][j])))
    #                     elif np.cos(self.angle[i][j]) < 0:
    #                         tmp = min(tmp, math.floor((v % self.step + 1) / abs(np.cos(self.angle[i][j]))))
    #                     r += tmp
    #                     u += math.ceil(tmp * np.sin(self.angle[i][j]))
    #                     v += math.ceil(tmp * np.cos(self.angle[i][j]))
    #                 r = 0
    #                 u = i
    #                 v = j
    #                 self.angle[i][j] += np.pi
    #                 while r < self.radius and u < self.y and v < self.x and u >= 0 and v >= 0:
    #                     if r > 0:
    #                         self.vote_matrix[math.floor(u / self.step)][math.floor(v / self.step)][
    #                             math.floor(r / self.step)] += 1
    #                     tmp = self.step - (r % self.step)
    #                     if np.sin(self.angle[i][j]) > 0:
    #                         tmp = min(tmp, math.floor((self.step - u % self.step) / np.sin(self.angle[i][j])))
    #                     elif np.sin(self.angle[i][j]) < 0:
    #                         tmp = min(tmp, math.floor((u % self.step + 1) / abs(np.sin(self.angle[i][j]))))
    #                     if np.cos(self.angle[i][j]) > 0:
    #                         tmp = min(tmp, math.floor((self.step - v % self.step) / np.cos(self.angle[i][j])))
    #                     elif np.cos(self.angle[i][j]) < 0:
    #                         tmp = min(tmp, math.floor((v % self.step + 1) / abs(np.cos(self.angle[i][j]))))
    #                     r += tmp
    #                     u += math.ceil(tmp * np.sin(self.angle[i][j]))
    #                     v += math.ceil(tmp * np.cos(self.angle[i][j]))
    #                 self.angle[i][j] -= np.pi
    #
    #     return self.vote_matrix
