import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import math

class Vocabulary:
    def __init__(self, k):
        self.k = k
        self.vocabulary = None

    def generateBoW(self, path, random_state):
        """

        通过聚类算法，生成词袋模型

        :param path:词袋模型存储路径

        :param self.k: 词袋模型中视觉词汇的个数

        :param random_state: 随机数种子

        :return: 词袋模型视觉词汇矩阵

        """
        print("Generating BoW ...")
        featureSet = np.load(path)
        np.random.shuffle(featureSet)
        km = MiniBatchKMeans(n_clusters=self.k, random_state=random_state, batch_size=200).fit(featureSet)
        centers = km.cluster_centers_
        self.vocabulary = centers
        np.save("Bow.npy", centers)
        return centers

    def getBow(self, path):
        """

        读取词袋模型文件

        :param path: 词袋模型文件路径

        :return: 词袋模型矩阵

        """
        centers = np.load(path)
        self.vocabulary = centers
        return centers

    def calSPMFeature(self, features, keypoints, center, img_x, img_y, numberOfBag):
        '''
        使用 SPM 算法，生成不同尺度下图片对视觉词汇的投票结果向量
        :param features:图片的特征点向量
        :param keypoints: 图片的特征点列表
        :param center: 词袋中的视觉词汇的向量
        :param img_x: 图片的宽度
        :param img_y: 图片的高度
        :param self.k: 词袋中视觉词汇的个数
        :return: 基于 SPM 思想生成的图片视觉词汇投票结果向量
        '''
        size = len(features)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(center)
        nearest_center = neigh.kneighbors(features, return_distance=False).ravel()
        histogramOfLevelZero = np.zeros((1, numberOfBag))
        histogramOfLevelOne = np.zeros((4, numberOfBag))
        histogramOfLevelTwo = np.zeros((16, numberOfBag))
        blo2_x = math.ceil(img_x / 2)
        blo2_y = math.ceil(img_y / 2)
        blo4_x = math.ceil(img_x / 4)
        blo4_y = math.ceil(img_y / 4)
        for i in range(size):
            x, y = keypoints[i].pt
            histogramOfLevelZero[0][nearest_center[i]] += 1
            histogramOfLevelOne[math.floor(y / blo2_y) * 2 + math.floor(x / blo2_x)][nearest_center[i]] += 1
            histogramOfLevelTwo[math.floor(y / blo4_y) * 4 + math.floor(x / blo4_x)][nearest_center[i]] += 1
        result = np.float32([]).reshape(0, numberOfBag)
        result = np.append(result, histogramOfLevelZero * 0.25, axis=0)
        result = np.append(result, histogramOfLevelOne * 0.25, axis=0)
        result = np.append(result, histogramOfLevelTwo * 0.5, axis=0)

        return result

    def Imginfo2SVMdata(self, data):
        """

        将图片特征点数据转化为 SVM 训练的投票向量

        :param self.vocabulary: 词袋模型

        :param data: 图片特征点数据 imgInfo

        :param self.k: 词袋模型中视觉词汇的数量

        :return: 投票向量矩阵，图片标签

        """
        print("Converting imginfo to SVMDate ...")
        dataset = np.float32([]).reshape(0, self.k * 21)
        labels = []

        for imginfo in data:
            spm = self.calSPMFeature(imginfo.descriptors, imginfo.keypoints, self.vocabulary, imginfo.width, imginfo.height, self.k)
            spm = spm.ravel().reshape((1, self.k*21))
            dataset = np.append(dataset, spm, axis=0)
            labels.append(imginfo.label)

        labels = np.array(labels)

        return dataset, labels

