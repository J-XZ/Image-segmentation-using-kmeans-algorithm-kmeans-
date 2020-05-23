from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.cluster import KMeans


class MyKMeans:

    @staticmethod
    def get_dis(a, b):
        """
        获取两点之间的距离
        """
        return np.sum(np.power(a - b, 2))

    @staticmethod
    def show():
        """显示画图结果"""
        plt.show()

    def get_standardized_image(self):
        """
        :return:将图片转换为总像素个数行，三列的数据集矩阵
        """
        return self.pic.reshape(self.pic.shape[0] * self.pic.shape[1], 3)

    def draw_data(self, title):
        """
        绘制整个图片的rgb空间分布散点图
        :return:绘图句柄
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title(title)
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], alpha=0.5, c=self.data / 255)

    def draw_pic(self, title):
        """绘制整个图片"""
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(self.pic)
        ax.set_title(title)
        return ax

    def get_centers(self):
        """随机生成聚类中心"""
        d = np.unique(self.data, axis=0)
        index = np.random.randint(0, d.shape[0], size=(1, self.k))
        ret = d[index][0]
        return ret, np.zeros(ret.shape)

    def main_k_means(self, allow_dis=1):
        """
        执行k均值聚类
        :param allow_dis:两次迭代之间允许的最大误差平方和
        """

        def get_change():
            return np.sum(np.power(self.centers - self.last, 2))

        cnt = 1
        while get_change() > allow_dis:
            print("第" + str(cnt) + "次迭代")
            cnt += 1
            belong = np.array([[0, 0, 0] for i in range(self.k)])
            num = [0 for i in range(self.k)]
            for point in self.data:
                to_which = self.get_belong(point)
                belong[to_which] += point
                num[to_which] += 1
            num = np.array(num).reshape(-1, 1)
            belong = belong / num
            self.last = self.centers
            self.centers = belong

    def get_belong(self, point):
        """
        根据聚类结果返回当前像素点属于哪个聚类
        :param point:待分类像素点
        :return: 像素点所属的聚类编号
        """
        dis = []
        for center in self.centers:
            dis.append(self.get_dis(point, center))
        dis = np.array(dis)
        return np.argsort(dis)[0]

    def draw_division(self, title):
        """
        绘制聚类结果
        """
        ax = self.draw_pic(title)
        x, y = np.array([i for i in range(self.pic.shape[1])]), np.array([i for i in range(self.pic.shape[0])])
        x, y = np.meshgrid(x, y)
        c = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                c[i][j] = self.get_belong(self.pic[i][j].reshape(1, -1))

        ax.contour(x, y, c)
        ax = self.draw_pic(title)
        ax.contourf(x, y, c, cmap='gray')

    def use_sk_learn(self, title):
        """
        使用内置sk_learn模块kmeans函数完成区块划分并绘图
        :param title:绘图标题
        """
        data = self.data
        y_pred = KMeans(n_clusters=self.k, random_state=None).fit_predict(data)
        y_pred = y_pred.reshape(self.pic.shape[0], self.pic.shape[1])
        ax = self.draw_pic(title)
        x, y = np.array([i for i in range(self.pic.shape[1])]), np.array(
            [i for i in range(self.pic.shape[0])])
        x, y = np.meshgrid(x, y)
        ax.contour(x, y, y_pred)
        ax = self.draw_pic(title)
        ax.contourf(x, y, y_pred, cmap='gray')

    def __init__(self, filename, k=4, allow_dis=1):
        """
        :param filename:图片文件的名字
        :param k:将图片分为多少个区块
        :param allow_dis:允许的两次迭代之间的最大平方误差和
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.pic = np.array(Image.open(filename))
        self.pic = self.pic[:, :, 0:3]
        self.k = k
        self.data = self.get_standardized_image()
        self.centers, self.last = self.get_centers()
        self.main_k_means(allow_dis)
