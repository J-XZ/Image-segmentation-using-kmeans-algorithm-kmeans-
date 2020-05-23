from myfunc import *

if __name__ == '__main__':
    # k_means = MyKMeans(filename="d🎨.png", k=4, allow_dis=1)
    # k_means = MyKMeans(filename="🐻.jpg", k=50, allow_dis=1)
    k_means = MyKMeans(filename="🦅.jpg", k=2, allow_dis=1)
    k_means.draw_pic("原始图像")
    k_means.draw_data("rgb三维分布图")
    k_means.draw_division("区块划分图")
    k_means.use_sk_learn("使用内置kmeans函数")
    k_means.show()
