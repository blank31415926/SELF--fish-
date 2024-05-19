"""
File Name: __init__.py
Author: Blank
Date: 2024/5/15
Description: SELF文件入口
"""



from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号
from SELF import self
from LFDA import lfda_algorithm
from pca_ import pca_algorithm
from sklearn.preprocessing import StandardScaler
from FDA_ import fda_algorithm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def visualize_data(data, labels, title="Visualization"):
    """
    可视化数据
    :param data: 数据矩阵，形状为 (n_samples, n_features)
    :param labels: 标签数组，形状为 (n_samples,)
    :param title: 图形标题
    """
    plt.figure(figsize=(10, 8))

    # 定义颜色映射和标签字典
    color_map = {-1: 'blue', -2: 'red', 0: 'green', 1: 'orange'}
    label_dic = {-1: "戴眼镜标记", -2: "不戴眼镜标记", 0: "戴眼镜未标记", 1: "不戴眼镜未标记"}

    # 绘制散点图，并指定颜色和大小
    # plt.scatter(data[:, 0], data[:, 1], c=[color_map[label] for label in labels], edgecolor='k', s=50)
    sizes = [50 for i in range(100)]
    for i in range(10):
        sizes[i * 10] = 100

    plt.scatter(data[:, 0], data[:, 1], c=[color_map[label] for label in labels], edgecolor='k', s=sizes, alpha=0.8)

    # 添加图例
    for label, color in color_map.items():
        plt.scatter([], [], c=color, label=label_dic[label])  # 创建一个透明的散点图，只用于显示图例

    plt.title(title)
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")
    plt.legend(loc='upper left')
    plt.savefig(title + ".png")
    plt.show()


def display_image(matrix, title):
    # 重塑矩阵为 54*46
    reshaped_matrix = np.reshape(matrix, (56, 46)).real

    # 显示图像
    plt.imshow(reshaped_matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    path = "../ORL56_46/"
    # 获取数据集 共100张image 10个有标记 3个戴眼镜 7个不戴眼镜
    # 获取前100张图
    # NOTE 矩阵为n * d
    face_matrix = np.empty((100, 2576))  # 创建一个形状为 100x2576 的空矩阵
    for i in range(1, 11):
        for j in range(1, 11):
            test_img = Image.open(path + f"orl{i}_{j}.bmp")
            # 获取矩阵 56 * 46
            single_face_matrix = np.array(test_img)
            # 转换为一维矩阵 以行为单位展平 face_vector: 1 * 2576
            face_vector = single_face_matrix.reshape(-1)
            face_matrix[(i - 1) * 10 + (j - 1)] = face_vector

    mean_vector = np.mean(face_matrix)
    std_vector = np.std(face_matrix)
    # normalized_matrix = (face_matrix - mean_vector) / std_vector
    normalized_matrix = face_matrix

    # 初始化标签矩阵 戴眼镜为0 不戴眼镜为1
    real_label_matrix = np.empty((100, 1))
    test_label_matrix = np.empty((100, 1))
    visual_label_matrix = np.empty((100, 1))

    # 根据图片人工辨别的结果
    without_glasses = [0, 2, 4, 6, 7, 8, 9]
    with_glasses = [1, 3, 5]

    # 生成真实标签数组
    for i in range(100):
        test_label_matrix[i] = -1
        if (i // 10 in without_glasses):
            real_label_matrix[i] = 1
            visual_label_matrix[i] = 1
        else:
            real_label_matrix[i] = 0
            visual_label_matrix[i] = 0

    # 选取10个有标记的图片
    for i in with_glasses:
        # 戴眼镜的3个标记
        # index = random.random(0, 9)
        index = 0
        test_label_matrix[i * 10 + index][0] = 0
        visual_label_matrix[i * 10 + index][0] = -1

    for i in without_glasses:
        # 不戴眼镜的7个标记
        # index = random.random(0, 9)
        index = 0
        test_label_matrix[i * 10 + index][0] = 1
        visual_label_matrix[i * 10 + index][0] = -2

    unlabeled_matrix = normalized_matrix[test_label_matrix.flatten() == -1]

    labeled_matrix = normalized_matrix[test_label_matrix.flatten() != -1]

    # transform_matrix_SELF = self(labeled_matrix, unlabeled_matrix, test_label_matrix[test_label_matrix != -1].flatten(),2, beta=0.5)
    # result = np.dot(normalized_matrix, transform_matrix_SELF)
    # visualize_data(result, visual_label_matrix.flatten(), f"SELF beta=0.5")

    for beta in np.arange(0.1, 1, 0.1):
        transform_matrix_SELF = self(labeled_matrix, unlabeled_matrix, test_label_matrix[test_label_matrix != -1].flatten(), 2, beta=beta)
        result = np.dot(normalized_matrix, transform_matrix_SELF)
        visualize_data(result, visual_label_matrix.flatten(), f"SELF beta={beta}")

    # transform_matrix_LFDA = lfda_algorithm(normalized_matrix, real_label_matrix.flatten(), r=2)
    # result_LFDA = np.dot(normalized_matrix, transform_matrix_LFDA).real
    # visualize_data(result_LFDA, visual_label_matrix.flatten(), "全标签LFDA")
    #
    # transform_matrix_LFDA = lfda_algorithm(labeled_matrix, test_label_matrix[test_label_matrix != -1].flatten(), r=2)
    # result_LFDA_less = np.dot(face_matrix, transform_matrix_LFDA)
    # visualize_data(result_LFDA_less, visual_label_matrix.flatten(), "少量标签LFDA")
    #
    # transform_matrix_PCA = pca_algorithm(face_matrix, 2)
    # result_PCA = np.dot(face_matrix, transform_matrix_PCA)
    # visualize_data(result_PCA, visual_label_matrix.flatten(), "PCA")

    # transform_matrix_FDA = fda_algorithm(labeled_matrix, test_label_matrix[test_label_matrix != -1].flatten(), r=2)
    # result_FDA_less = np.dot(labeled_matrix, transform_matrix_FDA)
    # visualize_data(result_FDA_less,  test_label_matrix[test_label_matrix != -1].flatten(), "少量标签FDA")

    # result_array = np.concatenate((result, real_label_matrix), axis=1)
    # print(result_array)
    # recover_data_PCA = np.dot(result_PCA, transform_matrix_PCA.T)
    # recover_data_LFDA = np.dot(result_LFDA, transform_matrix_LFDA.T)

    # display_image(recover_data_PCA[0, :], "PCA还原")
    # display_image(recover_data_LFDA[0, :], "LFDA还原")















