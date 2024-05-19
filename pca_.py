"""
Author:  Blank
Date: 2024/5/18
Description: 
"""
import numpy as np
import matplotlib.pyplot as plt


def pca_algorithm(data_matrix: np.ndarray, r: int):
    """
    PCA算法
    :param data_matrix: 数据矩阵 n * d n为数据数量 d为维数
    :param r: 目标维数
    :return: 映射矩阵T和降维后的数据
    """
    # 数据矩阵中心化
    mean_vector = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_vector

    # 计算协方差矩阵
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 按特征值从大到小排序特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 返回d*r转换矩阵
    return sorted_eigenvectors[:, :r]


# Toy data example
# def create_toy_data():
#     np.random.seed(42)
#     # Create some sample data
#     data_matrix = np.random.rand(100, 5)
#     return data_matrix
#
#
# if __name__ == '__main__':
#     data_matrix = create_toy_data()
#     T, projected_data = pca(data_matrix, r=2)
#     print("映射矩阵T：")
#     print(T)
#     print("降维后的数据：")
#     print(projected_data)
#
#     # 可视化降维后的数据
#     plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
#     plt.title("PCA Result")
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.show()
