"""
Author:  Blank
Date: 2024/5/19
Description: 
"""



import numpy as np



def fda_algorithm(data_matrix: np.ndarray, label_matrix: np.ndarray, r: int):
    n = data_matrix.shape[0]
    d = data_matrix.shape[1]


    num_class_dic = {}
    for i in range(n):
        label = label_matrix[i]
        if label not in num_class_dic:
            num_class_dic[label] = 0
        num_class_dic[label] += 1

    W_lb = np.zeros((n, n))
    W_lw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if label_matrix[i] == label_matrix[j]:
                W_lb[i][j] = (1 / n - 1 / num_class_dic[label_matrix[i]])
                W_lw[i][j] = 1 / num_class_dic[label_matrix[i]]
            else:
                W_lb[i][j] = 1 / num_class_dic[label_matrix[i]]
                W_lw[i][j] = 0

    one_n = np.ones((n, 1))
    diag_W_lb = np.diag(np.dot(W_lb, one_n).flatten())
    diag_W_lw = np.diag(np.dot(W_lw, one_n).flatten())

    S_lb = np.dot(data_matrix.T, np.dot(diag_W_lb - W_lb, data_matrix))
    S_lw = np.dot(data_matrix.T, np.dot(diag_W_lw - W_lw, data_matrix))
    aim_matrix = np.dot(np.linalg.inv(S_lw), S_lb)
    eigenvalues, eigenvectors = np.linalg.eig(aim_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvectors = sorted_eigenvectors[:, :r]

    # for i in range(r):
    #     sorted_eigenvectors[:, i] = sorted_eigenvectors[:, i] * np.sqrt(sorted_eigenvalues[i])
    #
    # print("特征向量乘以对应的根号特征值后：", sorted_eigenvectors)
    return sorted_eigenvectors

