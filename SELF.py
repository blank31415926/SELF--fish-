import numpy as np


def affinity_matrix(data_matrix: np.ndarray, label_list, k: int = 7):
    """
    亲和力矩阵
    :param data_matrix: n*d n为数据数量 d为维数
    默认亲和力计算中sigma的系数k=7
    :return: 亲和力矩阵A n * n
    """
    n = data_matrix.shape[0]
    if n < k:
        assert False, "data_matrix的列数小于k，无法继续执行"
    # 计算距离矩阵
    distance_matrix = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            distance = np.linalg.norm(data_matrix[i] - data_matrix[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    # 计算sigma的矩阵 x_i与x_i最近的第k个的欧式距离 矩阵大小为n * 1
    sigma_matrix = np.empty((n, 1))
    for i in range(n):
        k_th_distance = np.sort(distance_matrix[i])[k - 1]
        sigma_matrix[i] = k_th_distance
    # n * n
    A_matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if label_list[i] == label_list[j]:
                A_matrix[i][j] = np.exp(-distance_matrix[i][j] ** 2 / (sigma_matrix[i][0] * sigma_matrix[j][0]))
            else:
                A_matrix[i][j] = 0
    return A_matrix


def self(labeled_data_matrix: np.ndarray, unlabeled_data_matrix: np.ndarray, label_matrix: np.array, r: int, beta=0.5):
    """
    SELF算法
    :param labeled_data_matrix: 标签数据矩阵 n_prime * d n为数据数量 d为维数
    :param unlabeled_data_matrix: n*d n为标签数据数量 d为维数
    :param label_matrix: 标签矩阵 例如 [0, 0, ... , 1, 1] 对应每个列向量的标签
    :param beta: 权衡系数
    :param r: 目标维数
    :return: 映射矩阵T
    """
    # 计算亲和力矩阵
    A_matrix = affinity_matrix(labeled_data_matrix, label_matrix)
    n_prime = labeled_data_matrix.shape[0]
    d = unlabeled_data_matrix.shape[1]
    n = unlabeled_data_matrix.shape[0]
    # 获取各个标签的数量
    num_class_dic = {}
    for i in range(n_prime):
        if label_matrix[i] not in num_class_dic:
            num_class_dic[label_matrix[i]] = 0
        num_class_dic[label_matrix[i]] += 1

    # 计算LFDA的矩阵
    W_lb = np.empty((n_prime, n_prime))
    W_lw = np.empty((n_prime, n_prime))
    for i in range(n_prime):
        for j in range(n_prime):
            if label_matrix[i] == label_matrix[j]:
                W_lb[i][j] = A_matrix[i][j] * (1 / n_prime - 1 / num_class_dic[label_matrix[i]])
                W_lw[i][j] = A_matrix[i][j] / num_class_dic[label_matrix[i]]
            else:
                W_lb[i][j] = 1 / num_class_dic[label_matrix[i]]
                W_lw[i][j] = 0
    one_n_prime = np.ones((n_prime, 1))
    diag_W_lb = np.diag(np.dot(W_lb, one_n_prime).flatten())
    diag_W_lw = np.diag(np.dot(W_lw, one_n_prime).flatten())
    # d * d
    # 这里labeled_data_matrix是n_prime * d
    S_lb = np.dot(labeled_data_matrix.T, np.dot(diag_W_lb - W_lb, labeled_data_matrix))
    S_lw = np.dot(labeled_data_matrix.T, np.dot(diag_W_lw - W_lb, labeled_data_matrix))
    # # d * 1
    # unlabeled_mu = np.dot(unlabeled_data_matrix.T, np.ones((n, 1))) * (1 / n)
    # # 协方差矩阵 d * d
    # S_t = np.dot(unlabeled_data_matrix.T, unlabeled_data_matrix) - n * np.dot(unlabeled_mu.T, unlabeled_mu)

    # 数据矩阵中心化 计算协方差矩阵
    mean_vector = np.mean(unlabeled_data_matrix, axis=0)
    centered_data = unlabeled_data_matrix - mean_vector
    S_t = np.cov(centered_data, rowvar=False)

    S_rlb = (1 - beta) * S_lb + beta * S_t
    S_rlw = (1 - beta) * S_lw + beta * np.eye(d)

    aim_matrix = np.dot(np.linalg.inv(S_rlw), S_rlb)
    eigenvalues, eigenvectors = np.linalg.eig(aim_matrix)
    # 打印特征值和特征向量
    print("特征值：", eigenvalues)
    print("特征向量：", eigenvectors)
    # 降序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    print("排序后的特征值：", sorted_eigenvalues)
    print("排序后的特征向量：", sorted_eigenvectors)
    print(sorted_eigenvectors[:, :r])

    # for i in range(r):
    #     sorted_eigenvectors[:, i] = sorted_eigenvectors[:, i] * np.sqrt(sorted_eigenvalues[i])
    #
    # print("特征向量乘以对应的根号特征值后：", sorted_eigenvectors)

    return sorted_eigenvectors


# def generate_test_data():
#     n_prime = 10  # 标签数据数量
#     n = 20  # 未标签数据数量
#     d = 5  # 维数
#
#     labeled_data_matrix = np.random.rand(n_prime, d)  # 生成随机的标签数据矩阵
#     unlabel_data_matrix = np.random.rand(n, d)  # 生成随机的未标签数据矩阵
#     label_matrix = np.random.randint(2, size=n_prime)  # 随机生成标签矩阵
#
#     beta = 0.5  # 权衡系数
#     r = 3  # 目标维数
#
#     return labeled_data_matrix, unlabel_data_matrix, label_matrix, beta, r
#
# # 生成测试数据
# labeled_data_matrix, unlabel_data_matrix, label_matrix, beta, r = generate_test_data()
#
# # 调用 self 算法
# self(labeled_data_matrix, unlabel_data_matrix, label_matrix, r, beta)



