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

def lfda_algorithm(data_matrix: np.ndarray, label_matrix: np.ndarray, r: int):
    n = data_matrix.shape[0]
    d = data_matrix.shape[1]

    A_matrix = affinity_matrix(data_matrix, label_matrix.flatten())

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
                W_lb[i][j] = A_matrix[i][j] * (1 / n - 1 / num_class_dic[label_matrix[i]])
                W_lw[i][j] = A_matrix[i][j] / num_class_dic[label_matrix[i]]
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

    for i in range(r):
        sorted_eigenvectors[:, i] = sorted_eigenvectors[:, i] * np.sqrt(sorted_eigenvalues[i])

    print("特征向量乘以对应的根号特征值后：", sorted_eigenvectors)
    return sorted_eigenvectors

def create_toy_data():
    np.random.seed(42)
    labeled_data_matrix = np.random.rand(30, 5)
    label_matrix = np.array([0] * 10 + [1] * 10 + [2] * 10)

    return labeled_data_matrix, label_matrix

if __name__ == '__main__':
    labeled_data_matrix, label_matrix = create_toy_data()
    T = lfda_algorithm(labeled_data_matrix, label_matrix, r=2)
    print("映射矩阵T：")
    print(T)
