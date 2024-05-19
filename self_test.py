import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 生成示例数据
np.random.seed(0)
data = np.random.randn(200, 10)  # 200个样本，每个样本有10个特征
labels = np.random.randint(2, size=200)  # 随机生成0和1的标签

# 使用LDA进行降维
lda = LinearDiscriminantAnalysis(n_components=2)
data_lda = lda.fit_transform(data, labels)

# 可视化降维结果
plt.scatter(data_lda[labels == 0, 0], data_lda[labels == 0, 1], label='Class 0', c='r')
plt.scatter(data_lda[labels == 1, 0], data_lda[labels == 1, 1], label='Class 1', c='b')
plt.legend()
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.show()