"""
在推荐系统中，通常得到的是矩阵A(大矩阵)，会出现数据稀疏的问题
矩阵分解就是将大矩阵近似的分解成两个小矩阵的乘积，便于分类和预测
常见的矩阵分解有：SVD奇异值分解，PMF基于概率的矩阵分解

SVD：适用于任意矩阵的一种分解方法，在协同过滤推荐中的作用是数据降维，解决数据稀疏
    - 减少特征空间，去除数据噪声，提高推荐效果
    前提：矩阵是稠密的。方案：填充矩阵，再分解降维，计算复杂度高

PMF：是在正则化分解的基础上，引入了概率模型
用特征矩阵去预测评分矩阵中的未知值，前提：特征矩阵要符合高斯分布
场景：评分矩阵-》用户特征矩阵+物品特征矩阵-》补全矩阵
"""

import numpy as np

A = [
    [1, 2, 0],
    [3, 4, 0],
    [5, 6, 0]
]

data = np.mat(A)

# 特征值分解：Q * S * Q的逆矩阵

Q, S = np.linalg.eig(data)
_A = np.dot(S, np.diag(Q)).dot(S.I)
print(_A)
print(np.allclose(A, _A))


# 奇异值分解 U(m*m) * S(m*n) * V(n*n)
B = [
    [1, 2, 0, 0, 1],
    [3, 4, 0, 0, 0],
    [5, 6, 0, 0, 2],
    [0, 0, 0, 0, 0]
]

U, S, V = np.linalg.svd(B)
# print(U)
# 向量转矩阵
k = 3
_S = np.diag(S[:k]) 
# print(_S)
# print(V)
_B = np.dot(U[:, :k], _S).dot(V[:k, :])
print(_B)
print(np.allclose(B, _B))
