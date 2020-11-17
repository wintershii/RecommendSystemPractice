import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
"""
欧式距离
A(13,26) B(56,89)
"""

# numpy实现
x = np.array([13, 26])
y = np.array([56, 89])

_x = np.mat(x)
_y = np.mat(y)

dist_np = np.sqrt(np.sum(np.square(_x - _y)))

print("distance for numpy=", dist_np)

# scipy实现
z = np.vstack([x, y])
dist_scipy = pdist(z)

print(dist_scipy)

# 余弦相似度

A = np.mat(np.array([13, 26]))
B = np.mat(np.array([56, 89]))

# 分子:x1*x2 + y1*y2
i = float(A * B.T)
# 分母:a模*b模
n = np.linalg.norm(A) * np.linalg.norm(B)
sim = i / n

print("sim for numpy = ", sim)

# sklearn
z = np.array([[13, 26], [56, 89]])
# 返回的二维矩阵每个位置表示z[i],z[j]向量的余弦相似度

sim_sklearn = cosine_similarity(z)
print("sim for sklearn = ", sim_sklearn[0][1])
