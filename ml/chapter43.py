import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# 最小二乘法拟合一元一次线性方程
data = np.loadtxt('C:\\Users/mirac/Desktop/temperature_icecream.csv',
                  dtype=float,
                  usecols=(0, 1),
                  skiprows=1,
                  encoding="utf8",
                  delimiter=",")

x = data[:, 0]
y = data[:, 1]


# 确定函数表达式一元一次线性方程是一条直线a * x + b
def func(p, x):
    a, b = p
    y = a * x + b
    return y


# 求出损失函数, 即预测值_y减去观察值y
def errors(p, x, y):
    _y = func(p, x)
    return y - _y


res = leastsq(errors, [1, 0], args=(x, y))
_a, _b = res[0]
print("a=", _a)
print("b=", _b)

plt.scatter(x, y, color="red")
plt.plot(x, func([_a, _b], x), color="blue")
plt.show()