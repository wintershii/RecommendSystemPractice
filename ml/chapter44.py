import numpy as np
import matplotlib.pyplot as plt

# 梯度下降法拟合一元一次线性方程
data = np.loadtxt('C:\\Users/mirac/Desktop/temperature_icecream.csv',
                  dtype=float,
                  usecols=(0, 1),
                  skiprows=1,
                  encoding="utf8",
                  delimiter=",")

x = data[:, 0]
y = data[:, 1]


def func(p, x):
    k, b = p
    y = k * x + b
    return y


def cost(p, x, y):
    tatol_cost = 0
    data_num = len(data)
    for i in range(data_num):
        y_real = y[i]
        _x = x[i]
        _y = func(p, _x)
        tatol_cost += (_y - y_real)**2
    return tatol_cost / data_num


initial_p = [0, 0]
learning_rate = 0.0001
max_iter = 30


def grad(initial_p, learning_rate, max_iter):
    cost_list = []
    k, b = initial_p
    for i in range(max_iter):
        k, b = step_grad(k, b, learning_rate, x, y)
        _cost = cost([k, b], x, y)
        cost_list.append(_cost)
    return [k, b, cost_list]


def step_grad(current_k, current_b, learning_rate, x, y):
    sum_grad_k = 0
    sum_grad_b = 0
    data_num = len(data)
    for i in range(data_num):
        y_real = y[i]
        _x = x[i]
        _y = func([current_k, current_b], _x)
        # 参数k,b的偏导
        sum_grad_k += (_y - y_real) * _x
        sum_grad_b += _y - y_real
    # 参数k,b的梯度
    grad_k = 2 / data_num * sum_grad_k
    grad_b = 2 / data_num * sum_grad_b

    update_k = current_k - learning_rate * grad_k
    update_b = current_b - learning_rate * grad_b

    return update_k, update_b


plt.scatter(x, y, color="red")
# plt.show()

k, b, cost_list = grad(initial_p, learning_rate, max_iter)
plt.plot(x, func([k, b], x), color="blue")
# plt.plot(cost_list)
plt.show()