import numpy as np
import pandas as pd
import random
import operator


class KMeans:
    def __init__(self):
        pass

    def loadData(self, file):
        return pd.read_csv(file, header=0, sep=",")

    def filterAnomalyValue(self, data):
        upper = np.mean(data["price"]) + 3 * np.std(data["price"])
        lower = np.mean(data["price"]) - 3 * np.std(data["price"])
        upper_limit = upper if upper > 5000 else 5000
        lower_limit = lower if lower > 1 else 1
        print("最大异常值为{},最小异常值为{}".format(upper_limit, lower_limit))
        # 过滤低调大于最大异常值和小于最小异常值的
        newData = data[(data["price"] < upper_limit)
                       & (data["price"] > lower_limit)]
        return newData, upper_limit, lower_limit

    # 初始化簇类中心
    def initCenters(self, values, K, Cluster):
        random.seed(8848)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0, len(values))
            Cluster.setdefault(i, {})
            Cluster[i]["center"] = values[index]
            Cluster[i]["values"] = []

            oldCenters.append(values[index])
        return oldCenters, Cluster

    # 计算任意两条数据之间的欧式距离
    def distance(self, price1, price2):
        return np.emath.sqrt(pow(price1 - price2, 2))

    # 聚类
    def kMeans(self, data, K, maxIters):
        Cluster = dict()
        oldCenters, Cluster = self.initCenters(data, K, Cluster)
        print("初始簇类中心为:{}".format(oldCenters))
        # 标志变量，若为True则继续迭代
        clusterChanged = True
        i = 0
        while clusterChanged:
            for price in data:
                minDistance = np.inf
                minIndex = -1
                for key in Cluster.keys():
                    # 计算每条数据到簇类中心的距离
                    dis = self.distance(price, Cluster[key]["center"])
                    if dis < minDistance:
                        minDistance = dis
                        minIndex = key
                Cluster[minIndex]["values"].append(price)
            # for key in Cluster.keys():
            #     print("key:{}, values:{}".format(key, Cluster[key]["values"]))
            newCenters = list()

            for key in Cluster.keys():
                newCenter = np.round(np.mean(Cluster[key]["values"]), 3) 
                Cluster[key]["center"] = newCenter
                newCenters.append(newCenter)
            print("第{}次迭代后的簇类中心为:{}, old:{}".format(i, newCenters, oldCenters))
            if operator.eq(oldCenters, newCenters) or i > maxIters:
                clusterChanged = False
                
            else:
                oldCenters = newCenters
                i += 1
                # 删除Cluster中记录的簇类值
                for key in Cluster.keys():
                    Cluster[key]["values"] = []
        return Cluster


def createData():
    content = {}
    content['skuid'] = []
    content['price'] = []

    for i in range(100):
        content['skuid'].append(1000 + i)
        content['price'].append(random.randint(10, 10000))

    df = pd.DataFrame(content, columns=['skuid', 'price'])
    df.to_csv('C:\\Users\\mirac\\Desktop\\price.csv', index=False, sep=",")


if __name__ == "__main__":
    # createData()
    km = KMeans()
    file = 'C:\\Users\\mirac\\Desktop\\price.csv'
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    Cluster = km.kMeans(newData["price"].values, K=7, maxIters=200)
    print(Cluster)