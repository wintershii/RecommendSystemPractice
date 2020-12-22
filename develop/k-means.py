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

    # 计算对应的SSE值
    def SSE(self, data, mean):
        newData = np.mat(data) - mean
        return (newData * newData.T).tolist()[0][0]

    # 二分k-Means
    def diKMeans(self, data, K):
        clusterSSEResult = dict()
        clusterSSEResult.setdefault(0, {})
        clusterSSEResult[0]["values"] = data
        clusterSSEResult[0]["sse"] = np.inf
        clusterSSEResult[0]["center"] = np.mean(data)
        # print(clusterSSEResult[0])
        while len(clusterSSEResult) < K:
            maxSSE = -np.inf
            maxSSEKey = 0
            # 找到最大SSE值对应数据, 进行kmeans聚类
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]["sse"] > maxSSE:
                    maxSSE = clusterSSEResult[key]["sse"]
                    maxSSEKey = key
            clusterResult = self.kMeans(clusterSSEResult[maxSSEKey]["values"],
                                        K=2,
                                        maxIters=200)
            # 删除clusterSSE中的minKey对应的值
            del clusterSSEResult[maxSSEKey]
            # 将经过kMeans聚类后的结果赋值给clusterSSEResult
            clusterSSEResult.setdefault(maxSSEKey, {})
            # print(clusterSSEResult[0])
            clusterSSEResult[maxSSEKey]["center"] = clusterResult[0]["center"]
            clusterSSEResult[maxSSEKey]["values"] = clusterResult[0]["values"]
            clusterSSEResult[maxSSEKey]["sse"] = self.SSE(
                clusterResult[0]["values"], clusterResult[0]["center"])

            maxKey = max(clusterSSEResult.keys()) + 1
            clusterSSEResult.setdefault(maxKey, {})
            clusterSSEResult[maxKey]["center"] = clusterResult[1]["center"]
            clusterSSEResult[maxKey]["values"] = clusterResult[1]["values"]
            clusterSSEResult[maxKey]["sse"] = self.SSE(
                clusterSSEResult[1]["values"], clusterResult[1]["center"])

        return clusterSSEResult


if __name__ == "__main__":
    km = KMeans()
    file = 'C:\\Users\\mirac\\Desktop\\price.csv'
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    Cluster = km.diKMeans(newData["price"].values, K=7)
    print(Cluster)

# def createData():
#     content = {}
#     content['skuid'] = []
#     content['price'] = []

#     for i in range(100):
#         content['skuid'].append(1000 + i)
#         content['price'].append(random.randint(10, 10000))

#     df = pd.DataFrame(content, columns=['skuid', 'price'])
#     df.to_csv('C:\\Users\\mirac\\Desktop\\price.csv', index=False, sep=",")

# if __name__ == "__main__":
#     # createData()
#     km = KMeans()
#     file = 'C:\\Users\\mirac\\Desktop\\price.csv'
#     data = km.loadData(file)
#     newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
#     Cluster = km.kMeans(newData["price"].values, K=7, maxIters=200)
#     print(Cluster)