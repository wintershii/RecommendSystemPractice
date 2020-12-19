import numpy as np


class KNN:
    def __init__(self, k):
        self.K = k

    def createData(self):
        features = np.array([[180, 76], [158, 43], [176, 78], [161, 49]])
        labels = ["男", "女", "男", "女"]
        return features, labels
    
    # 数据进行Min-Max标准化
    def Normalization(self, data):
        maxs = np.max(data, axis=0)
        print(maxs)
        mins = np.min(data, axis=0)
        print(mins)
        new_data = (data - mins) / (maxs - mins)
        # print(new_data)
        return new_data, maxs, mins
    
    def classify(self, one, data, labels):
        differenceData = data - one
        print(differenceData)
        squareData = (differenceData ** 2).sum(axis=1)
        print(squareData)
        distance = squareData ** 0.5
        sortDistanceIndex = distance.argsort()
        print(sortDistanceIndex)
        # 统计K最邻近的label
        labelCount = dict()
        for i in range(self.K):
            label = labels[sortDistanceIndex[i]]
            labelCount.setdefault(label, 0)
            labelCount[label] += 1
        # 计算结果
        sortLabelCount = sorted(labelCount.items(), key=lambda x: x[1], reverse=True)
        print(sortLabelCount)
        return sortLabelCount[0][0]

if __name__ == "__main__":
    knn = KNN(3)
    features, labels = knn.createData()
    new_data, maxs, mins = knn.Normalization(features)

    one = np.array([176, 76])
    new_one = (one - mins) / (maxs - mins)
    result = knn.classify(new_one, new_data, labels)
    print("数据{}的预测性别为:{}".format(one, result))
