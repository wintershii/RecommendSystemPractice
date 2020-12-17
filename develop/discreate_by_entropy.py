import numpy as np
import math


# 基于信息熵进行数据离散化
class DiscreateByEntropy:
    def __init__(self, group, threshold):
        self.maxGroup = group  # 最大分组数
        self.minInfoThreshold = threshold  # 停止划分的最小熵
        self.result = dict()  # 保存划分结果

    def loadData(self):
        data = np.array([
            [56, 1],
            [87, 1],
            [129, 0],
            [23, 0],
            [342, 1],
            [641, 1],
            [63, 0],
            [2764, 1],
            [2323, 0],
            [453, 1],
            [10, 1],
            [9, 0],
            [88, 1],
            [222, 0],
            [97, 0],
            [2398, 1],
            [592, 1],
            [561, 1],
            [764, 0],
            [121, 1],
        ])
        return data

    # 计算数据的信息熵
    def calEntropy(self, data):
        numData = len(data)
        labelCounts = {}
        for feature in data:
            oneLabel = feature[-1]
            labelCounts.setdefault(oneLabel, 0)
            labelCounts[oneLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numData
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    # 分割数据集
    def split(self, data):
        # inf为正无穷
        minEntropy = np.inf
        # 记录分割索引
        index = -1
        # 按照第一列对数据进行排序
        sortData = data[np.argsort(data[:, 0])]

        lastE1, lastE2 = -1, -1

        S1 = dict()
        S2 = dict()
        for i in range(len(sortData)):
            # 分割数据集
            splitData1, splitData2 = sortData[:i + 1], sortData[i + 1:]
            entropy1, entropy2 = (
                self.calEntropy(splitData1),
                self.calEntropy(splitData2),
            )
            entropy = entropy1 * len(splitData1) / len(
                sortData) + entropy2 * len(splitData2) / len(sortData)
            # 如果调和平均熵小于最小值
            if entropy < minEntropy:
                minEntropy = entropy
                index = i
                lastE1 = entropy1
                lastE2 = entropy2
        S1["entropy"] = lastE1
        S1["data"] = sortData[:index + 1]
        S2["entropy"] = lastE2
        S2["data"] = sortData[index + 1:]
        print("s1:{}\n".format(S1["data"]))
        print("s2:{}\n".format(S2["data"]))
        return S1, S2, minEntropy

    # 数据离散化
    def train(self, data):
        needSplitKey = [0]
        self.result.setdefault(0, {})
        self.result[0]["entropy"] = np.inf
        self.result[0]["data"] = data
        group = 1
        for key in needSplitKey:
            S1, S2, entropy = self.split(self.result[key]["data"])
            if entropy > self.minInfoThreshold and group < self.maxGroup:
                self.result[key] = S1
                newKey = max(self.result.keys()) + 1
                self.result[newKey] = S2
                needSplitKey.extend([key])
                needSplitKey.extend([newKey])
                group += 1
            else:
                break


if __name__ == "__main__":
    dbe = DiscreateByEntropy(group=6, threshold=0.5)
    data = dbe.loadData()
    dbe.train(data)
print("result is {}".format(dbe.result))