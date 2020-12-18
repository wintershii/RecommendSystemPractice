import numpy as np
from sklearn import datasets


class PCATest:
    def __init__(self):
        pass

    def loadIris(self):
        data = datasets.load_iris()["data"]
        return data

    # 标准化数据
    def Standard(self, data):
        # axis 按列取均值
        mean_vector = np.mean(data, axis=0)
        return mean_vector, data - mean_vector

    # 计算协方差矩阵
    def getCovMatrix(self, newData):
        # rowvar = 0
        return np.cov(newData, rowvar=0)

    # 计算协方差矩阵的特征值和特征矩阵
    def getFValueAndFVector(self, covMatrix):
        fValue, fVector = np.linalg.eig(covMatrix)
        return fValue, fVector

    # 得到特征向量矩阵
    def getVectorMatrix(self, fValue, fVector, k):
        fValueSort = np.argsort(fValue)
        fValueTopN = fValueSort[:-(k + 1):-1]
        return fVector[:, fValueTopN]

    # 得到降维后的数据
    def getResult(self, data, vectorMatrix):
        return np.dot(data, vectorMatrix)


if __name__ == "__main__":
    pcatest = PCATest()
    data = pcatest.loadIris()
    print("原数据:\n{}".format(data))
    mean_vector, newData = pcatest.Standard(data)
    # print("标准化后数据:\n{}".format(newData))
    covMatrix = pcatest.getCovMatrix(newData)
    print("协方差矩阵为:\n{}".format(covMatrix))

    fValue, fVector = pcatest.getFValueAndFVector(covMatrix)
    print("特征值为:{}".format(fValue))
    print("特征矩阵为:\n{}".format(fVector))

    vectorMatrix = pcatest.getVectorMatrix(fValue, fVector, k=2)
    print("k维特征向量矩阵为:\n{}".format(vectorMatrix))

    result = pcatest.getResult(newData, vectorMatrix)
    print("最终降维结果为:\n{}".format(result))
    print("最终重构结果为:\n{}".format(np.mat(result) * vectorMatrix.T + mean_vector))
