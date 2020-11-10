import random
import math
import operator


# 读取数据
def ReadData():
    data = dict()
    with open('C://Users/mirac/Desktop/ml-1m/ratings.dat') as file_object:
        for line in file_object:
            line = line.strip('\n')
            user_data = line.split('::')
            user = user_data.pop(0)
            item = user_data.pop(0)
            if user not in data:
                data[user] = []
            data[user].append(item)
    return data


# 将数据分为训练集和测试集
def SplitData(data, M, k, seed):
    test = dict()
    train = dict()
    random.seed(seed)
    for user, items in data.items():
        for item in items:
            if random.randint(0, M) == k:
                if user not in test:
                    test[user] = []
                test[user].append(item)
            else:
                if user not in train:
                    train[user] = []
                train[user].append(item)
    return train, test


# 余弦相似度计算
def UserSimilarity(train):
    # 1.建立 物品-用户的倒排表
    item_users = dict()
    for u, items in train.items():
        for i in items:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    # 2.计算用户间的相同物品
    C = dict()
    N = dict()
    for item, users in item_users.items():
        for u in users:
            if u not in N:
                N[u] = 0
            N[u] += 1
            if u not in C:
                C[u] = dict()
            for v in users:
                if u == v:
                    continue
                else:
                    if v not in C[u]:
                        C[u][v] = 0
                    C[u][v] += 1
                    # 惩罚用户u和用户v共同兴趣列表中热门物品对他们相似度的影响
                    # C[u][v] += 1 / math.log(1+len(users))
    # 3.计算余弦相似度矩阵W
    W = dict()
    for u, related_users in C.items():
        W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W


# 对于一个用户的推荐算法
def Recommond(user, train, W):
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items(),
                         key=operator.itemgetter(1),
                         reverse=True)[0:K]:
        for v_item in train[v]:
            if v_item in interacted_items:
                continue
            if v_item not in rank:
                rank[v_item] = 0
            rank[v_item] += wuv * 1.0
    return rank


# 用于测试集的用户推荐算法
def GetRecommendation(user, N):
    rank = Recommond(user, train, W)
    rank_N = dict(
        sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N])
    return rank_N


# 召回率
def Recall(train, test, N):
    hit = 0
    all = 0
    rank = dict()
    for user in train.keys():
        if user in list(test.keys()):
            tu = test[user]
            rank = GetRecommendation(user, N)
            for item, pui in rank.items():
                if item in tu:
                    hit += 1
            all += len(tu)
    return hit / (all * 1.0)


# 准确率
def Precision(train, test, N):
    hit = 0
    all = 0
    tu = []
    rank = dict()
    for user in train.keys():
        if user in list(test.keys()):
            tu = test[user]
            rank = GetRecommendation(user, N)
            print(rank)
            for item, pui in rank.items():
                if item in tu:
                    hit += 1
            all += N
    print('hit ', hit)
    print('all ', all)
    return hit / (all * 1.0)


# 覆盖率
def Coverage(train, test, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank.items():
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)


# 流行度
def Popularity(train, test, N):
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for item, pui in rank.items():
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret


M = 8
k = 0
seed = 2000
N = 10
data = ReadData()
# print(data)
train = dict()
test = dict()
K = 5
train, test = SplitData(data, M, k, seed)
# print(test)
W = UserSimilarity(train)
# print(W)
p = Precision(train, test, N)
print('准确率为：', p)
r = Recall(train, test, N)
print('召回率为：', r)
print('覆盖率为：', Coverage(train, test, N))
print('新颖度为:', Popularity(train, test, N))
