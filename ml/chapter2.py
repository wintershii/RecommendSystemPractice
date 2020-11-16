import pandas as pd
from sklearn.model_selection import train_test_split

# 将csv文件中的数据分为训练集和测试集
csv_data = pd.read_csv('C://Users//mirac//Desktop//temperature_icecream.csv')
df = pd.DataFrame(csv_data)
print(df)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=40)

print(X_train)
print(X_test)
