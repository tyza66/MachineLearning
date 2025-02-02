import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error

def linear1():
    # 正规方程的优化方法对波士顿房价进行预测
    raw_df = pd.read_csv("./boston.csv", sep=r"\s+", skiprows=22, header=None)

    # 提取特征和目标
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # print(target)

    # 构建类似 load_boston 返回的字典
    boston = {
        'data': data,
        'target': target,
        'feature_names': np.array([
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT"
        ]),
        'DESCR': """Boston Housing Dataset
        =======================

        Notes
        -----
        Data Set Characteristics:
            :Number of Instances: 506
            :Number of Attributes: 13 continuous attributes (including "ZN" that can be treated as continuous)
            :Attribute Information (in order):
                - CRIM     per capita crime rate by town
                - ZN       proportion of residential land zoned for lots over 25,000 sq. ft.
                - INDUS    proportion of non-retail business acres per town
                - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                - NOX      nitric oxides concentration (parts per 10 million)
                - RM       average number of rooms per dwelling
                - AGE      proportion of owner-occupied units built prior to 1940
                - DIS      weighted distances to five Boston employment centers
                - RAD      index of accessibility to radial highways
                - TAX      full-value property-tax rate per $10,000
                - PTRATIO  pupil-teacher ratio by town
                - B        1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
                - LSTAT    % lower status of the population
            :Missing Attribute Values: None

        This dataset has been used for many machine learning experiments. See this [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for more details.
        """
    }

    x_train,x_test,y_train,y_test = train_test_split(boston['data'], boston['target'],random_state=22)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 正规方程求解
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)

    # 模型评估
    # 求均方误差
    y_predict = estimator.predict(x_test)
    print("正规方程-预测值为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)
    return None

def linear2():
    # 梯度下降的优化方法对波士顿房价进行预测
    raw_df = pd.read_csv("./boston.csv", sep=r"\s+", skiprows=22, header=None)

    # 提取特征和目标
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 构建类似 load_boston 返回的字典
    boston = {
        'data': data,
        'target': target,
        'feature_names': np.array([
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT"
        ]),
        'DESCR': """Boston Housing Dataset
        =======================

        Notes
        -----
        Data Set Characteristics:
            :Number of Instances: 506
            :Number of Attributes: 13 continuous attributes (including "ZN" that can be treated as continuous)
            :Attribute Information (in order):
                - CRIM     per capita crime rate by town
                - ZN       proportion of residential land zoned for lots over 25,000 sq. ft.
                - INDUS    proportion of non-retail business acres per town
                - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                - NOX      nitric oxides concentration (parts per 10 million)
                - RM       average number of rooms per dwelling
                - AGE      proportion of owner-occupied units built prior to 1940
                - DIS      weighted distances to five Boston employment centers
                - RAD      index of accessibility to radial highways
                - TAX      full-value property-tax rate per $10,000
                - PTRATIO  pupil-teacher ratio by town
                - B        1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
                - LSTAT    % lower status of the population
            :Missing Attribute Values: None

        This dataset has been used for many machine learning experiments. See this [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for more details.
        """
    }

    x_train,x_test,y_train,y_test = train_test_split(boston['data'], boston['target'],random_state=22)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 梯度下降求解 # 这里默认只用SGD
    estimator = SGDRegressor(eta0=0.01, max_iter=1000,learning_rate="constant",penalty="l1") # 学习率为0.01，最大迭代次数为1000
    estimator.fit(x_train, y_train)

    # 得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)

    # 模型评估
    # 求均方误差
    y_predict = estimator.predict(x_test)
    print("正规方程-预测值为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)
    return None


def linear3():
    # 岭回归对波士顿房价进行预测
    raw_df = pd.read_csv("./boston.csv", sep=r"\s+", skiprows=22, header=None)

    # 提取特征和目标
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 构建类似 load_boston 返回的字典
    boston = {
        'data': data,
        'target': target,
        'feature_names': np.array([
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT"
        ]),
        'DESCR': """Boston Housing Dataset
        =======================

        Notes
        -----
        Data Set Characteristics:
            :Number of Instances: 506
            :Number of Attributes: 13 continuous attributes (including "ZN" that can be treated as continuous)
            :Attribute Information (in order):
                - CRIM     per capita crime rate by town
                - ZN       proportion of residential land zoned for lots over 25,000 sq. ft.
                - INDUS    proportion of non-retail business acres per town
                - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                - NOX      nitric oxides concentration (parts per 10 million)
                - RM       average number of rooms per dwelling
                - AGE      proportion of owner-occupied units built prior to 1940
                - DIS      weighted distances to five Boston employment centers
                - RAD      index of accessibility to radial highways
                - TAX      full-value property-tax rate per $10,000
                - PTRATIO  pupil-teacher ratio by town
                - B        1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
                - LSTAT    % lower status of the population
            :Missing Attribute Values: None

        This dataset has been used for many machine learning experiments. See this [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for more details.
        """
    }

    x_train,x_test,y_train,y_test = train_test_split(boston['data'], boston['target'],random_state=22)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 岭回归求解 # 这里实现了SAG
    estimator = Ridge(alpha=0.5,max_iter=10000) # alpha为正则化力度(惩罚力度)
    estimator.fit(x_train, y_train)

    # 得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 模型评估
    # 求均方误差
    y_predict = estimator.predict(x_test)
    print("岭回归-预测值为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)
    return None

if __name__ == "__main__":
    linear3()