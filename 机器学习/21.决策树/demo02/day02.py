from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz
def knn_iris():
    # 用knn算法对鸢尾花数据集进行分类
    # 1 获取数据 2 划分数据集 3 特征工程标准化 4 knn算法预估器 5 评估模型

    # 1 获取
    iris = load_iris()

    # 2 划分
    # 特征值 目标值 随机数种子
    # x_train：训练集的特征数据。
    # x_test：测试集的特征数据。
    # y_train：训练集的目标标签。
    # y_test：测试集的目标标签。
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train) # 训练并转换
    # 如果我再用fit那两次的平均值和标准差会混在一起 因为使用fit方法时，平均值和标准差会存储在transfer里面
    x_test = transfer.transform(x_test) # 测试集直接转换 因为已经训练过了平均值和标准差 无需fit 直接用transform

    # 4 knn算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 将训练集的特征值和目标值传入
    estimator.fit(x_train, y_train)

    # 5 评估模型
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test) # 获得预测值
    print("预测值为：", y_predict)
    print("比对真实值和预测值：", y_test == y_predict) # 比对真实值和预测值

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：", score)

    return None

def knn_iris_gscv():
    # 用knn算法对鸢尾花数据集进行分类,添加网格搜索和交叉验证
    # 1 获取数据 2 划分数据集 3 特征工程标准化 4 knn算法预估器 5 评估模型

    # 1 获取
    iris = load_iris()

    # 2 划分
    # 特征值 目标值 随机数种子
    # x_train：训练集的特征数据。
    # x_test：测试集的特征数据。
    # y_train：训练集的目标标签。
    # y_test：测试集的目标标签。
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train) # 训练并转换
    # 如果我再用fit那两次的平均值和标准差会混在一起 因为使用fit方法时，平均值和标准差会存储在transfer里面
    x_test = transfer.transform(x_test) # 测试集直接转换 因为已经训练过了平均值和标准差 无需fit 直接用transform

    # 4 knn算法预估器
    estimator = KNeighborsClassifier() # 因为我们要试所以不传入参数

    # 加入网格搜索和交叉验证
    # 1）参数准备
    param_dict = {"n_neighbors":[1,3,5,7,9,11,20]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10) # 10折交叉验证(最常用)

    # 将训练集的特征值和目标值传入
    estimator.fit(x_train, y_train)

    # 5 评估模型
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test) # 获得预测值
    print("预测值为：", y_predict)
    print("比对真实值和预测值：", y_test == y_predict) # 比对真实值和预测值

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：", score)

    # 最佳参数
    print("最佳参数：", estimator.best_params_)
    # 最佳结果
    print("最佳结果：", estimator.best_score_)
    # 最佳估计器
    print("最佳估计器：", estimator.best_estimator_)
    # 交叉验证结果
    print("交叉验证结果：", estimator.cv_results_)

    return None

def nb_news():
    # 朴素贝叶斯算法对新闻分类

    # 获取数据集
    news = fetch_20newsgroups(subset="all")

    # 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(news.data, news.target)
    # 文本特征抽取
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 朴素贝叶斯算法预估器
    estimator = MultinomialNB() # 默认alpha=1.0
    estimator.fit(x_train, y_train) # 训练

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测值为：", y_predict)
    print("比对真实值和预测值：", y_test == y_predict) # 比对真实值和预测值
    score = estimator.score(x_test, y_test) # 计算准确率 使用测试集的特征值和目标值
    print("准确率为：", score)


    return None

def decison_iris():
    # 决策树对鸢尾花数据集进行分类

    # 1 获取数据
    iris = load_iris()

    # 2 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 决策树不用弄特征工程 标准化
    # 3 决策树预估器 estimator就是预估器
    estimator = DecisionTreeClassifier(criterion="entropy") # 默认是gini基尼系数 我们用信息熵 信息增益
    estimator.fit(x_train, y_train) # 训练

    # 4 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值为：", y_predict)
    print("比对真实值和预测值：", y_test == y_predict) # 比对真实值和预测值

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)
    # http://webgraphviz.com/查看
    return None


if __name__ == "__main__":
    decison_iris()