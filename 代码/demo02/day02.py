from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

if __name__ == "__main__":
    knn_iris()