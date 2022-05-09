from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def datasets_demo():
    #数据集的使用
    iris = load_iris()
    #print(iris)
    #print(iris['DESCR'])
    #print(iris.feature_names)
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print(x_train)
    return None


if __name__ == '__main__':
    datasets_demo()