from sklearn.datasets import load_iris

def datasets_demo():
    #数据集的使用
    iris = load_iris()
    #print(iris)
    print(iris['DESCR'])
    return None


if __name__ == '__main__':
    datasets_demo()