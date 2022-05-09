from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def datasets_demo():
    #数据集的使用
    iris = load_iris()
    #print(iris)
    #print(iris['DESCR'])
    #print(iris.feature_names)
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print(x_train)
    return None

def dict_demo():
    #字典特征抽取
    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':31}]
    #实例化转换器，调用transfrom
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print(data_new)
    print("特征名：",transfer.get_feature_names_out())
    return None

def count_demo():
    #文本特征抽取
    data = ["I am a cat,i like python!","You are a cat,you do not like python!"]
    #实例化转换器，调用fit_transform
    transfer = CountVectorizer(stop_words=["cat"])
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())
    print("特征名：", transfer.get_feature_names_out())
    return None

def count_chinese_demo():
    #中文文本特征抽取 自动分词
    data = ["I am a cat,i like python!","You are a cat,you do not like python!"]
    #实例化转换器，调用fit_transform
    transfer = CountVectorizer(stop_words=["cat"])
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())
    print("特征名：", transfer.get_feature_names_out())
    return None

if __name__ == '__main__':
    count_chinese_demo()