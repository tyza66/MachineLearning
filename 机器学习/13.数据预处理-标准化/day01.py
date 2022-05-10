from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

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

def cut_word(text):
    out = " ".join(list(jieba.cut(text)))
    return out

def count_chinese_demo():
    #中文文本特征抽取 自动分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    transfer = CountVectorizer(stop_words=["一种","所以"])
    data_final = transfer.fit_transform(data_new)
    print(data_final.toarray())
    print("特征名：", transfer.get_feature_names_out())
    return None

def tfidf_demo():
    #用tfidf的方法进行文版特征抽取
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])
    data_final = transfer.fit_transform(data_new)
    print(data_final.toarray())
    print("特征名：", transfer.get_feature_names_out())
    return None

def minmax_demo():
    #归一化
    #获取数据，实例化转换器类，带哦用fit_transform转换
    #data = pd.read_table("datingTestSet2.txt")
    data = pd.read_csv("datingTestSet2.txt")
    data = data.iloc[:, :3]
    transfer = MinMaxScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None

def stand_demo():
    #标准化
    data = pd.read_csv("datingTestSet2.txt")
    data = data.iloc[:, :3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


if __name__ == '__main__':
    stand_demo()