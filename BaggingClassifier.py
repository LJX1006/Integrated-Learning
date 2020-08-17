#基于bagging思想的套袋集成技术
#套袋方法可以提高不稳定模型的准确度的同时降低过拟合程度
#投票机制在训练每个分类器时都是用相同的全部样本，而bagging方法则是使用全部样本的一个随机抽样，每个
#分类器都是使用不同的样本进行训练，其他都是跟投票方法一模一样

from sklearn.model_selection import train_test_split  #划分训练集和测试集
from sklearn.preprocessing import StandardScaler #标准化数据
from sklearn.preprocessing import LabelEncoder #标签化分类变量
from sklearn.model_selection import cross_val_score #10折交叉验证评价模型
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.pipeline import  Pipeline #管道简化工作流
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import  roc_curve,auc,accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier

df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
                 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue',
                 'OD280/OD315 of diluted wines','Proline']
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
X=df_wine[['Alcohol','OD280/OD315 of diluted wines']].values
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
#使用单一决策树分类
tree=DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)
tree=tree.fit(x_train,y_train)
y_train_pred=tree.predict(x_train)
y_test_pred=tree.predict(x_test)
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'%(tree_train,tree_test))

#使用BaggingClassifier分类
bag=BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,
                      bootstrap=True,bootstrap_features=False,n_jobs=1,random_state=1)
bag=bag.fit(x_train,y_train)
y_train_pred=bag.predict(x_train)
y_test_pred=bag.predict(x_test)
bag_train=accuracy_score(y_train,y_train_pred)
bag_test=accuracy_score(y_test,y_test_pred)
print('BaggingClassifier train/test accuracies %.3f/%.3f'%(bag_train,bag_test))
