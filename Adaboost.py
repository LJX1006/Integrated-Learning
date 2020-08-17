#基于boosting思想的自适应增强方法
#与Bagging相比，Boosting思想可以降低偏差

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
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

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

#使用Adaboost集成建模
ada=AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
ada=ada.fit(x_train,y_train)
y_train_pred=ada.predict(x_train)
y_test_pred=ada.predict(x_test)
ada_train=accuracy_score(y_train,y_train_pred)
ada_test=accuracy_score(y_test,y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'%(ada_train,ada_test))

#观察决策树与Adaboost异同
x_min=x_train[:,0].min()-1
x_max=x_train[:,0].max()+1
y_min=x_train[:,1].min()-1
y_max=x_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
f,axarr=plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(12,6))
for idx,clf,tt in zip([0,1],[tree,ada],['Decision tree','Adaboost']):
    clf.fit(x_train,y_train)
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,z,alpha=0.3)
    axarr[idx].scatter(x_train[y_train==0,0],x_train[y_train==0,1],c='blue',marker='^')
    axarr[idx].scatter(x_train[y_train==1, 0], x_train[y_train == 1, 1], c='red', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0,-0.2,s='OD280/OD315 of diluted wines',ha='center',va='center',fontsize=12,transform=axarr[1].transAxes)
plt.show()
