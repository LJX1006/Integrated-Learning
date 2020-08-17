#基于投票思想的多数票机制的集成分类器
from sklearn.datasets import load_iris #以iris鸢尾花数据集为例
from sklearn.model_selection import train_test_split  #划分训练集和测试集
from sklearn.preprocessing import StandardScaler #标准化数据
from sklearn.preprocessing import LabelEncoder #标签化分类变量
from sklearn.model_selection import cross_val_score #10折交叉验证评价模型
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.pipeline import  Pipeline #管道简化工作流
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import  roc_curve,auc
import matplotlib.pyplot as plt

#数据预处理
data=load_iris()
X,y=data.data[50:,[1,2]],data.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=1,stratify=y)

#使用三种不同的分类器：逻辑回归+决策树+k-近邻分类器
model1=LogisticRegression(penalty='l2',C=0.001,random_state=1)
model2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
model3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1=Pipeline([['sc',StandardScaler()],['model',model1]])
pipe3=Pipeline([['sc',StandardScaler()],['model',model3]])
model_labels=['Logistic regression','Decision tree','KNN']
print('10-folds cross validation :\n')
for model,label in zip([pipe1,model2,pipe3],model_labels):
    scores=cross_val_score(estimator=model,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC:%0.2f(+/- %0.2f)[%s]"%(scores.mean(),scores.std(),label))

#使用MajorityVoteClassifier集成
mv_model=VotingClassifier(estimators=[('pipe1',pipe1),('model2',model2),('pipe3',pipe3)],voting='soft')
model_labels+=['MajorityVoteClassifier']
print('10-folds cross validation :\n')
for model,label in zip([pipe1,model2,pipe3,mv_model],model_labels):
    scores=cross_val_score(estimator=model,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC:%0.2f(+/- %0.2f)[%s]"%(scores.mean(),scores.std(),label))

#使用ROC曲线评估集成分类器
colors=['black','orange','blue','green']
linestyle=[':','--','-.','-']
plt.figure(figsize=(10,6))
for model,label,color,ls in zip([pipe1,model2,pipe3,mv_model],model_labels,colors,linestyle):
    y_pred=model.fit(x_train,y_train).predict_proba(x_test)[:,1]
    fpr,tpr,trhresholds=roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc=auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=color,linestyle=ls,label='%s(auc=%0.2f)'%(label,roc_auc))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=2)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xlabel('False positive rate(FPR)')
plt.xlabel('True positive rate(TPR)')
plt.show()