#分层模型集成框架stacking(叠加算法)

#简单堆叠3折CV分类
from sklearn import datasets
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X,y=iris.data[:,1:3],iris.target
RANDOM_SEED=42
# clf1=KNeighborsClassifier(n_neighbors=1)
# clf2=RandomForestClassifier(random_state=RANDOM_SEED)
# clf3=GaussianNB()
# lr=LogisticRegression()
#
# sclf=StackingCVClassifier(classifiers=[clf1,clf2,clf3],meta_classifier=lr,random_state=RANDOM_SEED)
# print('3-fold cross validation:\n')
# for clf,label in zip([clf1,clf2,clf3,sclf],['KNN','Random Forest','Naive Bayes','Stack']):
#     scores=cross_val_score(clf,X,y,cv=3,scoring='accuracy')
#     print("Accuract:%0.2f(+/- %0.2f)[%s]"%(scores.mean(),scores.std(),label))
# #画出决策边界
# gs=gridspec.GridSpec(2,2)
# fig=plt.figure(figsize=(10,8))
# for clf,lab,grd in zip([clf1,clf2,clf3,sclf],['KNN','Random Forest','Naive Bayes','Stack'],
#                        itertools.product([0,1],repeat=2)):
#     clf.fit(X,y)
#     ax=plt.subplot(gs[grd[0],grd[1]])
#     fig=plot_decision_regions(X=X,y=y,clf=clf)
#     plt.title(lab)
# plt.show()

#使用概率作为元特征
# clf1=KNeighborsClassifier(n_neighbors=1)
# clf2=RandomForestClassifier(random_state=RANDOM_SEED)
# clf3=GaussianNB()
# lr=LogisticRegression()
#
# sclf=StackingCVClassifier(classifiers=[clf1,clf2,clf3],use_probas=True,meta_classifier=lr,random_state=RANDOM_SEED)
# print('3-fold cross validation:\n')
# for clf,label in zip([clf1,clf2,clf3,sclf],['KNN','Random Forest','Naive Bayes','Stack']):
#     scores=cross_val_score(clf,X,y,cv=3,scoring='accuracy')
#     print("Accuract:%0.2f(+/- %0.2f)[%s]"%(scores.mean(),scores.std(),label))
#堆叠5折CV分类与网格搜索（结合网格搜索调参优化）：
clf1=KNeighborsClassifier(n_neighbors=1)
clf2=RandomForestClassifier(random_state=RANDOM_SEED)
clf3=GaussianNB()
lr=LogisticRegression()

sclf=StackingCVClassifier(classifiers=[clf1,clf2,clf3],meta_classifier=lr,random_state=RANDOM_SEED)
params={'kneighborsclassifier__n_neighbors':[1,5],
        'randomforestclassifier__n_estimators':[10,50],
        'meta_classifier__C':[0.1,10.0]}
grid=GridSearchCV(estimator=sclf,param_grid=params,cv=5,refit=True)
grid.fit(X,y)
cv_keys=('mean_test_score','std_test_score','params')
for r,_ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"%(grid.cv_results_[cv_keys[0]][r],
                                grid.cv_results_[cv_keys[1]][r]/2.0,
                                grid.cv_results_[cv_keys[2]][r]))
print("Best parameters:%s"%grid.best_params_)
print('Accuracy:%.2f'%grid.best_score_)