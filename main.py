#%%
from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import lightgbm

#%%
from sklearn.svm import LinearSVC

heart_dat = pd.read_csv('heart.csv')
print(heart_dat.columns)
print(heart_dat.skew())

print(heart_dat.isnull().sum())

print(heart_dat.describe())
#%%
sns.set(font_scale=1)
sns.heatmap(heart_dat.corr(), annot=True, cmap='RdYlBu', fmt='.1f')
plt.show()
#%%

encoder =LabelEncoder()

heart_dat['Sex']=encoder.fit_transform(heart_dat['Sex'])
heart_dat['RestingECG']=encoder.fit_transform(heart_dat['RestingECG'])
heart_dat['ChestPainType']=encoder.fit_transform(heart_dat['ChestPainType'])
heart_dat['ExerciseAngina']=encoder.fit_transform(heart_dat['ExerciseAngina'])
heart_dat['ST_Slope']=encoder.fit_transform(heart_dat['ST_Slope'])
#%%
df = heart_dat
from scipy import stats
print(len(heart_dat))
print(len(df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]))

#%%
x = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']
print(x.columns)

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#%%
logistic = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train, y_train)
y_predict_logistic = logistic.predict(X_test)
print(accuracy_score(y_test, y_predict_logistic))


xgboost = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.001, random_state=0)
xgboostFitted = xgboost.fit(X_train, y_train)
y_predict_xgboost = xgboostFitted.predict(X_test)
print(accuracy_score(y_test, y_predict_xgboost))



adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
adaboost_fitted = adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost_fitted.predict(X_test)
print(accuracy_score(y_test, y_pred_adaboost))

#%%
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(feature_imp)

#%%

clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

