#%%
from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import lightgbm

#%%
# loading the data
heart_dat = pd.read_csv('heart.csv')
print(heart_dat.columns)
print(heart_dat.skew())

print(heart_dat.isnull().sum())

print(heart_dat.describe())
#%%
# correlation graph
sns.set(font_scale=1)
sns.heatmap(heart_dat.corr(), annot=True, cmap='RdYlBu', fmt='.1f')
plt.show()
#%%
# encoder for categorical

encoder =LabelEncoder()

heart_dat['Sex']=encoder.fit_transform(heart_dat['Sex'])
heart_dat['RestingECG']=encoder.fit_transform(heart_dat['RestingECG'])
heart_dat['ChestPainType']=encoder.fit_transform(heart_dat['ChestPainType'])
heart_dat['ExerciseAngina']=encoder.fit_transform(heart_dat['ExerciseAngina'])
heart_dat['ST_Slope']=encoder.fit_transform(heart_dat['ST_Slope'])
#%%
# take out outlier

df = heart_dat
from scipy import stats
print(len(heart_dat))
print(len(df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]))

#%%
# devide data into traning and testing

x = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']
print(x.columns)

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#%%
logistic_result = list()
xgboost_result = list()
adaboosts_result = list()
rd_result = list()
SVC_result = list()
feature_result = list()


for i in range(500):
    # logistic regression
    logistic = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train, y_train)
    y_predict_logistic = logistic.predict(X_test)
    logis_ACC = accuracy_score(y_test, y_predict_logistic)
    logistic_result.append(logis_ACC)
    # xgboost
    xgboost = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.001, random_state=0)
    xgboostFitted = xgboost.fit(X_train, y_train)
    y_predict_xgboost = xgboostFitted.predict(X_test)
    xgBoost_acc = accuracy_score(y_test, y_predict_xgboost)
    xgboost_result.append(xgBoost_acc)
    # adaboost
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
    adaboost_fitted = adaboost.fit(X_train, y_train)
    y_pred_adaboost = adaboost_fitted.predict(X_test)
    adaboosts_acc = accuracy_score(y_test, y_pred_adaboost)
    adaboosts_result.append(adaboosts_acc)
    # random forest
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    rd_acc = accuracy_score(y_test, y_pred)
    rd_result.append(rd_acc)
    feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
    feature_result.append(feature_imp)

    # SVC
    SVC_model = svm.SVC(kernel='linear') # Linear Kernel
    SVC_model.fit(X_train, y_train)
    y_pred = SVC_model.predict(X_test)
    SVC_Acc = accuracy_score(y_test, y_pred)
    SVC_result.append(SVC_Acc)

    p_test = SVC_model.predict_proba(X_test)[:,1]
    testAuc = roc_auc_score(y_test, p_test)


#%%
    df_plot = pd.DataFrame({
            'LR': logistic_result,
            'XGboost': xgboost_result,
            'adaboosts': adaboosts_result,
            'RD': rd_result,
            'SVC': SVC_result,
    })

    plt.title('Model Accuracy for Different Algorithms')
    plt.ylabel('Accuracy Score')

    boxplot = df_plot.boxplot()
    boxplot.plot()

    plt.show()
#%%
print(X_test)
#%%
from xgboost import plot_importance
from matplotlib import pyplot

xgboost = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.001, random_state=0)
xgboostFitted = xgboost.fit(X_train, y_train)
y_predict_xgboost = xgboostFitted.predict(X_test)
#%%
feature_importance = list(xgboostFitted.feature_importances_)
colnamePlot = list(X_train.columns)

# Figure Size
fig, ax = plt.subplots(figsize=(16, 9))

# Horizontal Bar Plot
ax.barh(colnamePlot, feature_importance)

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

ax.invert_yaxis()

ax.set_title('Variable importance for XGboost Model',
             loc='left', )
sns.set(font_scale=0.7)

plt.show()