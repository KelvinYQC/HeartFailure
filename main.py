#%%
from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
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
sns.set(font_scale=0.7)
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
logistic_result = list()
xgboost_result = list()
adaboosts_result = list()
rd_result = list()
SVC_result = list()
feature_result = list()
knn_result = list()


logistic_precision = list()
xgboost_precision = list()
adaboosts_precision = list()
rd_precision = list()
SVC_precision = list()
feature_precision = list()
knn_precision = list()


logistic_recall = list()
xgboost_recall = list()
adaboosts_recall = list()
rd_recall = list()
SVC_recall = list()
knn_recall = list()


logistic_specificity = list()
xgboost_specificity = list()
adaboosts_specificity = list()
rd_specificity = list()
SVC_specificity = list()
knn_specificity = list()

logistic_sensitivity = list()
xgboost_sensitivity = list()
adaboosts_sensitivity = list()
rd_sensitivity = list()
SVC_sensitivity = list()
knn_sensitivity = list()



#%%
for i in range(500):
    print(i)
    # logistic regression
    logistic = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train, y_train)
    y_predict_logistic = logistic.predict(X_test)
    logis_ACC = accuracy_score(y_test, y_predict_logistic)
    logistic_result.append(logis_ACC)

    lrprecision = precision_score(y_test, y_predict_logistic)
    logistic_precision.append(lrprecision)

    lrrecall = recall_score(y_test, y_predict_logistic)
    logistic_recall.append(lrrecall)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict_logistic).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    logistic_specificity.append(specificity)
    logistic_sensitivity.append(sensitivity)


    # xgboost
    xgboost = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.001, random_state=0)
    xgboostFitted = xgboost.fit(X_train, y_train)
    y_predict_xgboost = xgboostFitted.predict(X_test)
    xgBoost_acc = accuracy_score(y_test, y_predict_xgboost)
    xgboost_result.append(xgBoost_acc)

    xgprecision = precision_score(y_test, y_predict_xgboost)
    xgboost_precision.append(xgprecision)
    xgboost_recall.append(recall_score(y_test, y_predict_xgboost))

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict_xgboost).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    xgboost_specificity.append(specificity)
    xgboost_sensitivity.append(sensitivity)



    # adaboost
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=0)
    adaboost_fitted = adaboost.fit(X_train, y_train)
    y_pred_adaboost = adaboost_fitted.predict(X_test)
    adaboosts_acc = accuracy_score(y_test, y_pred_adaboost)
    adaboosts_result.append(adaboosts_acc)

    adaprecision = precision_score(y_test, y_pred_adaboost)
    adaboosts_precision.append(adaprecision)

    adaboosts_recall.append(recall_score(y_test, y_pred_adaboost))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adaboost).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    adaboosts_specificity.append(specificity)
    adaboosts_sensitivity.append(sensitivity)



    # random forest
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    rd_acc = accuracy_score(y_test, y_pred)
    rd_result.append(rd_acc)
    feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
    feature_result.append(feature_imp)

    rdprecision = precision_score(y_test, y_pred)
    rd_precision.append(rdprecision)
    rd_recall.append(recall_score(y_test, y_pred))


    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    rd_specificity.append(specificity)
    rd_sensitivity.append(sensitivity)



    # SVC
    SVC_model = svm.SVC(kernel='linear') # Linear Kernel
    SVC_model.fit(X_train, y_train)
    y_pred = SVC_model.predict(X_test)
    SVC_Acc = accuracy_score(y_test, y_pred)
    SVC_result.append(SVC_Acc)

    svcprecision = precision_score(y_test, y_pred)
    SVC_precision.append(svcprecision)
    SVC_recall.append(recall_score(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    SVC_specificity.append(specificity)
    SVC_sensitivity.append(sensitivity)

    # SVC
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    KNN_Acc = accuracy_score(y_test, y_pred)
    knn_result.append(KNN_Acc)

    knnprecision = precision_score(y_test, y_pred)
    knn_precision.append(knnprecision)
    knn_recall.append(recall_score(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    knn_specificity.append(specificity)
    knn_sensitivity.append(sensitivity)


#%%
df_plot = pd.DataFrame({
        'LR': logistic_result,
        'XGboost': xgboost_result,
        'adaboosts': adaboosts_result,
        'RD': rd_result,
        'SVC': SVC_result,
        'knn': knn_result
    })

plt.title('Model Accuracy for Different Algorithms')
plt.ylabel('Accuracy Score')

boxplot_acc = df_plot.boxplot()
boxplot_acc.plot()
plt.show()


#%%
df_plot_precision = pd.DataFrame({
        'LR': logistic_precision,
        'XGboost': xgboost_precision,
        'adaboosts': adaboosts_precision,
        'RD': rd_precision,
        'SVC': SVC_precision,
        'knn': knn_precision

})

plt.title('Precision for Different Algorithms')
plt.ylabel('Precision Score')
boxplot_precision = df_plot_precision.boxplot()
boxplot_precision.plot()
plt.show()
#%%
df_plot_recall = pd.DataFrame({
        'LR': logistic_recall,
        'XGboost': xgboost_recall,
        'adaboosts': adaboosts_recall,
        'RD': rd_recall,
        'SVC': SVC_recall,
        'knn': knn_recall

})

plt.title('Recall for Different Algorithms')
plt.ylabel('Recall Score')
boxplot_recall = df_plot_recall.boxplot()
boxplot_recall.plot()
plt.show()

#%%
df_plot_specificity = pd.DataFrame({
        'LR': logistic_specificity,
        'XGboost': xgboost_specificity,
        'adaboosts': adaboosts_specificity,
        'RD': rd_specificity,
        'SVC': SVC_specificity,
        'knn': knn_specificity

})

plt.title('Specificity for Different Algorithms')
plt.ylabel('Specificity Score')
boxplot_specificity = df_plot_recall.boxplot()
boxplot_specificity.plot()
plt.show()
#%%
df_plot_sensitivity = pd.DataFrame({
        'LR': logistic_sensitivity,
        'XGboost': xgboost_sensitivity,
        'adaboosts': adaboosts_sensitivity,
        'RD': rd_sensitivity,
        'SVC': SVC_sensitivity,
        'knn': knn_sensitivity

})

plt.title('Sensitivity for Different Algorithms')
plt.ylabel('Sensitivity Score')
boxplot_sensitivity = df_plot_sensitivity.boxplot()
boxplot_sensitivity.plot()
plt.show()

#%%
plt.subplot(211)
boxplot_acc = df_plot.boxplot()
boxplot_acc.plot()
plt.subplot(212)
boxplot_precision = df_plot_precision.boxplot()
boxplot_precision.plot()
plt.subplot(213)
boxplot_recall = df_plot_recall.boxplot()
boxplot_recall.plot()
plt.show()

#%%
import statistics
print(statistics.stdev(xgboost_precision))
print(statistics.stdev(xgboost_recall))
print(statistics.stdev(xgboost_result))


#%%
# XGboost details

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
