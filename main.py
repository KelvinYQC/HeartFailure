#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

heart_dat = pd.read_csv('heart.csv')
#%%
print(heart_dat.isnull().sum())
#%%
print(heart_dat.describe())
#%%
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='RdYlBu')