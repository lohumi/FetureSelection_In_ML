# Feature Selection methods :- 
#1- Univariate Selection 2- Feature Importance 3- Correlation Matrix with Heatmap


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Importing the dataset
 #dataset = requests.get(f"https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv")
'''dataset = pd.read_csv('train.csv')
X= dataset.iloc[:,0:20]
y= dataset.iloc[:,-1]

# 1- Univariate Selection
#apply SelectKBest class to extract top 10 best features
bestfeatures=SelectKBest(score_func=chi2,k=10)
fit= bestfeatures.fit(X,y)

dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)
#concat two dataframes
featureScores = pd.concat([dfscores,dfcolumns],axis=1)
#rename columns
featureScores.columns=['Score','Specs']
#print 10 best features
print (featureScores.nlargest(10,'Score'))'''
#************************************************************
#2- Feature Importance

'''dataset = pd.read_csv('train.csv')
X= dataset.iloc[:,0:20]
y= dataset.iloc[:,-1]
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
classifier=ExtraTreesClassifier()
classifier.fit(X,y)
print(classifier.feature_importances_)

series=pd.Series(classifier.feature_importances_,index=X.columns)
series.nlargest(10).plot(kind='barh')
plt.show()'''

#3- Correlation Matrix with Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('train.csv')
X= dataset.iloc[:,0:20]
y= dataset.iloc[:,-1]
#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


