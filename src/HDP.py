
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
df = pd.read_csv('Heart_Disease_Prediction.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


#%%



data=pd.read_csv("Heart_Disease_Prediction.csv")
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


def heart_diseases(x):
    if x == 'Presence':
        return 1
    if x == 'Absence':
        return 0

print(heart_diseases(data))


#%%

#data.describe()
#print(X.dtypes) all attri numerical
#print(y.dtypes) object


data['Heart Disease'] = data['Heart Disease'].apply(heart_diseases)

y = data.iloc[:,-1]

data.shape()
data.head()

le = LabelEncoder()
data['Heart Disease'] = le.fit_transform(data['Heart Disease'])
data.head()
#print👍
#print(data.isnull().sum()) no attri have null value
#print(format(len(data[data.duplicated()])))# no data have duplicate record
name = data.columns
#print(name)
num_var = [ 'BP', 'Cholesterol', 'Max HR','Exercise angina','ST depression','Number of vessels fluro','Thallium']
cat_var = [item for item in name if item not in num_var]
num_var_data = data[name& num_var]
correlation=num_var_data.corr()
#print(correlation)
sns.heatmap(num_var_data.corr(), cmap="YlGnBu", annot=True)
sns.pairplot(num_var_data)
num_var_data[num_var_data['Cholesterol'] > 500]
sns.pairplot(data, hue = 'Heart Disease')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle=True)
LRModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)
LRModel.fit(X_train, y_train)
y_pred = LRModel.predict(X_test)
y_pred_prob = LRModel.predict_proba(X_test)
print('Predicted Value for LogisticRegressionModel is : ' , y_pred[:10])
print(y_train[:10])
print('Prediction Probabilities Value for LogisticRegressionModel is : ' , y_pred_prob[:10])
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
print(accuracy_score(y_test,y_pred))

