import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,MaxAbsScaler
data=pd.read_csv("../Dataset/Heart_Disease_Prediction.csv")
data['Max HR'].skew()

X=data[['BP','Cholesterol','Max HR','ST depression','Thallium','Slope of ST','Exercise angina']]
data['Cholesterol'].describe()
Q1= 213.000000
print(Q1)
Q3=280
print(Q3)

IQR=280-213
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR
data['Cholesterol'].skew()

print((data['Cholesterol']<Q1-1.5*IQR)|(data['Cholesterol']>Q3+1.5*IQR))
data['Cholesterol'].quantile(0.90)
data['Cholesterol']=np.where((data['Cholesterol']<Q1-1.5*IQR),194.8,data['Cholesterol'])
data['Cholesterol']=np.where(data['Cholesterol']>Q3+1.5*IQR,309.0,data['Cholesterol'])
data['Cholesterol'].skew()


plt.boxplot(data['BP'])
data['BP'].describe()
Q1=data['BP'].quantile(0.25)
print(Q1)
Q3=data['BP'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR
data['BP'].skew()

print((data['BP']<Q1-1.5*IQR)|(data['BP']>Q3+1.5*IQR))
data['BP'].quantile(0.90)
data['BP']=np.where((data['BP']<Q1-1.5*IQR),data['BP'].quantile(0.10),data['BP'])
data['BP']=np.where(data['BP']>Q3+1.5*IQR,data['BP'].quantile(0.90),data['BP'])
data['BP'].skew()



plt.boxplot(data['Max HR'])
data['Max HR'].describe()
Q1=data['Max HR'].quantile(0.25)
print(Q1)
Q3=data['Max HR'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['Max HR']<Q1-1.5*IQR)|(data['Max HR']>Q3+1.5*IQR))
data['Max HR'].quantile(0.90)
data['Max HR']=np.where((data['Max HR']<Q1-1.5*IQR),data['Max HR'].quantile(0.10),data['Max HR'])
data['Max HR']=np.where(data['Max HR']>Q3+1.5*IQR,data['Max HR'].quantile(0.90),data['Max HR'])
data['Max HR'].skew()


plt.boxplot(data['ST depression'])
data['ST depression'].describe()
Q1=data['ST depression'].quantile(0.25)
print(Q1)
Q3=data['ST depression'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['ST depression']<Q1-1.5*IQR)|(data['ST depression']>Q3+1.5*IQR))
data['ST depression'].quantile(0.90)
data['ST depression']=np.where((data['ST depression']<Q1-1.5*IQR),data['ST depression'].quantile(0.10),data['ST depression'])
data['ST depression']=np.where(data['ST depression']>Q3+1.5*IQR,data['ST depression'].quantile(0.90),data['ST depression'])
data['ST depression'].skew()
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

plt.boxplot(data['ST depression'])

plt.boxplot(data['Number of vessels fluro'])
data['Number of vessels fluro'].describe()
Q1=data['Number of vessels fluro'].quantile(0.25)
print(Q1)
Q3=data['Number of vessels fluro'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['Number of vessels fluro']<Q1-1.5*IQR)|(data['Number of vessels fluro']>Q3+1.5*IQR))
data['Number of vessels fluro'].quantile(0.90)
data['Number of vessels fluro']=np.where((data['Number of vessels fluro']<Q1-1.5*IQR),data['Number of vessels fluro'].quantile(0.10),data['Number of vessels fluro'])
data['Number of vessels fluro']=np.where(data['Number of vessels fluro']>Q3+1.5*IQR,data['Number of vessels fluro'].quantile(0.90),data['Number of vessels fluro'])
data['Number of vessels fluro'].skew()


plt.boxplot(data['Thallium'])
data['Thallium'].describe()
Q1=data['Thallium'].quantile(0.25)
print(Q1)
Q3=data['Thallium'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['Thallium']<Q1-1.5*IQR)|(data['Thallium']>Q3+1.5*IQR))
data['Thallium'].quantile(0.90)
data['Thallium']=np.where((data['Thallium']<Q1-1.5*IQR),data['Thallium'].quantile(0.10),data['Thallium'])
data['Thallium']=np.where(data['Thallium']>Q3+1.5*IQR,data['Thallium'].quantile(0.90),data['Thallium'])
data['Thallium'].skew()



plt.boxplot(data['Slope of ST'])
data['Slope of ST'].describe()
Q1=data['Slope of ST'].quantile(0.25)
print(Q1)
Q3=data['Slope of ST'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['Slope of ST']<Q1-1.5*IQR)|(data['Slope of ST']>Q3+1.5*IQR))
data['Slope of ST'].quantile(0.90)
data['Slope of ST']=np.where((data['Slope of ST']<Q1-1.5*IQR),data['Slope of ST'].quantile(0.10),data['Slope of ST'])
data['Slope of ST']=np.where(data['Slope of ST']>Q3+1.5*IQR,data['Slope of ST'].quantile(0.90),data['Slope of ST'])
data['Slope of ST'].skew()



plt.boxplot(data['Age'])
data['Age'].describe()
Q1=data['Age'].quantile(0.25)
print(Q1)
Q3=data['Age'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
Min=Q1-1.5*IQR
Max=Q3+1.5*IQR

print((data['Age']<Q1-1.5*IQR)|(data['Age']>Q3+1.5*IQR))
data['Age'].quantile(0.90)
data['Age']=np.where((data['Age']<Q1-1.5*IQR),data['Age'].quantile(0.10),data['Age'])
data['Age']=np.where(data['Age']>Q3+1.5*IQR,data['Age'].quantile(0.90),data['Age'])
data['Age'].skew()
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
#data.describe()
#print(X.dtypes) all attri numerical
#print(y.dtypes) object
def heart_diseases(x):  #function convert objects into numerical
    if x=='Presence':
        return 1
    if x=='Absence':
        return 0 
data['Heart Disease']=data['Heart Disease'].apply(heart_diseases)

y=data['Heart Disease']
data.corr()
data.shape
data.head()
name = data.columns
num_var = ['Age', 'BP', 'Cholesterol', 'Max HR', 'Heart Disease']
cat_var = [item for item in name if item not in num_var]

num_var_data = data[data.columns & num_var]
num_var_data.describe()
num_var_data.corr()
sns.heatmap(num_var_data.corr(), cmap="YlGnBu", annot=True)
sns.pairplot(num_var_data)
num_var_data[num_var_data['Cholesterol'] > 500]
sns.pairplot(num_var_data, hue = 'Heart Disease')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,shuffle=False)
sc_x = MinMaxScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
lRModel=LogisticRegression(penalty='l2',solver='liblinear',C=10.0,max_iter=10000,random_state=0)
lRModel.fit(X_train, y_train)
print('LogisticRegressionModel Train Score is : ' , lRModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , lRModel.score(X_test, y_test))
print('LogisticRegressionModel Classes are : ' , lRModel.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , lRModel.n_iter_)
r_sq = lRModel.score(X, y)
print(f"coefficient of determination: {r_sq}")
y_pred = lRModel.predict(X_test)
y_pred_prob = lRModel.predict_proba(X_test)

print('Predicted Value for LogisticRegressionModel is : ' , y_pred[:10])
print(y_train[:10])
print('Prediction Probabilities Value for LogisticRegressionModel is : ' , y_pred_prob[:10])
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
print("accurcy =",accuracy_score(y_test,y_pred))


