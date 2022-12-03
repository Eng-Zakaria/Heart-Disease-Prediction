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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,shuffle=False)
sc_x = MinMaxScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
lRModel=LogisticRegression(penalty='l2',solver='lbfgs',C=20,max_iter=10000,random_state=0)
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


print("-----------------------------------------------------------------------")

from sklearn.svm import SVR,SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,  shuffle =False)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
SVRModel = SVR(C = 100.0 ,epsilon=0.1,kernel = 'linear',max_iter=100000).fit(X_train,y_train)
print('SVRModel Train Score is : ' , SVRModel.score(X_train, y_train))
print('SVRModel Test Score is : ' , SVRModel.score(X_test, y_test))
y_pred = SVRModel.predict(X_test)
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred = clf.predict(X_test)
print("accurcy =",accuracy_score(y_test,y_pred))
print('-------------------------------------------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,  shuffle =False)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
SVCModel = SVC(kernel= 'poly',# it can be also linear,poly,sigmoid,precomputed
               max_iter=100,C=1.0)
SVCModel.fit(X_train, y_train)

#Calculating Details
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = SVCModel.predict(X_test)
print('Predicted Value for SVCModel is : ' , y_pred[:10])

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
