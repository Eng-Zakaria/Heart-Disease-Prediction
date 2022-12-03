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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,MaxAbsScaler
data=pd.read_csv("Heart_Disease_Prediction.csv")
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test)
lRModel=LogisticRegression(penalty='l2',solver='lbfgs',C=20.0)

lRModel.fit(X_train, y_train)
print('LogisticRegressionModel Train Score is : ' , lRModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , lRModel.score(X_test, y_test))
print('LogisticRegressionModel Classes are : ' , lRModel.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , lRModel.n_iter_)
y_pred = lRModel.predict(X_test)
y_pred_prob = lRModel.predict_proba(X_test)
print('Predicted Value for LogisticRegressionModel is : ' , y_pred[:10])
print(y_train[:10])
print('Prediction Probabilities Value for LogisticRegressionModel is : ' , y_pred_prob[:10])
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

