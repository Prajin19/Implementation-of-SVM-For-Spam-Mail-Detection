# Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and split the data into messages and labels, then into training and test sets.
2. Convert messages to numeric features using TF-IDF vectorization.
3. Train an SVM model to separate spam from ham messages.
4. Predict on test data and evaluate accuracy and classification metrics.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Prajin S
RegisterNumber:  212223230151
*/
```

```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
df=pd.read_csv('spam.csv',encoding='Windows-1252')
df.head()
df.tail()
df.info()
X=df['v2'].values
Y=df['v1'].values
X.shape
Y.shape
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
vec=TfidfVectorizer()
xtrain=vec.fit_transform(xtrain)
xtest=vec.transform(xtest)
xtrain
xtest
svc=SVC()
svc.fit(xtrain,ytrain)
ypred=svc.predict(xtest)
ypred
acc=accuracy_score(ytest,ypred)
print(acc)
res=classification_report(ytest,ypred)
print(res)
```

## Output:
![image](https://github.com/user-attachments/assets/1041425b-9012-4c9c-89c5-df2775282d74)

![image](https://github.com/user-attachments/assets/17c16aba-02d2-4d2e-9d12-5dd92a4ce5b9)

![image](https://github.com/user-attachments/assets/9a4e7ca3-59c1-42bd-84fd-1c68fc4faaae)

![image](https://github.com/user-attachments/assets/658ea6b0-4e77-4624-957a-41ae155b80af)

![image](https://github.com/user-attachments/assets/6f4f031a-9182-4b5d-8679-166a36fcba26)

![image](https://github.com/user-attachments/assets/1a946395-6c8e-42cb-8332-03a1eaf86dcc)

![image](https://github.com/user-attachments/assets/3dab26f3-0c47-45fc-a102-0b6fb2e4855d)

![image](https://github.com/user-attachments/assets/0f4a2060-f2b7-4af2-85fe-143c902e0ec0)

![image](https://github.com/user-attachments/assets/0f244f32-3cfd-44c3-8af6-589c5b262d39)

![image](https://github.com/user-attachments/assets/024e3102-644f-40fc-9442-fd3cf1fd1f4f)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
