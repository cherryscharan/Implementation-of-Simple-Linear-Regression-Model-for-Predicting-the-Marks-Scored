# *Developed by: NaveenKumar.T*
# *Register Number: 212223220067*
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

*Program to implement the simple linear regression model for predicting the marks scored.*
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## Dataset:

![278995575-c7816d33-6dab-45e2-8d19-9a11e9583cb5](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/77f02a64-78ed-4399-9259-689112602f13)

## Head values:

![278996479-7f3d7783-4601-4e70-989f-2ccbf87d0765](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/2228870b-adf9-4236-bf2a-f93acbc08937)

## Tail values:

![278996533-5343e114-fe3a-4ad7-8058-6b81db462fdc](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/0e784047-1170-45a3-bd22-43cd181fff54)

## X and Y values:

![278996577-f84947e0-99a3-444c-8286-c59cc0660a4e](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/a972a1b3-dcc7-47a8-864c-5bf42cf65a47)

## Predication values of X and Y:

![278996605-6ea46100-8530-4491-821e-079308a1eef5](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/3f2f6b09-9e82-4dcc-84fa-392a8e05569f)

## MSE,MAE and RMSE:

![278996622-0f3750f1-fec0-4008-abcf-7e7b971d82a9](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/1829f39a-a952-4061-8bc3-4f87430672c9)

## Training Set:

![278996909-088c3714-a70d-4ef0-b952-1d26c48e1fa8](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/7ef0574c-8a40-4ac3-8d7a-8f8395ea02fa)

## Testing Set:

![278996855-aa18e6a5-11f7-410e-bbd6-89c052ff52a6](https://github.com/cherryscharan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146930617/2a3f0b60-a2bb-4604-be0e-2cb6770e9d81)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
