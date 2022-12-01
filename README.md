# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1. Import the needed packages

2.Assigning hours To X and Scores to Y

3.Plot the scatter plot

4.Use mse,rmse,mae formmula to find
 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Gayathri A

RegisterNumber:  212221230028
*/
```
import numpy as np
import pandas as pd
dataset=pd.read_csv("student_scores.csv")
print(dataset.iloc[0:10])
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression() 
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title('Training set(H vs S) ')
plt.xlabel('Hours')
plt.ylabel('Scores')
y_pred=reg.predict(x_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,reg.predict(x_test),color='red')
plt.title('Test set(H vs S) ')
plt.xlabel('Hours')
plt.ylabel('Scores')
mse=mean_squared_error(y_test,y_pred)
print("MSE ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE ",mae)
rmse=np.sqrt(mse)
print("RMSE ",rmse)
```

## Output:

![21](https://user-images.githubusercontent.com/94154854/205006945-bd944f87-9507-4714-bda3-fef8e721df90.png)


![ml2 1](https://user-images.githubusercontent.com/94154854/193325253-1a8592a0-a0b0-4794-ad41-cae622231174.png)

![ml2 2](https://user-images.githubusercontent.com/94154854/193325279-20360f3d-9db7-4ebd-81d5-cac16c305b37.png)

![ml2 3](https://user-images.githubusercontent.com/94154854/193325301-9f16d029-fe9d-42e2-aa18-4455c4f0b540.png)

![ml2 4](https://user-images.githubusercontent.com/94154854/193325333-a315ec24-e9ac-4216-aa10-9eadcb74ff30.png)

![ml2 5](https://user-images.githubusercontent.com/94154854/193325366-b4337813-8313-46bb-ad69-bfc8627aedd8.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
