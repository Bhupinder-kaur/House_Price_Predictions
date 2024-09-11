
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv(r'C:\Users\bhupi\A VS CODE\1 Machine Learning\House Predictions\House_data.csv')
space=dataset['sqft_living']
price=dataset['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

pred=regressor.predict(xtest)

plt.scatter(xtrain,ytrain, color='purple')
plt.plot(xtrain,regressor.predict(xtrain), color='magenta')
plt.title("Visuals of Training DataSet")
plt.xlabel("space")
plt.ylabel("price")
plt.show()

plt.scatter(xtest, ytest, color= 'green')
plt.plot(xtrain, regressor.predict(xtrain), color = 'silver')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

y_2525 = regressor.predict([[2525]])
y_1000 = regressor.predict([[1000]])
print("Predicted House price for 2525 Sqaure feet : ${y_2525[0][0]:,.2f}")
print("Predicted House price for 1000 Sqaure feet : ${y_1000[0][0]:,.2f}")

# Predict
y_2525 = regressor.predict([[2525]])
y_1000 = regressor.predict([[1000]])
print("Predicted House price for 2525 Sqaure feet : ${y_2525[0][0]:,.2f}")
print("Predicted House price for 1000 Sqaure feet : ${y_1000[0][0]:,.2f}")

filename = 'houseprice_prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor,file)
    pickle.dump(regressor, file)
print("Model has been pickled and saved as houseprice_prediction_model.pkl")

from sklearn.metrics import r2_score,mean_squared_error
R2 = r2_score(ytest, pred)
MSE = mean_squared_error(ytest, pred)
RSME = np.sqrt(MSE)
print(' R-Square :{}'.format(R2),'\n','MSE :{} \n RSME : {}'.format(MSE,RSME))

# regression Table code
# introduce to OLS & stats.api
from statsmodels.api import OLS  # type: ignore
OLS(ytrain,xtrain).fit().summary()