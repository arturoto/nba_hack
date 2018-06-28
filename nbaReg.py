import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import statsmodels.api as saP

data = pd.read_csv('Aggregated_Data.csv')


X = data.iloc[:, 8:-1]
y = data.iloc[:, -1]

#X.to_csv('what.csv')

#print(X.describe())

regress = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
regress.fit(X_train, y_train)

#print(regress.coef_)

y_prediction = regress.predict(X_test)
#print(regress.score(X_test, y_test))

x_sm = saP.add_constant(X)
model = saP.OLS(y, X).fit()
#print(model.summary())

X.plot()
plt.show()


#print(y_test)
#print(y_prediction[1])
#print(np.mean((y_prediction - y_test)**2))
#print(len(X_train))
#print(len(y_train))

'''
plt.scatter(X, y, color = 'red')
plt.plot(X, regress.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''