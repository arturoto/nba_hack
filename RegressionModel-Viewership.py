
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from math import sqrt





df = pd.read_csv('Aggregated_Data.csv',header=0)
df['Total Households']=df['Away_DMA_TV_Households_Millions']+df['Home_DMA_TV_Households_Millions']
df['Network'].replace('ESPN2','ESPN',inplace=True)
df['avg_pace'] = (df['Home_Pace_Rating']+df['Away_Pace_Rating'])/2
    
df['Higher_Elo'] = df[['Home_Elo','Away_Elo']].max(axis=1)
df['Lower_Elo'] = df[['Home_Elo','Away_Elo']].min(axis=1)


df_viewershipmodel = df[['Total Viewership','Network','Total Households','Total_All_Stars','Holiday','Total_Top_Jersey_Sellers','Past_Playoff_Teams','Rivalry','avg_pace','Higher_Elo','Lower_Elo']]

df_viewershipmodel_Y = df_viewershipmodel.iloc[:,0]
df_viewershipmodel_X = df_viewershipmodel.iloc[:,1:]

#avg_viewership = df_viewershipmodel_Y.mean()

df_viewershipmodel_X = pd.concat([df_viewershipmodel_X, pd.get_dummies(df_viewershipmodel_X['Network'])], axis=1)
del df_viewershipmodel_X['Network']
del df_viewershipmodel_X['Regional']
del df_viewershipmodel_X['NBA TV']






X_viewership_train, X_viewership_test, y_viewership_train, y_viewership_test = train_test_split(df_viewershipmodel_X, df_viewershipmodel_Y, test_size=0.20, random_state=42)
 
regr_viewership = linear_model.LinearRegression()
regr_viewership.fit(X_viewership_train, y_viewership_train)
viewership_y_pred = regr_viewership.predict(X_viewership_test)

predicting = X_viewership_test
predicting.drop(predicting.index[[1,2]])



meanSquaredError = mean_squared_error(y_viewership_test , viewership_y_pred)
rootMeanSquaredError = sqrt(meanSquaredError)

plt.scatter(y_viewership_test, viewership_y_pred,  color='black')
plt.plot([1,4000], [1,4000], color='blue', linewidth=3)

plt.show()

df['SellOut'] = (df['Percent_Tickets_Sold']==1)
df_attendancemodel = df[['SellOut','Home_All_Star','Weekend','Home_Top_Jersey_Sellers','Away_Top_Jersey_Sellers','Past_Playoff_Teams','Rivalry','Home_Pace_Rating','Away_Pace_Rating','Home_Elo','Away_Elo']]

df_attendancemodel_Y = df_attendancemodel.iloc[:,0]
df_attendancemodel_X = df_attendancemodel.iloc[:,1:]

X_attendance_train, X_attendance_test, y_attendance_train, y_attendance_test = train_test_split(df_attendancemodel_X, df_attendancemodel_Y, test_size=0.20, random_state=42)
 
class_sellout = RandomForestClassifier(n_estimators=10, max_depth=None,)
class_sellout.fit(X_attendance_train, y_attendance_train)
sellout_y_pred = class_sellout.predict(X_attendance_test)
sellout_predicted_prob = class_sellout.predict_proba(X_attendance_test)

print("accuracy:",accuracy_score(y_attendance_test,sellout_y_pred))
importances = class_sellout .feature_importances_
std = np.std([tree.feature_importances_ for tree in class_sellout .estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]






print("Feature ranking:")

for f in range(df_attendancemodel_X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

df['Elo_Diff'] = df['Higher_Elo']-df['Lower_Elo']
df['Web_Viewing'] = df['Web_Viewing']/df['Web_Viewing'].mean()
    

df_WebInteractionsmodel = df[['Web_Viewing','Total_All_Stars','Total_Top_Jersey_Sellers','Elo_Diff']]

df_WebInteractionsmodel_Y = df_WebInteractionsmodel.iloc[:,0]
df_WebInteractionsmodel_X = df_WebInteractionsmodel.iloc[:,1:]

X_web_train, X_web_test, y_web_train, y_web_test = train_test_split(df_WebInteractionsmodel_X, df_WebInteractionsmodel_Y, test_size=0.20, random_state=42)
 
regr_web= linear_model.LinearRegression()
regr_web.fit(X_web_train, y_web_train)
web_y_pred = regr_web.predict(X_web_test)


print('Coefficients: \n', regr_web.coef_)

meanSquaredError = mean_squared_error(y_web_test , web_y_pred)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)

print('Variance score: %.2f' % r2_score(y_web_test , web_y_pred))
