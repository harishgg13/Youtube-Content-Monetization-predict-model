import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

pd.set_option("display.max_columns",None)
X=pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/X_feature")
y=pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/Y_feature")

# ------------------------------------------------------------------------------
# Checking Correlation
# print(X.corr()) # there the no strong correlation between input features

# Checking VIF
X_Vif=pd.DataFrame()
X_Vif=X
X_Vif=sm.add_constant(X_Vif)
vif=pd.DataFrame()
vif["Features"]=X_Vif.keys()
vif["values"]=[variance_inflation_factor(X_Vif.values,i) for i in range(X_Vif.shape[1])]
print(vif)
# print(X)
# print(y)

# ------------------------------------------------------------------------------
# Training the model
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# ------------------------------------------------------------------------------
# Linear Regression
model=LinearRegression()
model.fit(x_train,y_train)

#predict
y_train_prediction_LR=model.predict(x_train)
y_train_actual_LR=y_train

y_test_prediction_LR=model.predict(x_test)
y_test_actual_LR=y_test

# ------------------------------------------------------------------------------
# Linear Regression-Evaluation
mse_train_score_LR=mean_squared_error(y_train_actual_LR,y_train_prediction_LR)
r2_train_LR=r2_score(y_train_actual_LR,y_train_prediction_LR)
print("MSE_train_LR:",mse_train_score_LR)
print("R2_train_LR:",r2_train_LR) 

mse_test_score_LR=mean_squared_error(y_test_actual_LR,y_test_prediction_LR)
r2_test_LR=r2_score(y_test_actual_LR,y_test_prediction_LR)
print("MSE_test_LR:",mse_test_score_LR)
print("R2_test_LR:",r2_test_LR) 
# test:0.2
# R2 is 0.9989 Mse is 4.066 for DataCleaning. 
# R2 is 0.9938 Mse is 23.47477 for DC2. 
# R2 is 0.9482 Mse is 198.5 for DC3.
# joblib.dump(model, "model_LR.pkl")


'''# ------------------------------------------------------------------------------
# Decision Tree Regression
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
#predict
y_prediction_DTR=dtr.predict(x_test)
y_actual_DTR=y_test

# ------------------------------------------------------------------------------
# Decision Tree Regression-Evaluation
mse_score_DTR=mean_squared_error(y_actual_DTR,y_prediction_DTR)
r2_DTR=r2_score(y_actual_DTR,y_prediction_DTR)
print("mse_score_DTR :",mse_score_DTR)
print("r2_DTR :",r2_DTR) 
# test:0.2
# R2 is 0.9973 Mse is 10.2022 for DataCleaning. 
# R2 is 0.9876 Mse is 47.2139 for DC2. 
# R2 is 0.8980 Mse is 391.001 for DC3

# ------------------------------------------------------------------------------
# random forest regression

model_rfr=RandomForestRegressor(n_estimators=100,random_state=42)
model_rfr.fit(x_train,y_train)

#predict

y_prediction_rfr=model_rfr.predict(x_test)
y_actual_rfr=y_test

# ------------------------------------------------------------------------------
# random forest regression - Evaluation

mse_score_rfr=mean_squared_error(y_actual_rfr,y_prediction_rfr)
r2_rfr=r2_score(y_actual_rfr,y_prediction_rfr)
print("mse_score_rfr",mse_score_rfr)
print("r2_rfr",r2_rfr)

# test:0.2
# R2 is 0.9987 Mse is 4.66 for DataCleaning. 
# R2 is 0.9876 Mse is 47.2139 for DC2. 
# R2 is 0.8980 Mse is 391.001 for DC3

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Gradient boosting regression
model_gb=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,random_state=42,max_depth=3)
model_gb.fit(x_train,y_train)
#predict
y_prediction_gb=model_gb.predict(x_test)
y_actual_gb=y_test

# ------------------------------------------------------------------------------
# gradient boosting - Evaluation
mse_score_gb=mean_squared_error(y_actual_gb,y_prediction_gb)
r2_gb=r2_score(y_actual_gb,y_prediction_gb)
print("mse_score_gb",mse_score_gb)
print("r2_gb",r2_gb)
# test:0.2
# R2 is 0.9987 Mse is 4.8825 for DataCleaning. 
# R2 is 0.9938 Mse is 23.4887 for DC2. 
# R2 is 0.9479 Mse is 199.442 for DC3'''
