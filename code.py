# --------------
# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Code starts here
data = pd.read_csv(path)
print(data.shape)
print(data.describe())
data = data.drop(['Serial Number'],axis=1)
print(data.shape)

# code ends here




# --------------
#Importing header files
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 11)   # Df = number of variable categories(in purpose) - 1

# Code starts here
return_rating = data['morningstar_return_rating'].value_counts()
print(return_rating)
risk_rating = data['morningstar_risk_rating'].value_counts()
print(risk_rating)
observed = pd.concat([return_rating.transpose(),risk_rating.transpose()],axis=1,keys=['return','risk'])
print(observed)
chi2, p, dof, ex = chi2_contingency(observed)
print("p value")
print(p)
print("Chi Statistic")
print(chi2)
if chi2 > critical_value:
    print("Null Hypothesis is Rejected")
else:
    print("Null Hypothesis is Accepted")
# Code ends here


# --------------
# Code starts here
correlation = abs(data.corr())
print(correlation)
us_correlation = correlation.unstack()
print(us_correlation)
us_correlation = us_correlation.sort_values(ascending = False)
print(us_correlation)
max_correlated = us_correlation[(us_correlation>0.75) & (us_correlation<1)]
print(max_correlated)
data.drop(['morningstar_rating', 'portfolio_stocks', 'category_12' ,'sharpe_ratio_3y'],axis=1,inplace=True)
print(data.shape)
# code ends here


# --------------
# Code starts here
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10,25))
ax_1.boxplot(data['price_earning'])
ax_1.set_title("price_earning")
ax_2.boxplot(data['net_annual_expenses_ratio'])
ax_2.set_title("net_annual_expenses_ratio")
# code ends here


# --------------
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# Code starts here
# independent variable 
X = data.drop(columns = 'bonds_aaa')
# target variable 
y = data.bonds_aaa
# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=3)
# Instantiate linear model
lr = LinearRegression()
# fit the model on training data
lr.fit(X_train,y_train)
# predict on test
y_pred = lr.predict(X_test)
#Calculate rmse
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
print("The RMSE Score For Simple Linear Model is {}".format(rmse.round(2)))
# Code ends here


# --------------
# import libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso

# regularization parameters for grid search
ridge_lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
lasso_lambdas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

# Code starts here
ridge_model = Ridge()
ridge_grid = GridSearchCV(estimator = ridge_model, param_grid = dict(alpha = ridge_lambdas))
ridge_grid.fit(X_train,y_train)
ridge_pred = ridge_grid.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(ridge_pred,y_test))
print("RMSE Score For Ridge Model is {}".format(ridge_rmse.round(2)))

lasso_model = Lasso()
lasso_grid = GridSearchCV(estimator = lasso_model, param_grid = dict(alpha = lasso_lambdas))
lasso_grid.fit(X_train,y_train)
lasso_pred = lasso_grid.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(lasso_pred,y_test))
print("RMSE Score For Lasso Model is {}".format(lasso_rmse.round(2)))
# Code ends here


