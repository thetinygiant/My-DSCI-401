import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import seaborn as sns
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
ames_a = pd.read_csv('C:\GitHub\dsci_401\data\AmesHousingSetA.csv')
ames_a.head()
ames = ames_a.copy()
ames = ames.rename(columns = {'Year.Built' : 'Year_Built','Year.Remod.Add' : 'Year_Remod_Add','Garage.Yr.Blt':'Garage_Yr_Blt','Yr.Sold':'Yr_Sold'})
ames['Year_Built'] = pd.Categorical(ames.Year_Built)
ames['Year_Remod_Add'] = pd.Categorical(ames.Year_Built)
ames['Garage_Yr_Blt'] = pd.Categorical(ames.Garage_Yr_Blt)
ames['Yr_Sold'] = pd.Categorical(ames.Yr_Sold)
#Yr.Sold column is in years but should be treated as a string
del ames['PID']
#Must turn qualitative predictors to quantitative through one hot encoding 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
#Getting Dummy Variables
ames_2 = pd.get_dummies(ames, columns = cat_features(ames))
#Getting data_x, and data_y
features = list(ames_2)
features.remove('SalePrice')
data_x = ames_2[features]
data_y = ames_2['SalePrice']
#Getting values for missing values in the columns
imp = preprocessing.Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
data_x = imp.fit_transform(data_x)
data_x_norm = preprocessing.normalize(data_x, axis = 0)

#Looking at data plots to decide predictors
#ames_2.plot.scatter(x = 'Misc.Val', y = 'SalePrice')
#plt.show()
#ames_2.plot.scatter(x = 'Pool.QC', y = 'SalePrice')
#ames_2.plot.scatter(x = 'Total.Bsmt.SF', y = 'SalePrice')
#ames_2.plot.scatter(x = 'X1st.Flr.SF', y = 'SalePrice')
#ames_2.plot.scatter(x = 'Garage.Area'. y = 'SalePrice')

#Train-Test-Split for training data
x_train, x_test, y_train, y_test = train_test_split(data_x_norm,data_y, test_size = 0.2, random_state = 4)
base_mod = linear_model.LinearRegression()
base_mod.fit(x_train,y_train)
preds = base_mod.predict(x_test)
print('R^2(Base Model): ' + str(r2_score(y_test,preds)))

alphas = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,10.0,12.0,15.0,20.0]
for a in alphas:
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train, y_train)
	preds = lasso_mod.predict(x_test)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))
#Alpha of 15 seems to be the best option
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)]))
print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))							   
#Applying transforms to Ames_B dataset
ames_b = pd.read_csv('C:\GitHub\dsci_401\data\AmesHousingSetB.csv')
del ames_b['PID']
ames_b = ames_b.rename(columns = {'Year.Built' : 'Year_Built','Year.Remod.Add' : 'Year_Remod_Add','Garage.Yr.Blt':'Garage_Yr_Blt','Yr.Sold':'Yr_Sold'})
ames_b['Year_Built'] = pd.Categorical(ames_b.Year_Built)
ames_b['Year_Remod_Add'] = pd.Categorical(ames_b.Year_Built)
ames_b['Garage_Yr_Blt'] = pd.Categorical(ames_b.Garage_Yr_Blt)
ames_b['Yr_Sold'] = pd.Categorical(ames_b.Yr_Sold)
ames_b = pd.get_dummies(ames_b, columns = cat_features(ames_b))
features_b = list(ames_b)
features_b.remove('SalePrice')
data_x_b = ames_b[features_b]
data_y_b = ames_b['SalePrice']
impb = preprocessing.Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
data_x_b = impb.fit_transform(data_x_b)
data_x_norm_b = preprocessing.normalize(data_x_b, axis = 0)
#Testing model on Dataset B 
lasso_mod_b = linear_model.Lasso(alpha=0.25, normalize=True, fit_intercept=True)
lasso_mod_b.fit(data_x_norm_b, data_y_b)
predsb = lasso_mod_b.predict(data_x_norm_b)
print('R^2 (Lasso Model with alpha=' + str(0.25) + '): ' + str(r2_score(data_y_b, predsb)))
pprint.pprint(pd.DataFrame({'Actual':data_y_b, 'Predicted':predsb}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(data_y_b, predsb), \
							   median_absolute_error(data_y_b, predsb), \
							   r2_score(data_y_b, predsb), \
							   explained_variance_score(data_y_b, predsb)]))

#Base Model for Test Data
base_modb = linear_model.LinearRegression()
base_modb.fit(data_x_norm_b,data_y_b)
preds_base = base_modb.predict(data_x_norm_b)
print('R^2(Base Model): ' + str(r2_score(data_y_b,preds_base)))


