import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import naive_bayes
from sklearn.metrics import roc_curve
from sklearn import neighbors
from data_util import*
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('HR_comma_sep.csv')
data.head() #
data = data.rename(columns={'sales': 'Job_Type'})
data = data.sample(frac=1, random_state = 4).reset_index(drop=True) #Shuffling data set because column "left" was presorted.
#Performing Data Transformations
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
data = pd.get_dummies(data, columns = cat_features(data))
#Breaking up data into training data and validation data
training  = data.iloc[0:7501]
testing = data.iloc[7501:]
testing = testing.reset_index(drop=True)
#pd.plotting.scatter_matrix(training, diagonal = 'kde')
#plt.show()
training['last_evaluation'].median()   #.72 median  Above average workers 
features = list(training)
features.remove('left')
data_x = training[features]
data_y = training['left']

#Creating model using voting method
x_train, x_test,y_train,y_test = train_test_split(data_x,data_y,test_size = 0.3,random_state = 4)
m1 = svm.SVC()
m2 = naive_bayes.GaussianNB()
m3 = ensemble.RandomForestClassifier()
m4 = neighbors.KNeighborsClassifier(n_neighbors=3)
m5 = log_model = linear_model.LogisticRegression()
voting_mod = ensemble.VotingClassifier(estimators=[('svm',m1),('nb',m2),('rf',m3),('knn',m4),('log',m5)],voting = 'hard')
param_grid = {'svm__C':[0.2,0.5,1.0,2.0,5.0,10.0],'rf__n_estimators':[5,10,15,100],'rf__max_depth':[3,6,None]}
best_voting_mod  = GridSearchCV(estimator=voting_mod,param_grid = param_grid,cv = 5)
best_voting_mod.fit(x_train,y_train)
print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test,y_test)))  #score = 0.95735
#testing the model on validation set
features = list(testing)
features.remove('left')
data_x = testing[features]
data_y = testing['left']
preds = best_voting_mod.predict(data_x)
print('ROC AUC: ' + str(roc_auc_score(data_y,preds)))
#print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(data_y,preds)))
#print("Confusion Matrix: "+ str(confusion_matrix(data_y, preds)))

