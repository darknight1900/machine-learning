import numpy as np
import pandas as pd
import time

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score

random_state = 100
feats = data.columns.values.tolist()
data_copy = data.copy()

pred_lables = {}

for feat in feats:
    print "Select " + feat + " as target output"
    data_copy = data.copy()
    y_data = data_copy[feat].as_matrix()
    
    data_copy.drop([feat], axis = 1, inplace = True)
    X_data = data_copy.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=random_state)
    regressor = DecisionTreeRegressor(random_state=100)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    _r2_score = r2_score(y_test, y_pred)

    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f' % _r2_score)
    pred_lables[feat] = _r2_score

print pred_lables

###############################################################################
# feature scaling 