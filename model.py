from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import os
from joblib import dump, load


"""Getting current path"""
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data.csv')
model_path = os.path.join(BASE_DIR, 'Model.joblib')
# print(DATA_DIR)

# creating a DF of given csv file.
housingdf = pd.read_csv(DATA_DIR)


# creating testing and training data
'''
StratifiedShuffleSplit : Provides train/test indices to split data in train/test sets. Equally distriburted.
'''

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housingdf, housingdf['CHAS']):
    strat_train_set = housingdf.loc[train_index]
    strat_test_set = housingdf.loc[test_index]


housingdf = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

"""data processing"""

"""filling out missing data into the dataframe"""
imputer = SimpleImputer(strategy='median')
imputer.fit(housingdf)
# print(imputer.statistics_)
x = imputer.transform(housingdf)
housing_new = pd.DataFrame(x, columns=housingdf.columns)
# print(housing_new.describe())


"""creating a pipeline for filling up NA values in housingdf"""

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # 1
    # ..... no of operations are done one by one
    ('std_scaler', StandardScaler())  # 2
])

# housing numpy transformed array
housing_num_tr = my_pipeline.fit_transform(housing_new)


"""Selecting an accurate Model"""
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# testing the model on some data
# some_data = housingdf[:5]
# some_labels = housing_labels[:5]
# prepared_data = my_pipeline.transform(some_data)
# model.predict(prepared_data)
# print(list(some_labels))


"""Evaluating Model"""
housing_predictions = model.predict(housing_num_tr)
# finding out root mean squared Error
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

"""Using Better Evaluation technique - Cross Validation"""
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv = 10)
rmse_score = np.sqrt(-scores)


def print_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("standered Deviation : ", scores.std())

# print_scores(rmse_score)

# sems like model is predicting pretty good lets save the model
def save_model(model):
    try:
        dump(model, model_path)
    except Exception as e:
        print(e)

# save_model(model)

print(housingdf.columns[:14])