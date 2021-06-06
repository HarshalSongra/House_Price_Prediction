from model import *
from sklearn.metrics import mean_squared_error
import numpy as np

'''Paths'''
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data.csv')
model_path = os.path.join(BASE_DIR, 'Model.joblib')

'''Load Model'''
print("----using Random Forest Regressor Model----")
mdl = load(model_path)

# testing the model on test data
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = mdl.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# print(final_predictions)
# print(list(Y_test))

print("The Final root mean squared error is: ",final_rmse)