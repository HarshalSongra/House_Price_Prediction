from joblib import dump, load
import numpy as np
import os
from model import *

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data.csv')
model_path = os.path.join(BASE_DIR, 'Model.joblib')

mdl = load(model_path)

attributes  = ['CRIM', 'ZN', 'INDUS ', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


print("\n\n--------House Price Prediction--------")
print(f"\nAttributes : {attributes}")
values = input("Enter values: ")

splitted_att_values = values.split(',')
att_values = [float(val) for val in splitted_att_values]

'''
# -0.43942006,  7.12628155, -1.12165014, -0.27288841, -1.42262747, -0.24640239, -0.24640239, 2.61111401, -1.0016859, -1.0016859, -0.97491834,  -0.97491834, -0.86091034
'''

# print(f"\nGiven Attributes : {att_values}")
att_values_np = np.array([att_values])
# print(att_values_np)
# exit()

# Passing the given data through pipeline. 
prepared_attribute_values = my_pipeline.transform([att_values])
print(f"\n Pipelined Data : {prepared_attribute_values}")

result = mdl.predict(prepared_attribute_values)
print(f"\nThe Price of the House according to the given attributes is : {result}\n\n")