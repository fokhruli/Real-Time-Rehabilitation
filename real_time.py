import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from data_processing import Data_Loader
import data_processing
from data_processing import Test_Data_Loader
from graph import Graph
from sgcn_lstm import Stgcn_Lstm
#from stgcn import Stgcn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import PyKinectBodyGame
random_seed = 42  # for reproducibility


data_loader = Data_Loader("Kimore ex5")
graph = Graph(len(data_loader.body_part))
train_x, valid_x, train_y, valid_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)
print("Training instances: ", len(train_x))
print("Validation instances: ", len(valid_x))
algorithm = Stgcn_Lstm(train_x, train_y, valid_x, valid_y, graph.AD, graph.AD2, epoach = 1000)


test_data_loader = Test_Data_Loader("Test_ex5")
model = algorithm.build_model()
model.load_weights("best model/best_model_ex5.hdf5")


predictions = []
for i in range(test_data_loader.scaled_x.shape[0]): 
    prediction = model.predict(test_data_loader.scaled_x[i].reshape(1,test_data_loader.scaled_x[i].shape[0],test_data_loader.scaled_x[i].shape[1],test_data_loader.scaled_x[i].shape[2]))
    predictions.append(prediction[0,0])

final_prediction = sum(predictions[1:]) / (len(predictions)-1)  

file1 = open('prediction.txt', 'w')
file1.write(str(final_prediction))

"""import numpy as np 
def sig(x):
 return 1/(1 + np.exp(-x))

sigmoid = sig(prediction[0,0]) 


scaled_prediction = data_processing.sc2.inverse_transform(prediction)

prediction[0,0] = 2"""

