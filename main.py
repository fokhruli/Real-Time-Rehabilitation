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
from data_processing import Test_Data_Loader
from graph import Graph
from sgcn_lstm import Stgcn_Lstm
#from stgcn import Stgcn
from sklearn.metrics import mean_squared_error, mean_absolute_error
random_seed = 42  # for reproducibility

data_loader = Data_Loader("Kimore ex1")

graph = Graph(len(data_loader.body_part))

train_x, valid_x, train_y, valid_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)
print("Training instances: ", len(train_x))
print("Validation instances: ", len(valid_x))

algorithm = Stgcn_Lstm(train_x, train_y, valid_x, valid_y, graph.AD, graph.AD2, epoach = 500)
model = algorithm.build_model()
history = algorithm.train()

