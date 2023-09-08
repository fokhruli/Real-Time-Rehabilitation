import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras.callbacks import ModelCheckpoint

class Stgcn_Lstm():
    def __init__(self, train_x, train_y, valid_x, valid_y, AD, AD2, lr=0.0001, epoach=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.AD = AD
        self.AD2 = AD2
        self.lr = lr
        self.epoach =epoach
        self.batch_size = batch_size
                
    def sgcn(self, Input):
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(Input)
        x = Dropout(0.25)(x)
        gcn_1 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([self.AD, x])
        y = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(Input)
        y = Dropout(0.25)(y)
        gcn_2 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([self.AD2, y])
        gcn = concatenate([gcn_1, gcn_2], axis=-1)                                                                                                                                   
        
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu')(gcn)
        x = Dropout(0.25)(x)
        gcn_1 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([self.AD, x])
        y = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu')(gcn)
        y = Dropout(0.25)(y)
        gcn_2 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([self.AD2, y])
        gcn = concatenate([gcn_1, gcn_2], axis=-1)
        
        gcn = tf.keras.layers.Reshape(target_shape=(-1,gcn.shape[2]*gcn.shape[3]))(gcn)
        return gcn  

    def Lstm(self, x):
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(40, return_sequences=True)(rec)
        rec1 = Dropout(0.25)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.25)(rec2)
        rec3 = LSTM(80)(rec2)
        rec3 = Dropout(0.25)(rec3)
        output = Dense(1, activation = 'linear')(rec3)
        return output
    
    def build_model(self):
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        x = self.sgcn(seq_input)
        out = self.Lstm(x)
        self.model = Model(seq_input, out)
        return self.model
    
    def train(self):
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), optimizer= tf.keras.optimizers.Adam(lr=self.lr))
        checkpoint = ModelCheckpoint("best model/best_model_ex1.hdf5", monitor='val_loss', save_best_only=True, mode='auto', period=1)
        
        history = self.model.fit(self.train_x, self.train_y, validation_data = (self.valid_x,self.valid_y), epochs=self.epoach, batch_size=self.batch_size, callbacks=[checkpoint])
        return history
    
    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred
