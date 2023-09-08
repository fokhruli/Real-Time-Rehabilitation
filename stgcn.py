import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras.callbacks import ModelCheckpoint

class Stgcn():
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
        #self.distributed = tf.distribute.TPUStrategy()
                
    def stgcn(self, Input, no_filter):
          x = tf.keras.layers.Conv2D(filters=no_filter, kernel_size=(1,1), strides=1, activation='relu')(Input)
          gcn = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([self.AD, x])                                                                                                                                          
          #gcn = Dropout(0.5)(gcn)
          z = tf.keras.layers.Conv2D(no_filter, (9,1), padding='same', activation='relu')(gcn)                                                                                                                                   
          return z  
  
    def Lstm(self, x):
        x = tf.keras.layers.Reshape(target_shape=(-1,x.shape[2]*x.shape[3]))(x)
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
        x = self.stgcn(seq_input, 64)
        x1 = self.stgcn(x, 64)
        
        x2 = self.stgcn(x1, 64)
        x2 = x2 + x1
        
        x3 = self.stgcn(x2, 64)
        x3 = x3 + x2
        
        x4 = self.stgcn(x3, 64)
        x4 = x4 + x3
        
        x5 = self.stgcn(x4, 64)
        x5 = x5 + x4
        
        x6 = self.stgcn(x5, 128)
        
        x7 = self.stgcn(x6, 128)
        x7 = x7 + x6
        
        x8 = self.stgcn(x7, 128)
        
        x8 = tf.keras.layers.GlobalAveragePooling2D()(x8)
        out = Dense(1, activation='linear')(x8)
        
        #out = self.Lstm(x8)

        #x8 = Flatten()(x8)
        #x8 = Dropout(0.25)(x8)
        #fc1 = Dense(80, activation='relu')(x8)
        #fc1 = Dropout(0.25)(fc1)
        #out = Dense(1, activation='linear')(fc1)
        
        self.model = Model(seq_input, out)
        #self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), optimizer= tf.keras.optimizers.Adam(lr=self.lr))
        #checkpoint = ModelCheckpoint("best model/best_model.hdf5", monitor='val_loss', save_best_only=True, mode='auto', period=1)
      
        #history = self.model.fit(self.train_x, self.train_y, validation_data = (self.valid_x,self.valid_y), epochs=self.epoach, batch_size=self.batch_size, callbacks=[checkpoint])
        return self.model
    
    def train(self):
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), optimizer= tf.keras.optimizers.Adam(lr=self.lr))
        checkpoint = ModelCheckpoint("best model/best_model_ex5.hdf5", monitor='val_loss', save_best_only=True, mode='auto', period=1)
        history = self.model.fit(self.train_x, self.train_y, validation_data = (self.valid_x,self.valid_y), epochs=self.epoach, batch_size=self.batch_size, callbacks=[checkpoint])
        return history        
    
    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred
