import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tqdm.notebook import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

data = pd.read_csv('input_encoding.txt',sep = ',')

#Encoding Authors from text to integers
from keras.utils import to_categorical
author_id = list(data.Author.unique())
integer_mapping = {x: i for i,x in enumerate(author_id)}

author_encoding = data.Author.map(integer_mapping)
data['Author_encoding'] = author_encoding

x_train, x_valid, y_train, y_valid = train_test_split(data.encoding, data.Author_encoding, test_size=0.2, shuffle=True)


def pad_features(sequences,seq_length):
  ''' Return features of review_ints, where each review is padded with 0's
or truncated to the input seq_length.
'''
  features = np.zeros((len(sequences),seq_length),dtype=int)
  for i,row in enumerate(sequences):
    features[i,-len(row):] = np.array(row)[:seq_length]
  return features
  
x_train_id = pad_features(x_train.str.split(),seq_length=128)
y_train_id = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='int')
x_valid_id = pad_features(x_valid.str.split(),seq_length=128)
y_valid_id = tf.keras.utils.to_categorical(y_valid, num_classes=10, dtype='int')

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train_id, y_train_id))
    .repeat()
    .shuffle(2048)
    .batch(64)
)


valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid_id, y_valid_id))
    .batch(64)
    .cache()
)

#LSTM only
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout

vocabulary_size = 20
max_words = 128
embedding_size=64

model=Sequential()

model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words,trainable = False))

model.add(layers.Bidirectional(LSTM(64,return_sequences=True)))
model.add(layers.Bidirectional(LSTM(64)))
model.add(Dense(10,activation = 'softmax'))
print(model.summary())


#A 2D CNN before LSTM
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,Input
from keras.layers.convolutional import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

vocabulary_size = 20
max_words = 128
embedding_size=100


model=Sequential()

#First embedding the input from (128,) to (128,100) in order to feed them into the LSTM
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

#reshape the input into (128,100,1) to feed it to the 2D CNN
model.add(tf.keras.layers.Reshape((128,100,1), input_shape=(128,100)))

#after padding = same, the output shape keeps the same as the input shape
model.add(Conv2D(filters = 128, kernel_size = 5, activation = 'relu',input_shape = (128,100,1),padding ='same'))

model.add(AveragePooling2D(pool_size=2,strides=2))
model.add((Flatten()))

#After flatten the convoluted values, reshape the output as the form of (batch size, features*timestep) from the pooled values, which
#has the shape of (64,50,1) -->1/2(128,100,1)
model.add(tf.keras.layers.Reshape((64,50*128), input_shape=(64,50,1)))

#Feed the output to the LSTM layer
model.add(layers.Bidirectional(LSTM(64,return_sequences=True)))
model.add(layers.Bidirectional(LSTM(64)))
model.add(Dense(10,activation = 'softmax'))

print(model.summary())

#use this to make topK categorical accuracy as top3 categorical accuracy
import functools
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

model.compile(Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy',top3_acc])

history = model.fit(
    train_dataset,
    steps_per_epoch=100,
    validation_data=valid_dataset,
    epochs=10)