# This file is adapted from Tensorflow Text Generation with RNN official tutorial 
import tensorflow as tf
import numpy as np
import time
from operator import invert
import re

def make_input_target(sequence):
    input_text = sequence[:-1]
    target = sequence[1:]
    return input_text, target
  
def char_index_transfer(text,seq_length):
    path_to_file = tf.keras.utils.get_file('DATASET_FROM_KERAS')
    text = open(path_to_file,'r').read().decode(encoding = 'utf-8')
    vocab =sorted(set(text))
    char_to_num = tf.keras.layers.StringLookup(vocabulary = list(vocab),mask_token = None)
    num_to_char = tf.keras.layers.StringLookup(vocabulary = char_to_num.get_vocabulary(),invert = True, mask_token = None)
    vocab_size = len(char_to_num.get_vocabulary())
    all_ids = char_to_num(tf.strings.unicode_split(text,'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(make_input_target)
    return vocab_size, ids_dataset,sequences,dataset
  

def shuffle_data(dataset):
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE,drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset
    
   
class GenModel():
    def __init__(self,vocab_size,embedding_dim,rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,return_sequence = True, return_state = True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, states = None,return_state = False, training =False):
        x = inputs
        x = self.embedding(x,training = training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
         

model = GenModel(vocab_size= vocab_size, embedding_dim=256,rnn_units=512)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)


class SingleStep(tf.keras.Model):
    def __init__(self,model,num_to_char,char_to_num,temparature = 1.0):
        super().__init__()
        self.temperature = temparature
        self.model = model
        self.num_to_char = num_to_char
        self.char_to_num = char_to_num

        mask_index = self.char_to_num(['[UNK]'])[:,None]
        masks = tf.SparseTensor(
            values = [-float('inf')]*len(mask_index),
            indices=mask_index,
            dense_shape= [len(char_to_num.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(masks)
    
    def prediction_one_step(self,inputs,states = None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.char_to_num(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)

        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        predicted_logits = predicted_logits + self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.num_to_char(predicted_ids)

    # Return the characters and model state.
        return predicted_chars, states
        
