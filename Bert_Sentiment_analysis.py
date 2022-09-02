import transformers
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split




def read_data(file):
    data = pd.read_csv(file)
    return data

def train_test_split(data):
    x_train, x_valid, y_train, y_valid = train_test_split(data.sentence.astype(str), data.score.values, test_size=0.15, shuffle=True)
    return (x_train,x_valid,y_train,y_valid)


def encode(texts, tokenizer, max_length):
    ids = []
    for text in texts:
        id = tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)
        ids.append(id)
    return(np.array(ids))

def build_model(transformer, loss, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    x = tf.keras.layers.Dropout(0.2)(cls_token)
    out = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1.5e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-chinese')
        model = build_model(transformer_layer)
        print(model.summary())
    return model

def main():
    your_data = read_data('YOUR_PATH')
    x_train, x_valid, y_train, y_valid = train_test_split(your_data.sentence.astype(str), your_data.score.values, test_size=0.15, shuffle=True)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
    x_train_id = encode(x_train, tokenizer, max_length=32)
    x_valid_id = encode(x_valid, tokenizer, max_length=32)
    y_train_id = tf.keras.utils.to_categorical(y_train, num_classes=7, dtype='int')
    y_valid_id = tf.keras.utils.to_categorical(y_valid, num_classes=7, dtype='int')
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
    model = build_model(loss='binary_crossentropy', max_len=32)
    history = model.fit(
        train_dataset,
        steps_per_epoch=200,
        validation_data=valid_dataset,
        epochs=10
    )

if __name__ == "__main__":
    main()
