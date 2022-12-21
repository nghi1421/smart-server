import json
import re
import numpy as np
import pandas as pd
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.models import Model
from keras import regularizers
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras_preprocessing.text import tokenizer_from_json
from keras import regularizers
import tensorflow as tf

url_full_train_data = "C:\\Users\\ACER\\Downloads\\CNN\\dts-phuclong.xlsx"
url_word2vec_full = "D:\\dts-phuclong_for_Word2vec.model"
val_data_full_from = 29000  # 3001
val_data_full_to = 29001  # 6003
pad = ['post', 'pre']
drop = 0.2
epoch = 20
batch_size = 128
max_length = 300
NUM_WORDS = 50000
EMBEDDING_DIM = 300
test_num_full = 3004
num_filters = 300
activation_func = "relu"
L2 = 0.004
filter_sizes = [3, 4, 5]

str = "quản lí bộ nhớ"
similar = []

def load_aspect_model(model_json,weight):
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weight)
    return model


def predict(str, tok_sam, sample_seq, model, label):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # ant
    if len(str_temp)>1:
        sentences.append(str_temp)
        text = tok_sam.texts_to_sequences(sentences)
        seq = pad_sequences(text, maxlen=sample_seq.shape[1], padding='post')
        pred = model.predict(seq) # predict là hàm của hệ thống
        temp_aspect_detect = label[np.argmax(pred)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def load_data(url_file_data, sheet_name, max_length):
    train_data = pd.read_excel(url_file_data, sheet_name)
    # test_data = train_data[34161:35463]  # lan 1: 34166:35465 | lan 2: 34161:35463
    # train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])
    return tokenizer, X_train

aspect_text = "mình suy sụp"
labels = ['pos', 'nev']
token_label1, sam_label1 = load_data(url_full_train_data, "Sheet1", 300)
label1_pred = predict(aspect_text, token_label1, sam_label1, load_aspect_model('D:\\model\\CNN_train_3c_relu.json', 'D:\\HDH\\dts-phuclong_raw_train_2c-001-0.0144-1.0000.h5'),labels)
print(label1_pred)