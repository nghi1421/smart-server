import json
import numpy as np
import pandas as pd
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras_preprocessing.text import tokenizer_from_json
from keras import regularizers

url_full_train_data = "C:\\Users\\ACER\\Downloads\\CNN\\dts-phuclong.xlsx"
url_word2vec_full = "D:\\dts-phuclong_for_Word2vec.model"
pad = 'post'
epoch = 100
batch_size = 128
max_length = 300
NUM_WORDS = 50000
EMBEDDING_DIM = 300
test_num_full = 3004
L2 = 0.004


tokenizer = Tokenizer(num_words=50000)
labels = ['pos', 'nev']
with open('D:\\Jobs\\HV BCVT\\Do an TN\\HK1 - 2020-2021\\Le Thi Hong Anh\Resources\\vocab.json') as f:
    data = json.load(f)
dictionary = tokenizer_from_json(data)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text,filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n\'')
    wordIndices = []
    for word in words:
        if word in dictionary.word_docs:
            wordIndices.append(dictionary.word_docs[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# read in your saved model structure
# json_file = open('model.json', 'r')
json_file = open('D:\\model\\CNN_train_3c_relu.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
model.load_weights('D:\\HDH\\dts-phuclong_raw_train_2c-001-0.0144-1.0000.h5')

# okay here's the interactive part
sentence = []

evalSentence = "tôi chán"

evalSentence = evalSentence.lower()

testArr = convert_text_to_index_array(evalSentence)

    # print(testArr)
sentence.append(testArr)
sentence = pad_sequences(sentence, maxlen=300, padding='post')
pred = model.predict(sentence)
print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
del evalSentence
sentence = []
