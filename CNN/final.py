import json
import re
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

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

label1_weight_file = "Weights/label1.h5"

c4_ptbn_label3_weight_file = "Weights/c4_ptbn_label3.h5"
c4_ptbn_weight_file = "Weights/c4_ptbn.h5"
c4_pdbn_weight_file = "Weights/c4_pdbn.h5"
c4_pdbn_label3_weight_file = "Weights/c4_pdbn_label3.h5"
c4_pcbn_weight_file = "Weights/c4_pcbn.h5"
c4_pcbn_kn_weight_file = "Weights/c4_pcbn_kn.h5"
c4_pcbn_gt_weight_file = "Weights/c4_pcbn_gt.h5"
c4_pcbn_ch_weight_file = "Weights/c4_pcbn_ch.h5"
c4_dcvdlq_weight_file = "Weights/c4_dcvdlq.h5"
c4_dc_kn_weight_file = "Weights/c4_dc_kn.h5"
c4_dc_gt_weight_file = "Weights/c4_dc_gt.h5"
c4_dc_ch_weight_file = "Weights/c4_dc_ch.h5"
c4_bna_weight_file = "Weights/c4_bna.h5"
c4_bna_kn_weight_file = "Weights/c4_bna_kn.h5"
c4_bna_gt_weight_file = "Weights/c4_bna_gt.h5"
c4_bna_ch_weight_file = "Weights/c4_bna_ch.h5"

c3_tttn_weight_file = "Weights/c3_tttn.h5"
c3_tttn_label3_weight_file = "Weights/c3_tttn_label3.h5"
c3_kntt_weight_file = "Weights/c3_kntt.h5"
c3_kntt_label3_weight_file = "Weights/c3_kntt_label3.h5"
c3_dptt_weight_file = "Weights/c3_dptt.h5"
c3_dptt_label3_weight_file = "Weights/c3_dptt_label3.h5"

c2_tq_weight_file = "Weights/c2_tq.h5"
c2_tq_kn_weight_file = "Weights/c2_tq_kn.h5"
c2_tq_mt_weight_file = "Weights/c2_tq_mt.h5"
c2_tq_cht_weight_file = "Weights/c2_tq_cht.h5"
c2_qlkg_weight_file = "Weights/c2_qlkg.h5"
c2_qlkg_kn_weight_file = "Weights/c2_qlkg_kn.h5"
c2_qlkg_mt_weight_file = "Weights/c2_qlkg_mt.h5"
c2_qlkg_cht_weight_file = "Weights/c2_qlkg_cht.h5"
c2_ktmr_weight_file = "Weights/c2_ktmr.h5"
c2_ktmr_label3_weight_file = "Weights/c2_ktmr_label3.h5"

c1_label2_weight_file = "Weights/c1_label2.h5"
c1_knhdh_weight_file = "Weights/c1_knhdh.h5"
c1_cthdh_mt_weight_file = "Weights/c1_cthdh_mt.h5"
c1_cthdh_kn_weight_file = "Weights/c1_cthdh_kn.h5"
c1_cthdh_cht_weight_file = "Weights/c1_cthdh_cht.h5"

# json file
label1_weight_json = "label1_model_full.json"

c4_ptbn_label3_weight_json = "c4_ptbn_label3_model_full.json"
c4_ptbn_weight_json = "c4_ptbn_model_full.json"
c4_pdbn_weight_json = "c4_pdbn_model_full.json"
c4_pdbn_label3_weight_json = "c4_pdbn_label3_model_full.json"
c4_pcbn_weight_json = "c4_pcbn_model_full.json"
c4_pcbn_kn_weight_json = "c4_pcbn_kn_model_full.json"
c4_pcbn_gt_weight_json = "c4_pcbn_gt_model_full.json"
c4_pcbn_ch_weight_json = "c4_pcbn_ch_model_full.json"
c4_dcvdlq_weight_json = "c4_dcvdlq_model_full.json"
c4_dc_kn_weight_json = "c4_dc_kn_model_full.json"
c4_dc_gt_weight_json = "c4_dc_gt_model_full.json"
c4_dc_ch_weight_json = "c4_dc_ch_model_full.json"
c4_bna_weight_json = "c4_bna_model_full.json"
c4_bna_kn_weight_json = "c4_bna_kn_model_full.json"
c4_bna_gt_weight_json = "c4_bna_gt_model_full.json"
c4_bna_ch_weight_json = "c4_bna_ch_model_full.json"

c3_tttn_weight_json = "c3_tttn_model_full.json"
c3_tttn_label3_weight_json = "c3_tttn_label3_model_full.json"
c3_kntt_weight_json = "c3_kntt_model_full.json"
c3_kntt_label3_weight_json = "c3_kntt_label3_model_full.json"
c3_dptt_weight_json = "c3_dptt_model_full.json"
c3_dptt_label3_weight_json = "c3_dptt_label3_model_full.json"

c2_tq_weight_json = "c2_tq_model_full.json"
c2_tq_kn_weight_json = "c2_tq_kn_model_full.json"
c2_tq_mt_weight_json = "c2_tq_mt_model_full.json"
c2_tq_cht_weight_json = "c2_tq_cht_model_full.json"
c2_qlkg_weight_json = "c2_qlkg_model_full.json"
c2_qlkg_kn_weight_json = "c2_qlkg_kn_model_full.json"
c2_qlkg_mt_weight_json = "c2_qlkg_mt_model_full.json"
c2_qlkg_cht_weight_json = "c2_qlkg_cht_model_full.json"
c2_ktmr_weight_json = "c2_ktmr_model_full.json"
c2_ktmr_label3_weight_json = "c2_ktmr_label3_model_full.json"

c1_label2_weight_json = "c1_label2_model_full.json"
c1_knhdh_weight_json = "c1_knhdh_model_full.json"
c1_cthdh_mt_weight_json = "c1_cthdh_mt_model_full.json"
c1_cthdh_kn_weight_json = "c1_cthdh_kn_model_full.json"
c1_cthdh_cht_weight_json = "c1_cthdh_cht_model_full.json"

EMBEDDING_DIM = 300
NUM_WORDS = 50000
max_length1 = 1000
max_length2 = 300
pad = ['post', 'pre']
test_num_full = 400




url_an_toan_data = "MyData/dataset_HDH_fix.xlsx"


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

# [PhysicalDevice(name='/physical_device:GPU:0',device_type='GPU')]
# tf.config.list_physical_devices('GPU')

# a  = load_aspect_model(label1_weight_json, label1_weight_file);
# print (a)
# load_full_data_an_toan();

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# config.gpu_options.allow_growth = True
label1 = ['c1', 'c2', 'c3', 'c4']
c1_label2 = ['cthdh','knhdh']
c1_cthdh_kn = ['kn','none']
c1_cthdh_mt = ['mt','none']
c1_cthdh_cht = ['cht','none']
c1_knhdh = ['kn','mt']
c2_ktmr = ['ktmr','none']
c2_qlkg = ['qlkg','none']
c2_tq = ['tq','none']
c2_ktmr_label3 = ['kn','cht']
c2_qlkg_kn = ['kn','none']
c2_tq_kn = ['kn','none']
c2_qlkg_mt = ['mt','none']
c2_tq_mt = ['mt','none']
c2_qlkg_cht = ['cht','none']
c2_tq_cht = ['cht','none']
c3_dptt = ['none', 'dp_tt']
c3_kntt_label3 = ['kn','gt']
c3_kntt = ['none', 'kn_tt']
c3_dptt_label3 = ['kn', 'gt']
c3_tttn = ['none', 'tt_tn']
c3_tttn_label3 = ['kn', 'gt']
c4_bna = ['none','bna']
c4_dc = ['none','dc_vdlq']
c4_pcbn = ['none','pcbn']
c4_ptbn = ['none','ptbn']
c4_pdbn = ['none','pdbn']
c4_bna_kn = ['none','kn']
c4_bna_gt = ['none','gt']
c4_bna_ch = ['none','ch']
c4_dc_kn = ['none','kn']
c4_dc_gt = ['none','gt']
c4_dc_ch = ['none','ch']
c4_pcbn = ['none','pcbn']
c4_pcbn_kn = ['none','kn']
c4_pcbn_gt = ['none','gt']
c4_pcbn_ch = ['none','ch']
c4_ptbn_label3 = ['ch','gt']
c4_pdbn_label3 = ['ch','gt']

aspect_text = "Hệ điều hành là gì?"

prediction_label1 ="";
prediction_label2 ="";
prediction_label3 ="";

text_answer = ""
# ví dụ
token_label1, sam_label1 = load_data("MyData/label1.xlsx","Sheet1",max_length1)
label1_pred = predict(aspect_text, token_label1, sam_label1, load_aspect_model(label1_weight_json, label1_weight_file),label1)
if label1_pred == ['c1']:
  prediction_label1 = label1_pred[0]
  token_c1_label2, sam_c1_label2 = load_data("MyData/c1_label2.xlsx","c1",max_length1)
  c1_label2_pred = predict(aspect_text, token_c1_label2, sam_c1_label2, load_aspect_model(c1_label2_weight_json, c1_label2_weight_file),c1_label2)
  if c1_label2_pred == ['cthdh']:
    prediction_label2 =  c1_label2_pred[0];
    token_c1_cthdh_kn, sam_cthdh_kn = load_data("MyData/c1_label3.xlsx","c1_cthdh_kn")
    c1_cthdh_kn_pred = predict(aspect_text, token_c1_cthdh_kn, sam_cthdh_kn, load_aspect_model(c1_cthdh_kn_weight_json, c1_cthdh_kn_weight_file),c1_cthdh_kn)
    if c1_cthdh_kn_pred == ['kn']:
       prediction_label3 =  c1_cthdh_kn_pred[0];
    elif c1_cthdh_kn_pred == ['none']:
       token_c1_cthdh_mt, sam_cthdh_mt = load_data("MyData/c1_label3.xlsx","c1_cthdh_mt",max_length1)
       c1_cthdh_mt_pred = predict(aspect_text, token_c1_cthdh_mt, sam_cthdh_mt, load_aspect_model(c1_cthdh_mt_weight_json, c1_cthdh_mt_weight_file),c1_cthdh_mt)
       if c1_cthdh_mt_pred == ['mt']:
        prediction_label3 =  c1_cthdh_mt_pred[0];
       elif c1_cthdh_mt_pred == ['none']:
        token_c1_cthdh_cht, sam_cthdh_cht = load_data("MyData/c1_label3.xlsx","c1_cthdh_cht",max_length1)
        c1_cthdh_cht_pred = predict(aspect_text, token_c1_cthdh_cht, sam_cthdh_cht, load_aspect_model(c1_cthdh_cht_weight_json, c1_cthdh_cht_weight_file),c1_cthdh_cht)
        if c1_cthdh_cht_pred == ['cht']:
          prediction_label3 =  c1_cthdh_cht_pred[0];
        elif c1_cthdh_cht_pred == ['none']:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
  elif c1_label2_pred == ['knhdh']:
    prediction_label2 =  c1_label2_pred[0];
    token_c1_knhdh_label3, sam_knhdh_label3 = load_data("MyData/c1_label3.xlsx","c1_cthdh_kn",max_length1)
    c1_knhdh_label3_pred = predict(aspect_text, token_c1_knhdh_label3, sam_knhdh_label3, load_aspect_model(c1_knhdh_weight_json, c1_knhdh_weight_file),c1_knhdh)
    if c1_knhdh_label3_pred == ['kn']:
       prediction_label3 =  c1_knhdh_label3_pred[0];
    elif c1_knhdh_label3_pred == ['mt']:
       prediction_label3 =  c1_knhdh_label3_pred[0];
    else :
       print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
  else:
    print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
elif label1_pred == ['c2']:
  prediction_label1 = label1_pred[0]
  token_c2_ktmr, sam_c2_ktmr = load_data("MyData/c2_label2.xlsx","c2_ktmr",max_length2)
  c2_ktmr_pred = predict(aspect_text, token_c2_ktmr, sam_c2_ktmr, load_aspect_model(c2_ktmr_weight_json, c2_ktmr_weight_file),c2_ktmr)
  if c2_ktmr_pred == ['ktmr']:
    prediction_label2 =  c2_ktmr_pred[0];
    token_c2_ktmr_label3, sam_c2_ktmr_label3 = load_data("MyData/c2_label3.xlsx","c2_ktmr", max_length2)
    c2_ktmr_label3_pred = predict(aspect_text, token_c2_ktmr_label3, sam_c2_ktmr_label3, load_aspect_model(c2_ktmr_label3_weight_json, c2_ktmr_label3_weight_file),c2_ktmr_label3)
    if c2_ktmr_label3_pred == ['kn']:
      prediction_label3 =  c2_ktmr_label3_pred[0];
    elif c2_ktmr_label3_pred == ['cht']:
      prediction_label3 =  c2_ktmr_label3_pred[0];
    else:
       print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
  elif c2_ktmr_pred == ['none']:
    token_c2_qlkg, sam_c2_qlkg = load_data("MyData/c2_label2.xlsx","c2_qlkg",max_length2)
    c2_qlkg_pred = predict(aspect_text, token_c2_qlkg, sam_c2_qlkg, load_aspect_model(c2_qlkg_weight_json, c2_qlkg_weight_file),c2_qlkg)
    if c2_qlkg_pred == ['qlkg']:
      prediction_label2 =  c2_qlkg_pred[0];
      token_c2_qlkg_kn, sam_c2_qlkg_kn = load_data("MyData/c2_label2.xlsx","c2_qlkg_kn",max_length2)
      c2_qlkg_kn_pred = predict(aspect_text, token_c2_qlkg_kn, sam_c2_qlkg_kn, load_aspect_model(c2_qlkg_kn_weight_json, c2_qlkg_kn_weight_file),c2_qlkg_kn)
      if c2_qlkg_kn_pred == ['kn']:
        prediction_label3 =  c2_qlkg_kn_pred[0];
      elif c2_qlkg_kn_pred == ['none']:
        token_c2_qlkg_mt, sam_c2_qlkg_mt = load_data("MyData/c2_label3.xlsx","c2_qlkg_mt",max_length2)
        c2_qlkg_mt_pred = predict(aspect_text, token_c2_qlkg_mt, sam_c2_qlkg_mt, load_aspect_model(c2_qlkg_mt_weight_json, c2_qlkg_mt_weight_file),c2_qlkg_mt)
        if c2_qlkg_mt_pred == ['mt']:
          prediction_label3 =  c2_qlkg_mt_pred[0];
        elif c2_qlkg_mt_pred == ['none']:
          token_c2_qlkg_cht, sam_c2_qlkg_cht = load_data("MyData/c2_label3.xlsx","c2_qlkg_cht",max_length2)
          c2_qlkg_cht_pred = predict(aspect_text, token_c2_qlkg_cht, sam_c2_qlkg_cht, load_aspect_model(c2_qlkg_cht_weight_json, c2_qlkg_cht_weight_file),c2_qlkg_cht)
          if c2_qlkg_cht_pred == ['cht']:
            prediction_label3 =  c2_qlkg_cht_pred[0];
          elif c2_qlkg_cht_pred == ['none']:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
    elif c2_qlkg_pred == ['none']:
      token_c2_tq, sam_c2_tq = load_data("MyData/c2_label3.xlsx","c2_tq",max_length2)
      c2_tq_pred = predict(aspect_text, token_c2_tq, sam_c2_tq, load_aspect_model(c2_tq_weight_json, c2_tq_weight_file),c2_tq)
      if c2_tq_pred == ['tq']:
        prediction_label2 =  c2_tq_pred[0];
        token_c2_tq_kn, sam_c2_tq_kn = load_data("MyData/c2_label3.xlsx","c2_tq_kn",max_length2)
        c2_tq_kn_pred = predict(aspect_text, token_c2_tq_kn, sam_c2_tq_kn, load_aspect_model(c2_tq_kn_weight_json, c2_tq_kn_weight_file),c2_tq_kn)
        if c2_tq_kn_pred == ['kn']:
          prediction_label3 =  c2_tq_kn_pred[0];
        elif c2_tq_kn_pred == ['none']:
          token_c2_tq_mt, sam_c2_tq_mt = load_data("MyData/c2_label3.xlsx","c2_tq_mt",max_length2)
          c2_tq_mt_pred = predict(aspect_text, token_c2_tq_mt, sam_c2_tq_mt, load_aspect_model(c2_tq_mt_weight_json, c2_tq_mt_weight_file),c2_tq_mt)
          if c2_tq_mt_pred == ['mt']:
            prediction_label3 =  c2_tq_mt_pred[0];
          elif c2_tq_mt_pred == ['none']:
            token_c2_tq_cht, sam_c2_tq_cht = load_data("MyData/c2_label3.xlsx","c2_tq_cht",max_length2)
            c2_tq_cht_pred = predict(aspect_text, token_c2_tq_cht, sam_c2_tq_cht, load_aspect_model(c2_tq_cht_weight_json, c2_tq_cht_weight_file),c2_tq_cht)
            if c2_tq_cht_pred == ['cht']:
              prediction_label3 =  c2_tq_cht_pred[0];
            elif c2_tq_cht_pred == ['none']:
              print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        elif c2_tq_pred == ['none']:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
elif label1_pred == ['c3']:
  prediction_label1 = label1_pred[0]
  token_c3_dptt, sam_c3_dptt = load_data("MyData/c3.xlsx","c3_dptt",max_length2)
  c3_dptt_pred = predict(aspect_text, token_c3_dptt, sam_c3_dptt, load_aspect_model(c3_dptt_weight_json, c3_dptt_weight_file),c3_dptt)
  if c3_dptt_pred == ['dp_tt']:
    prediction_label2 =  c3_dptt_pred[0];
    token_c3_kntt_label3, sam_c3_kntt_label3 = load_data("MyData/c3_label3_fix.xlsx","c3_kntt",max_length2)
    c3_kntt_label3_pred = predict(aspect_text, token_c3_kntt_label3, sam_c3_kntt_label3, load_aspect_model(c3_kntt_label3_weight_json, c3_kntt_label3_weight_file),c3_kntt_label3)
    if c3_kntt_label3_pred == ['kn']:
      prediction_label3 =  c3_kntt_label3_pred[0];
    elif c3_kntt_label3_pred == ['gt']:
      prediction_label3 =  c3_kntt_label3_pred[0];
    else:
      print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
  elif c3_dptt_pred == ['none']:
      token_c3_kntt, sam_c3_kntt = load_data("MyData/c3.xlsx","c3_kntt",max_length2)
      c3_kntt_pred = predict(aspect_text, token_c3_kntt, sam_c3_kntt, load_aspect_model(c3_kntt_weight_json, c3_kntt_weight_file),c3_kntt)
      if c3_kntt_pred == ['kn_tt']:
        prediction_label2 =  c3_kntt_pred[0];
        token_c3_dptt_label3, sam_c3_dptt_label3 = load_data("MyData/c3_label3_fix.xlsx","c3_dptt",max_length2)
        c3_dptt_label3_pred = predict(aspect_text, token_c3_dptt_label3, sam_c3_dptt_label3, load_aspect_model(c3_dptt_label3_weight_json, c3_dptt_label3_weight_file),c3_dptt_label3)
        if c3_dptt_label3_pred == ['kn']:
          prediction_label3 =  c3_dptt_label3_pred[0];
        elif c3_dptt_label3_pred == ['gt']:
          prediction_label3 =  c3_dptt_label3_pred[0];
        else:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
      elif c3_kntt_pred == ['none']:
        token_c3_tttn, sam_c3_tttn = load_data("MyData/c3.xlsx","c3_tttn",max_length2)
        c3_tttn_pred = predict(aspect_text, token_c3_tttn, sam_c3_tttn, load_aspect_model(c3_tttn_weight_json, c3_tttn_weight_file),c3_tttn)
        if c3_tttn_pred == ['tt_tn']:
          prediction_label2 =  c3_tttn_pred[0];
          token_c3_tttn_label3, sam_c3_tttn_label3 = load_data("MyData/c3_label3_fix.xlsx","c3_tttn",max_length2)
          c3_tttn_label3_pred = predict(aspect_text, token_c3_tttn_label3, sam_c3_tttn_label3, load_aspect_model(c3_tttn_label3_weight_json, c3_tttn_label3_weight_file),c3_tttn_label3)
          if c3_tttn_label3_pred == ['kn']:
            prediction_label3 =  c3_tttn_label3_pred[0];
          elif c3_tttn_label3_pred == ['gt']:
            prediction_label3 =  c3_tttn_label3_pred[0];
          else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        elif c3_tttn_pred == ['none']:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        else:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
      else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
elif label1_pred == ['c4']:
  prediction_label1 = label1_pred[0]
  token_c4_bna, sam_c4_bna = load_data("MyData/c4_label2.xlsx","c4_bna",max_length2)
  c4_bna_pred = predict(aspect_text, token_c4_bna, sam_c4_bna, load_aspect_model(c4_bna_weight_json, c4_bna_weight_file),c4_bna)
  if c4_bna_pred == ['bna']:
    prediction_label2 =  c4_bna_pred[0];
    token_c4_bna_kn, sam_c4_bna_kn = load_data("MyData/c4_label3.xlsx","c4_bna_kn",max_length2)
    c4_bna_kn_pred = predict(aspect_text, token_c4_bna_kn, sam_c4_bna_kn, load_aspect_model(c4_bna_kn_weight_json, c4_bna_kn_weight_file),c4_bna_kn)
    if c4_bna_kn_pred == ['kn']:
      prediction_label3 =  c4_bna_kn_pred[0];
    elif c4_bna_kn_pred == ['none']:
      token_c4_bna_gt, sam_c4_bna_gt = load_data("MyData/c4_label3.xlsx","c4_bna_gt",max_length2)
      c4_bna_gt_pred = predict(aspect_text, token_c4_bna_gt, sam_c4_bna_gt, load_aspect_model(c4_bna_gt_weight_json, c4_bna_gt_weight_file),c4_bna_gt)
      if c4_bna_gt_pred == ['gt']:
        prediction_label3 =  c4_bna_gt_pred[0];
      elif c4_bna_gt_pred == ['none']:
        token_c4_bna_ch, sam_c4_bna_ch = load_data("MyData/c4_label3.xlsx","c4_bna_ch",max_length2)
        c4_bna_ch_pred = predict(aspect_text, token_c4_bna_ch, sam_c4_bna_ch, load_aspect_model(c4_bna_ch_weight_json, c4_bna_ch_weight_file),c4_bna_ch)
        if c4_bna_ch_pred == ['ch']:
          prediction_label3 =  c4_bna_ch_pred[0];
        elif c4_bna_ch_pred == ['none']:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        else:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
      else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
    else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
  elif c4_bna_pred == ['none']:
    token_c4_dc, sam_c4_dc = load_data("MyData/c4_label2.xlsx","c4_dc",max_length2)
    c4_dc_pred = predict(aspect_text, token_c4_dc, sam_c4_dc, load_aspect_model(c4_dcvdlq_weight_json, c4_dcvdlq_weight_file),c4_dc)
    if c4_dc_pred == ['dc']:
      prediction_label2 =  c4_dc_pred[0];
      token_c4_dc_kn, sam_c4_dc_kn = load_data("MyData/c4_label3.xlsx","c4_dc_kn",max_length2)
      c4_dc_kn_pred = predict(aspect_text, token_c4_dc_kn, sam_c4_dc_kn, load_aspect_model(c4_dc_kn_weight_json, c4_dc_kn_weight_file),c4_dc_kn)
      if c4_dc_kn_pred == ['kn']:
        prediction_label3 =  c4_dc_kn_pred[0];
      elif c4_dc_kn_pred == ['none']:
        token_c4_dc_gt, sam_c4_dc_gt = load_data("MyData/c4_label3.xlsx","c4_dc_gt",max_length2)
        c4_dc_gt_pred = predict(aspect_text, token_c4_dc_gt, sam_c4_dc_gt, load_aspect_model(c4_dc_gt_weight_json, c4_dc_gt_weight_file),c4_dc_gt)
        if c4_dc_gt_pred == ['gt']:
          prediction_label3 =  c4_dc_gt_pred[0];
        elif c4_dc_gt_pred == ['none']:
          token_c4_dc_ch, sam_c4_dc_ch = load_data("MyData/c4_label3.xlsx","c4_dc_ch",max_length2)
          c4_dc_ch_pred = predict(aspect_text, token_c4_dc_ch, sam_c4_dc_ch, load_aspect_model(c4_dc_ch_weight_json, c4_dc_ch_weight_file),c4_dc_ch)
          if c4_dc_ch_pred == ['ch']:
            prediction_label3 =  c4_dc_ch_pred[0];
          elif c4_dc_ch_pred == ['none']:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        else:
          print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
      else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
    elif c4_dc_pred == ['none']:
      token_c4_pcbn, sam_c4_pcbn = load_data("MyData/c4_label2.xlsx","c4_pcbn",max_length2)
      c4_pcbn_pred = predict(aspect_text, token_c4_pcbn, sam_c4_pcbn, load_aspect_model(c4_pcbn_weight_json, c4_pcbn_weight_file),c4_pcbn)
      if c4_pcbn_pred == ['pcbn']:
        prediction_label2 =  c4_pcbn_pred[0];
        token_c4_pcbn_kn, sam_c4_pcbn_kn = load_data("MyData/c4_label3.xlsx","c4_pcbn_kn",max_length2)
        c4_pcbn_kn_pred = predict(aspect_text, token_c4_pcbn_kn, sam_c4_pcbn_kn, load_aspect_model(c4_pcbn_kn_weight_json, c4_pcbn_kn_weight_file),c4_pcbn_kn)
        if c4_pcbn_kn_pred == ['kn']:
          prediction_label3 =  c4_pcbn_kn_pred[0];
        elif c4_pcbn_kn_pred == ['none']:
          token_c4_pcbn_gt, sam_c4_pcbn_gt = load_data("MyData/c4_label3.xlsx","c4_pcbn_gt",max_length2)
          c4_pcbn_gt_pred = predict(aspect_text, token_c4_pcbn_gt, sam_c4_pcbn_gt, load_aspect_model(c4_pcbn_gt_weight_json, c4_pcbn_gt_weight_file),c4_pcbn_gt)
          if c4_pcbn_gt_pred == ['gt']:
            prediction_label3 =  c4_pcbn_gt_pred[0];
          elif c4_pcbn_gt_pred == ['none']:
            token_c4_pcbn_ch, sam_c4_pcbn_ch = load_data("MyData/c4_label3.xlsx","c4_pcbn_ch",max_length2)
            c4_pcbn_ch_pred = predict(aspect_text, token_c4_pcbn_ch, sam_c4_pcbn_ch, load_aspect_model(c4_pcbn_ch_weight_json, c4_pcbn_ch_weight_file),c4_pcbn_ch)
            if c4_pcbn_ch_pred == ['ch']:
              prediction_label3 =  c4_pcbn_ch_pred[0];
            elif c4_pcbn_ch_pred == ['none']:
              print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
            else:
              print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
      elif c4_pcbn_pred == ['none']:
        token_c4_ptbn, sam_c4_ptbn = load_data("MyData/c4_label2.xlsx","c4_ptbn",max_length2)
        c4_ptbn_pred = predict(aspect_text, token_c4_ptbn, sam_c4_ptbn, load_aspect_model(c4_ptbn_weight_json, c4_ptbn_weight_file),c4_ptbn)
        if c4_ptbn_pred == ['ptbn']:
          prediction_label2 =  c4_ptbn_pred[0];
          token_c4_ptbn_label3, sam_c4_ptbn_label3 = load_data("MyData/c4_label3.xlsx","c4_ptbn",max_length2)
          c4_ptbn_label3_pred = predict(aspect_text, token_c4_ptbn_label3, sam_c4_ptbn_label3, load_aspect_model(c4_ptbn_label3_weight_json, c4_ptbn_label3_weight_file),c4_ptbn_label3)
          if c4_ptbn_label3_pred == ['gt']:
            prediction_label3 =  c4_ptbn_label3_pred[0];
          elif c4_ptbn_label3_pred == ['ch']:
            prediction_label3 =  c4_ptbn_label3_pred[0];
          else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
        elif c4_ptbn_pred == ['none']:
          token_c4_pdbn, sam_c4_pdbn = load_data("MyData/c4_label2.xlsx","c4_pdbn",max_length2)
          c4_pdbn_pred = predict(aspect_text, token_c4_pdbn, sam_c4_pdbn, load_aspect_model(c4_pdbn_weight_json, c4_pdbn_weight_file),c4_pdbn)
          if c4_pdbn_pred == ['pdbn']:
            prediction_label2 =  c4_pdbn_pred[0];
            token_c4_pdbn_label3, sam_c4_pdbn_label3 = load_data("MyData/c4_label3.xlsx","c4_pdbn",max_length2)
            c4_pdbn_label3_pred = predict(aspect_text, token_c4_pdbn_label3, sam_c4_pdbn_label3, load_aspect_model(c4_pdbn_label3_weight_json, c4_pdbn_label3_weight_file),c4_pdbn_label3)
            if c4_pdbn_label3_pred == ['gt']:
              prediction_label3 =  c4_pdbn_label3_pred[0];
            elif c4_pdbn_label3_pred == ['ch']:
              prediction_label3 =  c4_pdbn_label3_pred[0];
            else:
              print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
          elif c4_pdbn_pred == ['none']:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
          else:
            print('Không thể tìm thấy câu trả lời cho câu hỏi này.')
else:
  print('Không thể tìm thấy câu trả lời cho câu hỏi này.')

print('Ba label dự đoán để trả lời là: ',prediction_label1, prediction_label2, prediction_label3)
