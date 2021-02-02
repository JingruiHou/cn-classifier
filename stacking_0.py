#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle

import sys
sys.path.append('./')
import meta_classifiers


def build_cn_corpus(texts, segmented):
    texts = [re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]', '', txt) for txt in texts]
    new_txts = []
    if segmented:
        for txt in texts:
            seg_list = jieba.cut(txt)
            line = ' '.join(seg_list)
            new_txts.append(line)
    else:
        new_txts = [' '.join(list(txt)) for txt in texts]
    return new_txts


def build_models(model_name, feature_count):
    if model_name == 'DNN':
        return meta_classifiers.build_DNN(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, classification_type=output_dim)
    if model_name == 'CNN':
        return meta_classifiers.build_text_cnn(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, classification_type=output_dim)
    if model_name == 'GRU':
        return meta_classifiers.build_gru(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, classification_type=output_dim)
    if model_name == 'self-attention':
        return meta_classifiers.build_transformer(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, classification_type=output_dim)
    if model_name == 'RCNN':
        return meta_classifiers.build_text_rcnn(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, classification_type=output_dim)


SEED = 9
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
segmented = 0
max_features = {0: 10000, 1: 50000}

dfs = [pd.read_csv(open('../../radical_sent/data/corpus.csv', 'r', encoding='UTF-8')),
       pd.read_csv(open('../../stacked_model/data/corpus.csv', 'r', encoding='UTF-8'))]

df_names = ['sent', 'domain']

for df_idx, df in enumerate(dfs):
    df_name = df_names[df_idx]
    texts = df['text']
    labels = df['label']
    texts = build_cn_corpus(texts, segmented)
    txt_tokenizer = Tokenizer(nb_words=max_features[segmented])
    txt_tokenizer.fit_on_texts(texts)
    txt_seq = txt_tokenizer.texts_to_sequences(texts)
    X = pad_sequences(txt_seq, maxlen=MAX_SEQUENCE_LENGTH)
    unique_label = list(labels.unique())
    output_dim = len(unique_label)
    Y = [unique_label.index(label) for label in labels]
    X_, X_predict, Y_, Y_predict = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for optimizer in ['Adadelta', 'SGD', 'RMSprop']:
        model_names = ['CNN']
        for model_name in model_names:
            slices = 0
            for train, test in kfold.split(X_, Y_):
                x_train = X_[train]
                y_train = to_categorical(np.asarray(Y_))[train]
                x_validate = X_[test]
                y_validate = to_categorical(np.asarray(Y_))[test]
                print(model_name)
                model = build_models(model_name, max_features[segmented])

                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['acc'])
                model.summary()
                history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), nb_epoch=20, batch_size=100, verbose=True)
                with open('./{}/history/{}_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(history.history, f)
                train_predict = model.predict(x_validate, batch_size=100)
                with open('./{}/model/{}_train_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(train_predict, f)
                test_predict = model.predict(X_predict, batch_size=100)
                with open('./{}/model/{}_test_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(test_predict, f)
                slices += 1
                del model