#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import jieba
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


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


def add_ngram(sequences, token_indice, ngram_range, maxlen):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:maxlen]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def create_ngram_dic(X, ngram_range):
    gram_dic = {}
    for ngram_value in range(2, ngram_range + 1):
        for input_list in X:
            for index in range(len(input_list) - ngram_value + 1):
                gram_key = tuple(input_list[index: index + ngram_value])
                if gram_key in gram_dic:
                    gram_dic[gram_key] += 1
                else:
                    gram_dic[gram_key] = 1
        print('gram_dic', len(gram_dic))
    return gram_dic


def build_fast_text_input(X, ngram_range, max_features, maxlen):
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        start_index = max_features + 1
        gram_dic = create_ngram_dic(X, ngram_range)
        # 根据值排序，并返回key 的list
        gram_list = sorted(gram_dic, key=gram_dic.__getitem__, reverse=True)
        # 去除低频元素
        ngram_set = gram_list[:max_features]

        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}

        indice_token = {token_indice[k]: k for k in token_indice}
        max_features = np.max(list(indice_token.keys())) + 1
        X = add_ngram(X, token_indice, ngram_range, maxlen)
    X = pad_sequences(X, maxlen=maxlen * ngram_range)
    return X, max_features


def get_input_token(inputs, max_word):
    txt_tokenizer = Tokenizer(nb_words=max_word)
    txt_tokenizer.fit_on_texts(inputs)
    txt_seq = txt_tokenizer.texts_to_sequences(inputs)
    return txt_seq


def build_model(max_words, embedding_dim, output_dim, sequence_length):
    print('Build model...')
    model = models.Sequential()
    model.add(layers.Embedding(max_words,
                               embedding_dim,
                               input_length=sequence_length))

    # 我们增加 GlobalAveragePooling1D, 这将平均计算文档中所有词汇的的词嵌入
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    model.add(layers.Activation('softmax'))
    return model


EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 500
SEED = 9
segmented = 0
max_features = {0: 10000, 1: 50000}

dfs = [pd.read_csv(open('../../radical_sent/data/corpus.csv', 'r', encoding='UTF-8')),
       pd.read_csv(open('../../stacked_model/data/corpus.csv', 'r', encoding='UTF-8'))]

df_names = ['sent', 'domain']

for df_idx, df in enumerate(dfs):
    df_name = df_names[df_idx]
    texts = df['text']
    labels = df['label']
    unique_label = list(labels.unique())
    output_dim = len(unique_label)
    Y = [unique_label.index(label) for label in labels]
    texts = build_cn_corpus(texts, segmented)

    tokenizer = Tokenizer(nb_words=max_features[segmented])
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    for gram in [3]:
        for optimizer in ['Nadam']:
            model_name = 'FT_{}'.format(gram)
            X, MAX_WORDS = build_fast_text_input(sequences, gram, max_features[segmented], MAX_SEQUENCE_LENGTH)
            print(X.shape)
            print(len(Y))
            X_, X_predict, Y_, Y_predict = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            slices = 0
            for train, test in kfold.split(X_, Y_):
                x_train = X_[train]
                y_train = to_categorical(np.asarray(Y_))[train]
                x_validate = X_[test]
                y_validate = to_categorical(np.asarray(Y_))[test]
                print(model_name)
                model = build_model(max_words=MAX_WORDS, embedding_dim=EMBEDDING_DIM, output_dim=output_dim, sequence_length=MAX_SEQUENCE_LENGTH * gram)
                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['acc'])
                model.summary()
                history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), nb_epoch=30, batch_size=100, verbose=True)
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
