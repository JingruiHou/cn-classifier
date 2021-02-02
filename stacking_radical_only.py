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
from keras import layers
from keras import models
import pickle
import sys
sys.path.append('./')
sys.path.append('./cnradical')
import radical_meta_classifiers
from radical import Radical, RunOption


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


def build_radical_features(texts):
    radical = Radical(RunOption.Radical)
    radicals_feature = []
    distinct_radicals = set()
    for txt in texts:
        radical_txt = []
        for w in txt:
            ra = radical.trans_ch(w)
            if ra is not None:
                radical_txt.append(ra)
        radicals_feature.append(' '.join(radical_txt))
        distinct_radicals.update(radical_txt)
    return len(distinct_radicals), radicals_feature


def build_models(model_name, feature_count):
    if model_name == 'DNN':
        return radical_meta_classifiers.build_racial_DNN(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if model_name == 'CNN':
        return radical_meta_classifiers.build_radical_cnn(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if model_name == 'GRU':
        return radical_meta_classifiers.build_radical_GRU(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if model_name == 'self-attention':
        return radical_meta_classifiers.build_transformer(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if model_name == 'RCNN':
        return radical_meta_classifiers.build_text_rcnn(max_words=feature_count, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM)


SEED = 9
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
segmented = 0

dfs = [
    pd.read_csv(open('../../radical_sent/data/corpus.csv', 'r', encoding='UTF-8')),
       pd.read_csv(open('../../stacked_model/data/corpus.csv', 'r', encoding='UTF-8'))]

df_names = ['sent', 'domain']

for df_idx, df in enumerate(dfs):
    df_name = df_names[df_idx]
    texts = df['text']
    labels = df['label']
    texts = build_cn_corpus(texts, segmented)
    max_radicals, radical_features = build_radical_features(texts)

    radical_tokenizer = Tokenizer(nb_words=max_radicals)
    radical_tokenizer.fit_on_texts(radical_features)
    radical_seq = radical_tokenizer.texts_to_sequences(radical_features)
    X_radicals = pad_sequences(radical_seq, maxlen=MAX_SEQUENCE_LENGTH)

    unique_label = list(labels.unique())
    output_dim = len(unique_label)
    Y = [unique_label.index(label) for label in labels]

    X_radicals_, X_radicals_predict, Y_, Y_predict = train_test_split(X_radicals, Y, test_size=0.2, random_state=SEED, stratify=Y)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_names = ['CNN', 'DNN', 'GRU']
    epochs = [20, 20, 10]
    for _idx, model_name in enumerate(model_names):
        for optimizer in ['Adagrad', 'Adamax', 'Adadelta', 'SGD', 'RMSprop', 'Nadam']:
            slices = 0
            for train, test in kfold.split(X_radicals_, Y_):
                x_radical_train = X_radicals_[train]
                y_train = to_categorical(np.asarray(Y_))[train]
                x_radical_validate = X_radicals_[test]
                y_validate = to_categorical(np.asarray(Y_))[test]
                print(model_name)
                radical_input, radical_out = build_models(model_name, max_radicals)

                x = layers.Dense(output_dim, activation='softmax')(radical_out)
                model = models.Model(inputs=radical_input, outputs=x)

                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['acc'])
                model.summary()
                history = model.fit(x_radical_train, y_train, validation_data=(x_radical_validate, y_validate), nb_epoch=epochs[_idx], batch_size=100, verbose=True)
                with open('./{}/history/{}_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(history.history, f)
                train_predict = model.predict(x_radical_validate, batch_size=100)
                with open('./{}/model/{}_train_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(train_predict, f)
                test_predict = model.predict(X_radicals_predict, batch_size=100)
                with open('./{}/model/{}_test_{}_{}_{}.plk'.format(model_name, df_name, model_name, slices, optimizer), 'wb') as f:
                    pickle.dump(test_predict, f)
                slices += 1
                del model
