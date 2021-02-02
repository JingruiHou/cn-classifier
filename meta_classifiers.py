#! /usr/bin/env python
from __future__ import print_function
from keras import layers
from keras import models
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers


class Word_Attention(Layer):

    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(Word_Attention, self).__init__()

    def build(self, input_shape):
        print(len(input_shape))
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(Word_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


def build_DNN(max_words=20000, maxlen=200, embedding_dim=400, classification_type=2):
    S_inputs = layers.Input(shape=(None,), dtype='int32')
    embeddings = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(S_inputs)
    dense_1 = layers.Dense(100, activation='tanh')(embeddings)
    max_pooling = layers.GlobalMaxPooling1D()(dense_1)
    outputs = layers.Dense(classification_type, activation='softmax')(max_pooling)
    model = models.Model(inputs=S_inputs, outputs=outputs)
    return model


def build_transformer(max_words=20000, maxlen=200, embedding_dim=400, classification_type=2):
    S_inputs = layers.Input(shape=(None,), dtype='int32')
    embeddings = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(S_inputs)
    embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = layers.GlobalAveragePooling1D()(O_seq)
    O_seq = layers.Dropout(0.5)(O_seq)
    outputs = layers.Dense(classification_type, activation='softmax')(O_seq)
    model = models.Model(inputs=S_inputs, outputs=outputs)
    return model


def build_text_cnn(max_words=20000, maxlen=200, embedding_dim=400, classification_type=2):
    model = models.Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))  # 使用Embeeding层将每个词编码转换为词向量
    model.add(layers.Conv1D(256, 5, padding='same'))
    model.add(layers.MaxPooling1D(3, 3, padding='same'))
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.MaxPooling1D(3, 3, padding='same'))
    model.add(layers.Conv1D(64, 3, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())  # (批)规范化层
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(classification_type, activation='softmax'))
    return model


def build_text_rcnn(max_words=20000, maxlen=200, embedding_dim=400, classification_type=2):
    sentence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = layers.Embedding(max_words, embedding_dim)(sentence_input)
    x_backwords = layers.GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.32 * 0.1), recurrent_regularizer=regularizers.l2(0.32), go_backwards=True)(embedded_sequences)
    x_backwords_reverse = layers.Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
    x_fordwords = layers.GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.32 * 0.1), recurrent_regularizer=regularizers.l2(0.32), go_backwards=False)(embedded_sequences)
    x_feb = layers.Concatenate(axis=2)([x_fordwords, embedded_sequences, x_backwords_reverse])
    x_feb = layers.Dropout(0.32)(x_feb)
    # Concatenate后的embedding_size
    dim_2 = K.int_shape(x_feb)[2]
    x_feb_reshape = layers.Reshape((maxlen, dim_2, 1))(x_feb)
    filters = [2, 3, 4, 5]
    conv_pools = []
    for filter in filters:
        conv =layers. Conv2D(filters=300,
                      kernel_size=(filter, dim_2),
                      padding='valid',
                      kernel_initializer='normal',
                      activation='relu',
                      )(x_feb_reshape)
        pooled = layers.MaxPooling2D(pool_size=(maxlen - filter + 1, 1),
                              strides=(1, 1),
                              padding='valid',
                              )(conv)
        conv_pools.append(pooled)

    x = layers.Concatenate()(conv_pools)
    x = layers.Flatten()(x)
    output = layers.Dense(classification_type, activation='softmax')(x)
    model = models.Model(sentence_input, output)
    return model


def build_gru(maxlen=200, max_words=20000, embedding_dim=400, classification_type=2):
    embedding_layer = layers.Embedding(max_words, embedding_dim,input_length=maxlen)
    sequence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = layers.Bidirectional(layers.GRU(100))(embedded_sequences)
    dense_2 = layers.Dense(classification_type, activation='softmax')(l_gru)
    model = models.Model(sequence_input, dense_2)
    return model


def build_gru_attention(maxlen=200, max_words=20000, embedding_dim=400, classification_type=2):
    embedding_layer = layers.Embedding(max_words, embedding_dim,input_length=maxlen)
    sequence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = layers.Bidirectional(layers.GRU(100, return_sequences=True))(embedded_sequences)
    dense_1 = Word_Attention(100)(l_gru)
    dense_2 = layers.Dense(classification_type, activation='softmax')(dense_1)
    model = models.Model(sequence_input, dense_2)
    return model

def build_cnn_lstm(maxlen=200, max_words=20000, embedding_dim=400, classification_type=2):
    model = models.Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(64, 5,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(classification_type, activation='softmax'))
    return model



def build_han(maxlen, max_sent_len, max_words, embedding_dim, classification_type):
    embedding_layer = layers.Embedding(max_words, embedding_dim, input_length=maxlen)
    sentence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = layers.Bidirectional(layers.GRU(100, return_sequences=True))(embedded_sequences)
    l_att = Word_Attention(100)(l_lstm)
    sentEncoder = models.Model(sentence_input, l_att)
    sentEncoder.summary()
    review_input = layers.Input(shape=(max_sent_len, maxlen), dtype='int32')
    review_encoder = layers.TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = layers.Bidirectional(layers.GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = Word_Attention(100)(l_lstm_sent)
    preds = layers.Dense(classification_type, activation='softmax')(l_att_sent)
    model = models.Model(review_input, preds)
    return model
