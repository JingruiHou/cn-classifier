import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split


train_size = 40000
test_size = 10000
output_dim = 10

models = ['DNN', 'CNN', 'GRU', 'FT_3']
optimizers = ['RMSprop', 'Adagrad', 'Nadam', 'Nadam']

SEED = 9

df = pd.read_csv(open('../stacked_model/data/corpus.csv', 'r', encoding='UTF-8'))
labels = df['label']

unique_label = list(labels.unique())
output_dim = len(unique_label)
Y = [unique_label.index(label) for label in labels]
X = np.zeros((len(Y)))
X_, X_predict, Y_, Y_predict = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

train_labels = []
for slices in range(0, 5):
    with open('./k-folds-domain/test_{}.plk'.format(slices), 'rb') as f:
        test = pickle.load(f)
        [train_labels.append(Y_[idx]) for idx in list(test)]


train_stackX = None
for idx, model_name in enumerate(models):
    data = np.zeros((train_size, output_dim), dtype=np.float)
    for s in range(0, 5):
        with open('./semantic-only/{}/model/domain_train_{}_{}_{}.plk'.
                          format(model_name, model_name, s, optimizers[idx]), 'rb') as f:
            arr = pickle.load(f)
            print(arr.shape)
            start_idx = s*arr.shape[0]
            end_idx = start_idx + arr.shape[0]
            data[start_idx: end_idx] = arr
    if train_stackX is None:
        train_stackX = data
    else:
        train_stackX = np.dstack((train_stackX, data))

print(train_stackX.shape)
# flatten predictions to [rows, members x probabilities]
train_stackX = train_stackX.reshape((train_stackX.shape[0], train_stackX.shape[1] * train_stackX.shape[2]))
print(train_stackX.shape)

test_stackX = None
for idx, model_name in enumerate(models):
    avg_arr = 0
    for s in range(0, 5):
        with open('./semantic-only/{}/model/domain_test_{}_{}_{}.plk'.
                          format(model_name, model_name, s, optimizers[idx]), 'rb') as f:
            arr = pickle.load(f)
            avg_arr += arr/5
    if test_stackX is None:
        test_stackX = avg_arr
    else:
        test_stackX = np.dstack((test_stackX, avg_arr))

print(test_stackX.shape)
test_stackX = test_stackX.reshape((test_stackX.shape[0], test_stackX.shape[1] * test_stackX.shape[2]))
print(test_stackX.shape)


def xgboost(train_data, train_label, test_data, test_label):
    import xgboost as xgb
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dtest = xgb.DMatrix(test_data, label=test_label)
    param = {}
    param['eta'] = 0.01
    param['max_depth'] = 6
    param["booster"] = "gbtree"
    param['nthread'] = 4
    param["silent"] = 1
    param["min_child_weight"] = 3
    param['num_class'] = 10
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    num_round = 100
    # train1
    param['objective'] = 'multi:softmax'  # shape: (5080,), 直接输出标签[0,1,3,..]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    # get prediction
    pred = bst.predict(dtest)  #
    del bst
    del xgb
    return pred

xgboost_preds = xgboost(train_stackX, train_labels, test_stackX, Y_predict)

print('ACCURACY OF XGBOOST IS %.3f' % accuracy_score(Y_predict, xgboost_preds))

with open('stacked_1/xgboost_result_sent.pkl', 'wb') as f:
    pickle.dump((Y_predict, xgboost_preds), f)





