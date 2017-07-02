import cPickle
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.constraints import unitnorm
from keras.regularizers import l2

from sklearn.metrics import roc_auc_score


def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    x = []
    pad = kernel_size - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l, kernel_size)
        sent.append(rev['y'])
        if rev['split'] == 1:
            train.append(sent)
        elif rev['split'] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=np.int)
    val = np.array(val, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return [train, val, test]


print "loading data..."
x = cPickle.load(open("/Users/bigsnarfdude/Desktop/bagOfPopcorn/imdb-train-val-test.pickle", "rb"))
revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
print "data loaded!"


datasets = make_idx_data(revs, word_idx_map, max_l=2633, kernel_size=5)

N = datasets[0].shape[0]
conv_input_width = W.shape[1]
conv_input_height = int(datasets[0].shape[1]-1)

train_X = np.zeros((N, conv_input_height), dtype=np.int)
train_Y = np.zeros((N, 2), dtype=np.int)
for i in xrange(N):
    for j in xrange(conv_input_height):
        train_X[i, j] = datasets[0][i, j]
    train_Y[i, datasets[0][i, -1]] = 1
    
print 'train_X.shape = {}'.format(train_X.shape)
print 'train_Y.shape = {}'.format(train_Y.shape)

Nv = datasets[1].shape[0]

val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
val_Y = np.zeros((Nv, 2), dtype=np.int)
for i in xrange(Nv):
    for j in xrange(conv_input_height):
        val_X[i, j] = datasets[1][i, j]
    val_Y[i, datasets[1][i, -1]] = 1
    
print 'val_X.shape = {}'.format(val_X.shape)
print 'val_Y.shape = {}'.format(val_Y.shape)

N_fm = 300
kernel_size = 8
model = Sequential()
model.add(Embedding(input_dim=W.shape[0], 
                    output_dim=W.shape[1], 
                    input_length=conv_input_height,
                    weights=[W], 
                    W_constraint=unitnorm()))

model.add(Reshape((1, conv_input_height, conv_input_width)))
model.add(Convolution2D(N_fm, 
                        kernel_size, 
                        conv_input_width, 
                        border_mode='valid', 
                        W_regularizer=l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])


epoch = 0
val_acc = []
val_auc = []
N_epoch = 3

for i in xrange(N_epoch):
    model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1)
    output = model.predict_proba(val_X, batch_size=10, verbose=1)
    # find validation accuracy using the best threshold value t
    vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
    # find validation AUC
    vauc = roc_auc_score(val_Y, output)
    val_acc.append(vacc)
    val_auc.append(vauc)
    print 'Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc)
    epoch += 1
    
print '{} epochs passed'.format(epoch)
print 'Accuracy on validation dataset:'
print val_acc
print 'AUC on validation dataset:'
print val_auc

model.save_weights('/Users/bigsnarfdude/Desktop/bagOfPopcorn/cnn_3epochs.model')
Nt = datasets[2].shape[0]
test_X = np.zeros((Nt, conv_input_height), dtype=np.int)
for i in xrange(Nt):
    for j in xrange(conv_input_height):
        test_X[i, j] = datasets[2][i, j]
    
print 'test_X.shape = {}'.format(test_X.shape)
p = model.predict_proba(test_X, batch_size=10)

import pandas as pd
data = pd.read_csv('/Users/bigsnarfdude/Desktop/bagOfPopcorn/testData.tsv', sep='\t')
df = pd.DataFrame({'id': data['id'], 'sentiment': p[:,0]})
df.to_csv('/Users/bigsnarfdude/Desktop/bagOfPopcorn/cnn_3epochs.csv', index=False)

"""
Using Theano backend.
loading data...
data loaded!
train_X.shape = (20032, 2641)
train_Y.shape = (20032, 2)
val_X.shape = (4968, 2641)
val_Y.shape = (4968, 2)
Epoch 1/1
20032/20032 [==============================] - 3865s - loss: 0.5965 - acc: 0.7112
4968/4968 [==============================] - 267s
Epoch 0: validation accuracy = 81.804%, validation AUC = 89.865%
Epoch 1/1
20032/20032 [==============================] - 4147s - loss: 0.4124 - acc: 0.8393
4968/4968 [==============================] - 313s
Epoch 1: validation accuracy = 87.440%, validation AUC = 94.344%
Epoch 1/1
20032/20032 [==============================] - 3982s - loss: 0.3293 - acc: 0.8832
4968/4968 [==============================] - 261s
Epoch 2: validation accuracy = 88.748%, validation AUC = 95.366%
3 epochs passed
Accuracy on validation dataset:
[0.81803542673107887, 0.87439613526570048, 0.8874798711755234]
AUC on validation dataset:
[0.8986494579306501, 0.94344191792273768, 0.95366227242382329]
test_X.shape = (25000, 2641)
25000/25000 [==============================] - 1269s
"""
