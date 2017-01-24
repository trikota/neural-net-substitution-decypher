import sys, time, re
import numpy as np
from random import randint as ri
from string import ascii_lowercase
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dense, recurrent
from keras.layers.embeddings import Embedding
from keras import backend as K

INPUT_LENGTH = 80
DATA_FILE = "wiki.txt"

RNN = recurrent.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 64

CHAR_SET = list(ascii_lowercase) + [' ', '.']

def to_one_hot(x, max = len(CHAR_SET)):
    assert x < max
    r = ([0]*max)
    r[x] = 1
    return r

def get_frequecy_vector(s, dictify = False):
    i_vec = []
    for c in CHAR_SET:
        i_vec.append(s.count(c))
    return np.asarray(i_vec)

#converts letter to it's place in list of letters reverse-sorted by number of occurrences
def string_to_freq_seq(s, one_hot = True, zero_padded = True):
    freq_seq = []
    if zero_padded: #add zeros from left to match RNN input shape
                    #though in our case len(s) == INPUT_LENGTH always
        for _ in range(INPUT_LENGTH - len(s)):
            freq_seq.append([0]*len(CHAR_SET) if one_hot else -1)
    freq_dict = {i:num for i,num in enumerate(get_frequecy_vector(s))}
    top_keys = sorted(freq_dict, key=freq_dict.get, reverse=True)
    for c in s:
        idx = top_keys.index(CHAR_SET.index(c))
        if one_hot:
            oh_idx = [0]*len(CHAR_SET)
            oh_idx[idx] = 1
            idx = oh_idx
        freq_seq.append(idx)
    return np.asarray(freq_seq), top_keys

#get input and label. x is shifted freq_seq by mod len(CHAR_SET), such that target char is always represented by 0 
def string_to_rel_seq(s):
    res_x = []
    res_y = []
    freq_seq, top_keys = string_to_freq_seq(s, one_hot = False, zero_padded = False)
    for place in set(freq_seq):
        y = to_one_hot(top_keys[place])
        x = np.remainder(freq_seq-place, len(CHAR_SET))
        res_x.append(x)
        res_y.append(y)
    return res_x, res_y   

#(x,y) to string
def rel_seq_to_str(xs,ys):
    res = [None]*len(xs[0])
    for i_x,x in enumerate(xs):
        for i_pos,x_pos in enumerate(x):
            if x_pos == 0:
                res[i_pos] = CHAR_SET[list(ys[i_x]).index(1)]
    return ''.join(res) 

def load_data(filename=DATA_FILE):
    X,Y = [],[]
    with open(DATA_FILE) as f:
        text = f.read().lower()
        for c in set(text):
            if c not in CHAR_SET:
                text = text.replace(c, '')
        text = re.sub('\s+', ' ', text) #replace multiple spaces with one
    for i in range(1,len(text) - INPUT_LENGTH):
        if text[i-1] not in [' ', '.']: #take sequences that don't break word at start
            continue
        str_seq = text[i:i+INPUT_LENGTH]
        xs, ys = string_to_rel_seq(str_seq)
        X.extend(xs)
        Y.extend(ys)
    return np.asarray(X),np.asarray(Y)
    
def build_model():
    model = Sequential()
    # Encoder
    model.add(Embedding(input_dim = len(CHAR_SET), output_dim = len(CHAR_SET), input_length=INPUT_LENGTH))
    model.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_LENGTH, len(CHAR_SET)), return_sequences=True))
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    #Last encoder layer shouldn't return sequence.
    model.add(RNN(HIDDEN_SIZE, return_sequences=False))
    # At this moment output is 1d
    #Final dense layer
    model.add(Dense(len(CHAR_SET)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train():
    print "Loading data"
    X, Y = load_data()   
    print "X, Y shapes:", X.shape, Y.shape
    X_train, X_test, Y_train, Y_test = train_test_split( #shuffle and split data
                    X, Y, test_size=0.2, random_state=42) 
    print "Train on", len(X_train), "samples"

    model = build_model()
    model.summary()

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=5,
                validation_data=(X_test, Y_test))
                
    model.save('model_{}.h5'.format(int(time.time())))
    
PRETRAINED_PATH = 'pretrained_wiki.h5'
model = build_model()
model.load_weights(PRETRAINED_PATH)

TEST_DATA_PATH = 'cryptology_articles.txt' #completely different data
with open(DATA_FILE) as f:
    text = f.read().lower()
    for c in set(text):
        if c not in CHAR_SET:
            text = text.replace(c, '')
    text = re.sub('\s+', ' ', text)
i=0
while i<len(text):
    i+=1
    if text[i-1] not in [' ', '.']: #take sequences that don't break word at start
        continue    
    str_seq = text[i:i+INPUT_LENGTH]
    xs, ys = string_to_rel_seq(str_seq)

    ys_pred = model.predict(np.asarray(xs))
    argmaxes = []
    for amax in ys_pred.argmax(axis=1):
        argmaxes.append(to_one_hot(amax))
    argmaxes = np.asarray(argmaxes)
    print '-'*10
    print rel_seq_to_str(xs,ys)   
    print rel_seq_to_str(xs,argmaxes)   
    raw_input()



