#coding: utf8
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random, sys


path = './poetry.txt'
print 'opening txt'
text = open(path).read().lower().decode('utf-8')
print 'corpus length:', len(text)

chars = set(text)
print 'total chars:', len(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print 'nb sequences:', len(sentences)

#########向量化###########

print 'Vectorization...'
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print 'Build model...'
model = Sequential()
# model.add(LSTM(len(chars), 512, return_sequences=True))
#######network#########
#两层的LSTM，注意在首层中使用input_shape
#中间的各层都不需要再次计算输入层数，网络自行计算
#LSTM后不需要像CNN那样接一个激活函数，在LSTM内部存在
#######################


model.add(LSTM(512,return_sequences = True, input_shape=(maxlen,len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
print 'Modelling finishing'
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

# train the model, output generated text after each iteration
for iteration in range(1, 100):
    print '-' * 50
    print 'Iteration', iteration
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5,0.8, 1.0,1.1, 1.2, 1.5]:
        print '----- diversity:', diversity

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print '----- Generating with seed: "' + sentence + '"'

        for iteration in range(120):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        print generated
