import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
from keras.optimizers import adam
#checks pickles folder for existing data
# rb is read bytes wb is write bytes
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

#function to build the data, preprocessed, to push through the model as a matrix
#min and max are taken as their worst possible values so that they can be updated
#X being the mfcc array and y being the array of the index of labels
def build_rand_feat():
    #checks if data is already available
    tmp = check_data()
    if tmp :
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), float('-inf')
    for _ in tqdm(range(n_samples)):
        #randomly generates a filename 
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        label = df.at[file,'label']
        if wav.shape[0]-config.step<=0:
            continue
        else:
            rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample,rate,numcep=config.nfeat,
                        nfilt=config.nfilt,nfft=config.nfft)
        _min = min(np.amin(X_sample),_min)
        _max = max(np.amax(X_sample),_max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X,y = np.array(X), np.array(y)
    #normalizes the values between 0 and 1
    X = (X - _min)/(_max - _min)
    #Don't know why, for CNN, X is a 4D matrix and 3D for RNN 
    if config.mode == 'conv':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2])
    #hotencoding the index of labels (y) to a matrix, meaning : 
    #0,0 element of the matrix = Acoustic guitar, (1,0) element = Bass_drum etc.
    #categorical cross entropy ? 
    y = to_categorical(y, num_classes=10)
    config.data = (X, y)
    
    with open(config.p_path, 'wb') as handle :
        pickle.dump(config, handle, protocol=2)
    return X,y


#convolutional model function    
#uses sequential api from keras
#stacks layers sequentially 
#pooling layer follows conv layer
#need to learn more about how convolution works to learn activation, strides, padding
#Conv2D(number of neurons, activation function, strides, padding, *(only for first layer)input_shape)
#activation is softmax for last layer
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same', input_shape=input_shape))
    model.add(Conv2D(32,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same'))
    model.add(Conv2D(64,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same'))
    model.add(Conv2D(128,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same'))
    model.add(Conv2D(256,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.summary()
    opt = adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer =opt, metrics =['acc'])

    return model

#recurrent model function
#neurons have LSTM units
#model features that change over time
#shape of data for RNN is (number of neurons, time, features)
#takes longer to train the more layers it has
#takes longer to train and has less accuracy since optimizing LSTM's is difficult due to back propagation
#CNN's are better for classification
def get_recurrent_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(256, activation='relu',)))
    model.add(TimeDistributed(Dense(128, activation='relu',)))
    model.add(TimeDistributed(Dense(64, activation='relu',)))
    model.add(TimeDistributed(Dense(32, activation='relu',)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    opt = adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer =opt, metrics =['acc'])

    return model

df = pd.read_csv('UrbanSound8K.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

print (classes)
print(class_dist)

#2 * total length of all data divided into 1/10 of a 1s segments
n_samples = 1* int(df['length'].sum()/0.1)
#probability distribution so that total of the class distribution is 1
prob_dist = class_dist/class_dist.sum()
#chooses a random class depending on probability distribution 
choices = np.random.choice(class_dist.index, p=prob_dist)



#fig, ax = plt.subplots()
#ax.set_title('Class Distribution', y=1.08)
#ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
#      shadow=False, startangle=90)
#ax.axis('equal')
#plt.show()
#change mode to conv for CNN and time for RNN 
config = Config(mode='conv')

#builds random feature set from choices variable
if config.mode =='conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    #in keras, Xshape[0] which is the number of samples, is not needed for the first layer
    input_shape = (X.shape[1],X.shape[2],1)
    model = get_conv_model()
elif config.mode =='time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    input_shape = (X.shape[1],X.shape[2])
    model = get_recurrent_model()

#class weights allows the neural network to compensate enough for classes with 
#low probability distributions so that it learns those features better
#for example bass drums are only 2.7% of the distribution
#slightly improves accuracy/loss function reduces bias
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                            save_best_only=True, save_weights_only=False, period=1)

#model.fit takes the X,y matrices and creates batches of the data
#an epoch is one cycle through the full training dataset
history = model.fit(X, y, epochs=100, batch_size=256, shuffle=True, 
            class_weight=class_weight, validation_split=0.1, 
            callbacks = [checkpoint])

model.save(config.model_path)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
