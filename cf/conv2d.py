import pandas as pd
import numpy as np
import tensorflow as tf
import os.path
from os import path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from numpy import asarray
from numpy import save
import os
import librosa
import librosa.display
import glob 
import skimage

from tqdm import tqdm

df = pd.read_csv("D:\DEFENSE PRAC\code\\UrbanSound8K.csv")

arr = np.array(df["slice_file_name"])
fold = np.array(df["fold"])
cla = np.array(df["class"])

feature = []
label = []
def parser(row):
    # Function to load files and extract features
    for i in tqdm(range(8732)):
        file_name = 'D:\DEFENSE PRAC\code\\fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)        
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]

if path.exists("data.npy")==False:
    temp = parser(df)
    temp = np.array(temp)
    np.save('data.npy', temp)
else:
    temp = np.load('data.npy',allow_pickle=True)
    
data = temp.transpose()


X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([8732, 128])

for i in range(8732):
    X[i] = (X_[i])

Y = to_categorical(Y)

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)

X_train = X_train.reshape(6549, 16, 8, 1)
X_test = X_test.reshape(2183, 16, 8, 1)

input_dim = (16, 8, 1)
model = Sequential()

model = Sequential()
model.add(Conv2D(16,(3,3), activation='relu', strides=(1, 1), 
                padding = 'same', input_shape=input_dim))
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
opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer =opt, metrics =['acc'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 32, validation_data = (X_test, Y_test))
model.summary()

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


model.save('models/urban8kcnnmodel') 

new_model = tf.keras.models.load_model('models/urban8kcnnmodel')
predictions = new_model.predict(X_test)
score = new_model.evaluate(X_test, Y_test)
print(score)

preds = np.argmax(predictions, axis = 1)


result = pd.DataFrame(preds)
result.to_csv("urban8kResults.csv")
