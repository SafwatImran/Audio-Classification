import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

#tqdm shows progress bar

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1



#calculates envelope of a signal which is also called noisefloor detection
#it's an estimation of the significant parts of a signal, in order to remove the 'dead' parts
def envelope (y,rate,threshold):
    mask=[]
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1, center=True).mean()
    for mean in y_mean : 
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask 
#function to calculate fft
#takes the signal and rate as parameters
def calc_fft (y , rate):
    n = len(y)
    #uses numpy realfft function, takes length of the signal and spacing as parameters
    #to calculate frequency 
    freq = np.fft.rfftfreq(n, d=1/rate)
    #takes magnitude of the complex return value of rfft
    #normalizes using the length of the signal n
    Y = abs(np.fft.rfft(y)/n)
    #returns the magnitude and frequency 
    return (Y, freq)

df = pd.read_csv('fold1.csv')
#setting filename column as the index
df.set_index('fname',inplace=True)

# iterates through the indexes
# df.index holds the filename column
for f in df.index :
    #reads in individual file's rate and signal from wavfiles
    rate, signal = wavfile.read('wavfiles/'+f)
    #creates new column called length in seconds
    df.at[f,'length'] = signal.shape[0]/rate

#classes are the unique labels (saxophone,drums)
classes = list (np.unique(df.label))

#class distribution - group different labels together, access length and find out the mean
class_dist = df.groupby(['label'])['length'].mean()

#Plotting class distribution piechart
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
#piechart with labels from class distribution, values formatted to 1 d.p., starts from 90 degrees 
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
#moves filename backs to it's own column in the dataframe
df.reset_index(inplace=True)

#dictionaries of the classes 
signals={}
fft = {}
fbank = {}
mfccs = {}

#taking 1 from each class
for c in classes : 
    wav_file = df[df.label == c].iloc[0,0]
    signal, rate = librosa.load('wavfiles/'+ wav_file, sr=44100)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    #calculates fft 
    fft[c] = calc_fft(signal, rate)
    #signal[:rate] is length=1s
    #nfft = 44100/40 is the window length for fft
    #adds padding to match some power of 2 
    bank = logfbank(signal[:rate],rate, nfilt=26, nfft=1103)
    fbank[c]= bank
    #numcep is half the number of ceptuals kept after calculating the Discrete Cosine trasnforms on the filter bank energies 
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103) 
    mfccs[c] = mel

plot_signals(signals)
plt.show()

plot_fft (fft)
plt.show

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

#downsamples the wavfiles if the clean folder is empty and applies the mask
#which is used in the actual model
#this is done because there's not much data in very high frequencies
if len(os.listdir('clean1')) == 0:
    for f in tqdm (df.fname):
        signal,rate = librosa.load('wavfiles/'+f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean1/'+f, rate=rate,data=signal[mask])
