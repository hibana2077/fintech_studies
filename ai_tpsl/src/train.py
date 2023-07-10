# import sklearn
import os
import sys
import argparse
import logging
import requests
import time
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#logging
log_format = "\033[1;36m" + "{:=^50}" + "\033[0m"  # Cyan color
log_content_format = "\033[1;33m" + "{:^10}" + "\033[0m"  # Yellow color
log_success_format = "\033[1;32m" + "{:^10}" + "\033[0m"  # Green color
log_loding_format = "\033[1;34m" + "{:^10}" + "\033[0m"  # Blue color
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#argparse
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-i', '--input', type=str, help='input file name', required=True)
parser.add_argument('-e', '--epochs', type=int, help='epochs', required=True)
parser.add_argument('-b', '--batch', type=int, help='batch size', required=True)
parser.add_argument('-l', '--learning', type=float, help='learning rate', required=True)
parser.add_argument('-m', '--model', type=str, help='model name', required=True)
parser.add_argument('-w', '--window', type=int, help='window size', required=True)
parser.add_argument('-ep', '--earlystop-patience', type=int, help='earlystop patience', required=True)
parser.add_argument('-ev', '--earlystop-verbose', type=int, help='earlystop verbose', required=True)

#train gpu cpu setting
logging.info(log_loding_format.format(' Train GPU CPU Setting '))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        logging.error(e)
        exit(1)
else:
    logging.info('No GPU')
logging.info(log_success_format.format(' End of Train GPU CPU Setting '))
logging.info(log_format.format(' Train GPU CPU Setting '))
logging.info(log_content_format.format('GPU: ' + str(gpus)))
logging.info(log_success_format.format(' End of Train GPU CPU Setting '))

#input info
args = parser.parse_args()
input_file:str = args.input
epochs:int = args.epochs
batch_size:int = args.batch
learning_rate:float = args.learning
model_name:str = args.model
window_size:int = args.window
earlystop_patience:int = args.earlystop_patience
earlystop_verbose:int = args.earlystop_verbose
if input_file == '':
    logging.error('input file is empty')
    exit(1)
if input_file.split('.')[-1] != 'csv':
    logging.error('input file is not csv')
    exit(1)
if not os.path.exists(input_file):
    logging.error('input file not exists')
    exit(1)
if epochs <= 0:
    logging.error('epochs must be positive')
    exit(1)
if batch_size <= 0:
    logging.error('batch size must be positive')
    exit(1)
if learning_rate <= 0:
    logging.error('learning rate must be positive')
    exit(1)
if model_name == '':
    logging.error('model name is empty')
    exit(1)
if window_size <= 0:
    logging.error('window size must be positive')
    exit(1)
if earlystop_patience <= 0:
    logging.error('earlystop patience must be positive')
    exit(1)
if earlystop_verbose <= 0:
    logging.error('earlystop verbose must be positive')
    exit(1)
logging.info(log_format.format(' Input Info '))
logging.info('Input File: ' + log_content_format.format(input_file))
logging.info('Epochs: ' + log_content_format.format(str(epochs)))
logging.info('Batch Size: ' + log_content_format.format(str(batch_size)))
logging.info('Learning Rate: ' + log_content_format.format(str(learning_rate)))
logging.info('Model Name: ' + log_content_format.format(str(model_name)))
logging.info('Window Size: ' + log_content_format.format(str(window_size)))
logging.info('Earlystop Patience: ' + log_content_format.format(str(earlystop_patience)))
logging.info('Earlystop Verbose: ' + log_content_format.format(str(earlystop_verbose)))
logging.info(log_format.format(' End of Input Info '))

#load data
logging.info(log_format.format(' Load Data '))
logging.info('Loading data from ' + log_content_format.format(input_file))
data = pd.read_csv(input_file)
logging.info('Data shape: ' + log_content_format.format(str(data.shape)))
logging.info('Data columns: ' + log_content_format.format(str(data.columns)))
logging.info('Data head: ')
print(data.head())
logging.info('Data tail: ')
print(data.tail())
logging.info(log_format.format(' End of Load Data '))

#make dataset
logging.info(log_format.format(' Make Dataset '))
logging.info('Making dataset')
#create sequence
X = data.drop(['date', 'TP1','TP2','TP3','SL'], axis=1)
y = data[['TP1','TP2','TP3','SL']]
print(X.head())
print(y.head())
if window_size > len(X):
    logging.error('window size must be less than data length , please change window size or add more data')
    exit(1)
def make_sequence(X, y, window_size):
    Xs, ys = [], []
    for i in range(window_size, len(X)):
        Xs.append(X.iloc[i-window_size:i].values)
        ys.append(y.iloc[i].values)
    return np.array(Xs), np.array(ys)
X_seq, y_seq = make_sequence(X, y, window_size)
logging.info(log_format.format(' Data Info '))
logging.info('X_seq shape: ' + log_content_format.format(str(X_seq.shape)))
logging.info('y_seq shape: ' + log_content_format.format(str(y_seq.shape)))
logging.info(log_format.format(' End of Data Info '))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
logging.info(log_format.format(' Train Test Split '))
logging.info('X_train shape: ' + log_content_format.format(str(X_train.shape)))
logging.info('y_train shape: ' + log_content_format.format(str(y_train.shape)))
logging.info('X_test shape: ' + log_content_format.format(str(X_test.shape)))
logging.info('y_test shape: ' + log_content_format.format(str(y_test.shape)))
logging.info(log_format.format(' End of Train Test Split '))
logging.info(log_format.format(' End of Make Dataset '))

#build model
logging.info(log_format.format(' Build Model '))
logging.info('Building model')
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(BatchNormalization(input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))

model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='log_cosh', metrics=['mae','mse','msle'])#huber,log_cosh
model.summary()
logging.info(log_format.format(' End of Build Model '))

#train model
logging.info(log_format.format(' Train Model '))
logging.info('Training model')
early_stop = EarlyStopping(monitor='val_loss', patience=earlystop_patience, verbose=earlystop_verbose)
#patience=10, verbose=1
#patience is the number of epochs to wait before early stop, verbose is the verbosity mode
#1 is the default value, 0 is silent
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=False, callbacks=[early_stop])
logging.info(log_format.format(' End of Train Model '))

#save model
logging.info(log_format.format(' Save Model '))
logging.info('Saving model')
#save model to file
model_name = model_name + '.h5' if not model_name.endswith('.h5') else model_name
model.save(model_name)
logging.info(log_format.format(' End of Save Model '))

#evaluate model
logging.info(log_format.format(' Evaluate Model '))
logging.info('Evaluating model')
score = model.evaluate(X_test, y_test, batch_size=batch_size)
logging.info('Test loss: ' + log_content_format.format(str(score)))
# logging.info('X_test [0]: ' + log_content_format.format(str(X_test[0])))
logging.info('y_test [0]: ' + log_content_format.format(str(y_test[0])))
pred = model.predict(X_test)
logging.info('pred [0]: ' + log_content_format.format(str(pred[0])))
# logging.info('X_train [0]: ' + log_content_format.format(str(X_train[0])))
logging.info('y_train [0]: ' + log_content_format.format(str(y_train[0])))
pred = model.predict(X_train)
logging.info('pred [0]: ' + log_content_format.format(str(pred[0])))
logging.info(log_format.format(' End of Evaluate Model '))

#plot loss
logging.info(log_format.format(' Plot Loss '))
logging.info('Plotting loss')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
#save plot
logging.info('Saving loss plot')
plt.savefig('loss.png')
logging.info(log_format.format(' End of Plot Loss '))