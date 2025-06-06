import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import sklearn.metrics as metrics
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm
import statsmodels
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pathlib
import shutil
import tempfile
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

df = pd.read_excel(r"/Users/trieuhoanghiep/Documents/Me/Project/VN STOCK/Simplize_ACB_PriceHistory_20240526.xlsx")
df_ts = df
df_ts['DATE'] = pd.to_datetime(df_ts['DATE'])
X = df_ts[['DATE','CLOSE']]
X.set_index('DATE',inplace = True)
X = X.iloc[-200:]
diff_1 = X.diff(1).dropna()

df_ACB = df_ts
X_ACB_diff_1 = diff_1

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return X, y

# define input sequence
# choose a number of time steps
n_steps = 5
# split into samples
ACB_recent = X_ACB_diff_1['CLOSE']

X,y= split_sequence(ACB_recent, n_steps)

size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
size_test = int(len(test) * 0.8)
valid, oot = test[0:size_test], test[size_test:len(test)]

size1 = int(len(y) * 0.8)
train1, test1 = y[0:size], y[size:len(y)]
size_test1 = int(len(test1) * 0.8)
valid1, oot1 = test1[0:size_test], test1[size_test:len(test)]

X_train = np.hstack(train).reshape(len(train),n_steps,1)
y_train = np.hstack(train1).reshape(len(train1),1)
X_valid = np.hstack(valid).reshape(len(valid),n_steps,1)
y_valid = np.hstack(valid1).reshape(len(valid1),1)
print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape)

input_length = X_train.shape[1]
input_dim = X_train.shape[2]

def eval_metrics(actual, pred):
    y_true, y_pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mlflow.set_tracking_uri("http://0.0.0.0:5005")
mlflow.set_experiment("Time-series")
with mlflow.start_run():
    N_VALIDATION = int(1e3)
    N_TRAIN = int(1e4)
    BUFFER_SIZE = int(1e4)
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.1,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=40,
    staircase=False)

    def get_optimizer():
        return tf.keras.optimizers.Adam(lr_schedule)
    logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)   
    def get_callbacks(name):
        return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode="min",patience=5,restore_best_weights = True),
        tf.keras.callbacks.TensorBoard(logdir/name),
        ]
    def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
        if optimizer is None:
            optimizer = get_optimizer()
        model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=[tf.keras.losses.MeanAbsoluteError()])

        history = model.fit(
            X_train, y_train,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs=max_epochs,
            callbacks=get_callbacks(name),
            validation_data=(X_valid, y_valid),
            verbose=0)
        return history,model
    
    tiny_model = Sequential()
    tiny_model.add(LSTM(64, activation='silu', input_shape=(n_steps, 1)))
    tiny_model.add(Dense(1))
    size_histories = {}
    size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny','adam')
    tiny_model.save('tiny_model.h5')

    yhat_train = tiny_model.predict(X_train)
    signature = infer_signature(X_train, yhat_train)

    X_test = np.hstack(test).reshape(len(test),n_steps,1)
    yhat = tiny_model.predict(X_test)
    mape = eval_metrics(test1, yhat)
    print(f"Decision Tree model:")
    print("  MAPE: %s" % mape)

    mlflow.log_metric("MAPE", mape)
    mlflow.tensorflow.log_model(tiny_model,artifact_path="time-series-model")