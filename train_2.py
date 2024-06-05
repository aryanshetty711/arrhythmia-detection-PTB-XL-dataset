"""NOTES: Batch data is different each time in keras, which result in slight differences in results."""
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, MaxPooling2D, Reshape, multiply, Conv2D, GlobalAveragePooling2D, Dense, Multiply
from keras.models  import  Model, load_model
from tensorflow.keras.layers import Input
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, accuracy_score, auc, precision_score
import random
from sklearn.neighbors import NearestNeighbors
from keras.utils import np_utils

# utils
def load_data(path):
    x_train = np.load(path +'x_train.npy',allow_pickle=True)
    Y_train = np.load(path + 'y_train.npy',allow_pickle=True)
    x_test = np.load(path + 'x_test.npy',allow_pickle=True)
    Y_test = np.load(path + 'y_test.npy',allow_pickle=True)

    x_train = x_train.transpose(0, 2, 1)  # transpose working correctly
    x_test = x_test.transpose(0, 2, 1)

    # Two classes
    y_test = np.zeros((x_test.shape[0], 1))
    for i in range(x_test.shape[0]):
        y_test[i] = 1 if 'MI' in Y_test[i] or 'STTC' in Y_test[i] or 'CD' in Y_test[i] or 'HYP' in Y_test[i] else 0
    y_test = np_utils.to_categorical(y_test, num_classes=2)

    y_train = np.zeros((x_train.shape[0], 1))
    for i in range(x_train.shape[0]):
        y_train[i] = 1 if 'MI' in Y_train[i] or 'STTC' in Y_train[i] or 'CD' in Y_train[i] or 'HYP' in Y_train[i] else 0
    y_train = np_utils.to_categorical(y_train, num_classes=2)

    x_train = x_train.reshape(x_train.shape[0], 12, 1000, 1)
    x_test  = x_test.reshape(x_test.shape[0], 12, 1000, 1)
    print('Data loaded')
    print("x_train :", x_train.shape)
    print("y_train :", y_train.shape)
    print("x_test  :", x_test.shape)
    print("y_test  :", y_test.shape)
    return x_train, x_test, y_train, y_test


# Upsampling
def get_minority_samples(X, y):
    """
    return
    X_sub: the feature vector minority array
    y_sub: the target vector minority array
    """
    tail_labels = 0 #Normal
    y_tail = y[:,tail_labels]
    index = np.argwhere(y_tail==1)
    X_sub = X[index]
    y_sub = np.squeeze(y[index])
    return X_sub, y_sub

def nearest_neighbour(X, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: array, input vector
    y: array, feature vector
    n_sample: int, number of newly generated sample

    return
    new_X: array, augmented feature vector data
    target: array, augmented target vector data
    """
    X = X.reshape(X.shape[0],-1)
    indices2 = nearest_neighbour(X, neigh=neigh)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = np.squeeze(y[all_point])
        ser = np.sum(nn_df,axis = 0)
        for j in range(y.shape[1]):
            val = ser[j]
            target[i][j]= 1 if val > 0 else 0
        ratio = random.random()
        gap = X[reference] - X[neighbor]
        new_X[i] = np.array(X[reference] + ratio * gap)
    new_X = new_X.reshape(-1, 12, 1000, 1)
    return new_X, target

def SMOLTE_cat_wrapper(X, y,  nsamples):
    x_sub, y_sub = get_minority_samples(X, y)
    X_up, y_up = MLSMOTE(x_sub, y_sub, nsamples, 5)
    print('Number of new samples created: %d' %(len(y_up)))
    X_up = np.append(X, X_up, axis=0)
    y_up = np.append(y, y_up, axis=0)
    return X_up, y_up

# Model
def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr

def create_model(input_shape, weight=1e-3):
    #CNN module
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input1)
    x1 = Conv2D(64, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = Conv2D(96, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling2D(pool_size=3, padding="same")(x1)
    x1 = Conv2D(128, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling2D(pool_size=5, padding="same")(x1)

    # Channel-wise attention module
    squeeze = GlobalAveragePooling2D()(x1)
    excitation = Dense(96, activation='relu')(squeeze)
    excitation = Dense(128, activation='sigmoid')(excitation)
    excitation = Reshape((1,1,128))(excitation)
    scale = Multiply()([x1, excitation])

    #Fully connection
    x = GlobalAveragePooling2D()(scale)
    dp = Dropout(0.5)(x)
    output = Dense(2, activation='sigmoid')(dp)
    model = Model(inputs=input1, outputs=output)
    return model

# Evaluation metrics
def sklearn_metrics(y_true, y_pred):
    y_bin = np.copy(y_pred)
    y_bin[y_bin >= 0.5] = 1
    y_bin[y_bin < 0.5] = 0
    # Compute area under precision-Recall curve
    auc_sum = 0
    for i in range(2):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        auc_sum += auc(recall, precision)
    acc = accuracy_score(y_true.flatten(), y_bin.flatten()) * 100
    prec = precision_score(y_true.flatten(), y_bin.flatten()) * 100
    AUC = roc_auc_score(y_true, y_pred, average='macro') * 100
    auprc = (auc_sum / 2) * 100
    f1 = f1_score(y_true, y_bin, average='micro') * 100
    print("Accuracy        : {:.2f}".format(acc))
    print("Precision       : {:.2f}".format(prec))
    print("Macro AUC score : {:.2f}".format(AUC))
    print('AUPRC           : {:.2f}'.format(auprc))
    print("Micro F1 score  : {:.2f}".format(f1))
    return np.array([acc,prec,AUC,auprc,f1])

#Cross validation
def run_cv(x_train, x_test, y_train, y_test ,nsamples=100):
    N_FOLDS = 9
    histories = {x: '' for x in range(1,N_FOLDS)}
    models = {x: '' for x in range(1,N_FOLDS)}
    results = {x: '' for x in range(1,N_FOLDS)}
    Y = pd.read_csv('./1.0.1/ptbxl_database.csv', index_col='ecg_id')
    Y.drop(Y[Y.strat_fold ==10].index,inplace=True)
    res = [0, 0,  0,  0,  0]
    for foldno in range(1,10):
        x_train_fold = x_train[Y.strat_fold != foldno]
        y_train_fold = y_train[Y.strat_fold != foldno]
        x_val_fold = x_train[Y.strat_fold == foldno]
        y_val_fold = y_train[Y.strat_fold == foldno]
        
        train_sample_size = len(y_train_fold)
        val_sample_size = len(y_val_fold)
        print(" ")
        print(f"Fold-%d" % (foldno))
        print("Original Train sample size:", train_sample_size, ", Original validation sample size:", val_sample_size)
        x_train_fold, y_train_fold = SMOLTE_cat_wrapper(x_train_fold, y_train_fold,  nsamples=nsamples)
        print("Upsampled Train sample size: %d" % (len(x_train_fold)))

        # Model & Fit
        models[foldno] = create_model(x_train.shape[1:])
        filepath = 'weights.best.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        lr_scheduler = LearningRateScheduler(lr_schedule)
        models[foldno].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        callbacks_list = [lr_scheduler, checkpoint]
        histories[foldno] = models[foldno].fit(x_train_fold, y_train_fold, batch_size=128, epochs=100,
                         validation_data=(x_val_fold, y_val_fold), callbacks=callbacks_list)

        # Test Predictions
        strat_fold = models[foldno].predict(x_test)
        results[foldno] = sklearn_metrics(y_test,strat_fold)
        res = res + results[foldno]
        models[foldno].save(f'weights-fold{foldno}.hdf5')
    res = res / N_FOLDS
    print('\n')
    print('Summary')
    # Mean out of score -- Test set
    print('Mean score -- Test set:')
    print("Accuracy        : {:.2f}".format(res[0]))
    print("Precision       : {:.2f}".format(res[1]))
    print("Macro AUC score : {:.2f}".format(res[2]))
    print('AUPRC           : {:.2f}'.format(res[3]))
    print("Micro F1 score  : {:.2f}".format(res[4]))
    return res, histories

if __name__ == "__main__":
    # load_data
    path = './1.0.1/' # Path to dataset
    x_train, x_test, y_train, y_test= load_data(path)
    res, histories = run_cv(x_train, x_test, y_train,  y_test,nsamples=int(0.5*len(x_train)))
