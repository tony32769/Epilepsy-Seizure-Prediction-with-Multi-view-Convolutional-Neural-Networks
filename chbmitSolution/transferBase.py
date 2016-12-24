import numpy
numpy.random.seed(1337)

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from preprocessing.Processor import *
from preprocessing.Feature import *
from pandas import DataFrame
from keras.regularizers import l1l2, activity_l2, l2, l1
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adadelta
from multiprocessing import Pool
from keras.layers import noise
from sklearn.cross_validation import StratifiedKFold, train_test_split,KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report,precision_score
import theano
from Setting import *

def run(setting, X_train,X_pca_train, Y_train, X_train_chb01, X_pca_train_chb01, Y_train_chb01):

    nb_filters = setting.nb_filter
    batch_size = setting.batch_size
    nb_epoch = setting.nb_epoch
    channels = X_train.shape[1]
    bins = X_train.shape[2]
    bins_pca = X_pca_train.shape[2]
    steps = X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1] * X_train.shape[2], X_train.shape[3])
    X_pca_train = X_pca_train.reshape(X_pca_train.shape[0], 1, X_pca_train.shape[1] * X_pca_train.shape[2], X_pca_train.shape[3])
    Y_train = np_utils.to_categorical(Y_train, 2)

    X_train_chb01 = X_train_chb01.reshape(X_train_chb01.shape[0], 1, X_train_chb01.shape[1] * X_train_chb01.shape[2], X_train_chb01.shape[3])
    X_pca_train_chb01 = X_pca_train_chb01.reshape(X_pca_train_chb01.shape[0], 1, X_pca_train_chb01.shape[1] * X_pca_train_chb01.shape[2], X_pca_train_chb01.shape[3])
    Y_train_chb01 = np_utils.to_categorical(Y_train_chb01, 2)

    input1 = Input(name="input1", shape = (1, channels * bins, steps))
    seq1 = noise.GaussianNoise(setting.noise, input_shape=(1, channels * bins, steps))(input1)
    seq1 = Convolution2D(nb_filters, channels * bins, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * bins, steps),
                           activation="relu"
                           )(seq1)
    #seq1 = Dropout(0.2)(seq1)
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)
    #seq1 = Dropout(0.2)(seq1)
    #seq1 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq1)
    #seq1 = Dropout(0.2)(seq1)
    #seq1 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq1)
    #seq1 = Dropout(0.2)(seq1)
    #seq1 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq1)
    seq1 = Flatten()(seq1)
    output1 = Dense(setting.output1, activation="tanh")(seq1)

    input2 = Input(name="input2", shape=(1, channels * bins_pca, steps))
    seq2 = noise.GaussianNoise(setting.noise, input_shape=(1, channels * bins_pca, steps))(input2)
    seq2 = Convolution2D(nb_filters, channels * bins_pca, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * bins_pca, steps),
                           activation="relu"
                           )(seq2)
    #seq2 = Dropout(0.2)(seq2)
    seq2 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq2)
    #seq2 = Dropout(0.2)(seq2)
    #seq2 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq2)
    #seq2 = Dropout(0.2)(seq2)
    #seq2 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq2)
    #seq2 = Dropout(0.2)(seq2)
    #seq2 = Convolution2D(nb_filters, 1, 3,
    #                       #init="uniform",
    #                       W_regularizer=l2(l=setting.l2),
    #                       activation="relu"
    #                       )(seq2)
    seq2 = Flatten()(seq2)
    output2 = Dense(setting.output2, activation="tanh")(seq2)

    merged = merge([output1, output2], mode="concat")
    merged = Dense(512, activation="tanh")(merged)
    merged = Dense(256, activation="tanh")(merged)
    #merged = Dense(128, activation="tanh")(merged)

    output = Dense(2, activation="softmax", name="output")(merged)
    model = Model(input = [input1, input2], output = [output])
    sgd = SGD(lr = 0.01)
    model.compile(loss="binary_crossentropy", optimizer = sgd, metrics=['accuracy'])

    #callback = ModelCheckpoint(filepath = "my_model_weights.h5", save_best_only = True)

    #history = model.fit({'input1':X_train, 'input2':X_pca_train}, {'output':Y_train}, callbacks = [callback], validation_data = ({"input1":X_train_chb01, "input2":X_pca_train_chb01}, {"output":Y_train_chb01}), nb_epoch=nb_epoch, verbose = 1, batch_size = batch_size, class_weight = [1, 1])
    history = model.fit({'input1':X_train, 'input2':X_pca_train}, {'output':Y_train},  validation_split = 0.3, batch_size = batch_size, class_weight = [1, 1], nb_epoch = nb_epoch)
    model.save_weights("my_model_weights.h5")

nameList = ["chb00"]
setting = Setting("chbmitSolution/transfersettings.yml")
for name in nameList:
    setting = setting.loadSettings("chb01")
    feature = Feature(setting.name, "chbmitSolution/transfersettings.yml")
    X_train_chb01, Y_train_chb01 = feature.loadFromDisk("mitfft","train")
    X_train_chb01, Y_train_chb01 = feature.overlapInEachHour()
    X_train_chb01, _ = feature.scaleAcrossTime(X_train_chb01)
    X_pca_train_chb01, Y_pca_train_chb01 = feature.loadFromDisk("mitpca","train")
    X_pca_train_chb01[numpy.isneginf(X_pca_train_chb01)] = 0
    X_pca_train_chb01, Y_pca_train_chb01 = feature.overlapInEachHour()
    X_pca_train_chb01, _ = feature.scaleAcrossTime(X_pca_train_chb01)

    setting = setting.loadSettings(name = name)
    feature = Feature(setting.name, "chbmitSolution/transfersettings.yml")
    X_train, Y_train = feature.loadFromDisk("mitfft","train")
    X_train, Y_train = feature.overlapInEachHour()
    X_train, _ = feature.scaleAcrossTime(X_train)
    X_pca_train, Y_pca_train = feature.loadFromDisk("mitpca","train")
    X_pca_train[numpy.isneginf(X_pca_train)] = 0
    X_pca_train, Y_pca_train = feature.overlapInEachHour()
    X_pca_train, _ = feature.scaleAcrossTime(X_pca_train)
    #cv = StratifiedKFold(Y_train_chb01, n_folds = 3, shuffle = True)
    #for i, (train, test) in enumerate(cv):
    run(setting, X_train, X_pca_train, Y_train, X_train_chb01, X_pca_train_chb01, Y_train_chb01)
