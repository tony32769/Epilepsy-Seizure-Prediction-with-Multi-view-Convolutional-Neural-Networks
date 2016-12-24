import numpy
numpy.random.seed(1337)

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from preprocessing.Processor import *
from preprocessing.Feature import *
from pandas import DataFrame
from keras.regularizers import l1l2, activity_l2, l2, l1
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adadelta
from multiprocessing import Pool
from keras.layers import noise
from Setting import *

def run(setting):
    nb_filters = setting.nb_filter
    batch_size = setting.batch_size
    nb_epoch = setting.nb_epoch
    featureName = setting.name
    feature = Feature(featureName, "kaggleSolution/kaggleSettings.yml")
    X_train, Y_train = feature.loadFromDisk("fft", "train")
    X_train, Y_train = feature.overlapInEachHour(shuffle = True)
    X_train, _ = feature.scaleAcrossTime(X_train)
    X_test,  Y_test = feature.loadFromDisk("fft", "test")
    X_test, _ = feature.scaleAcrossTime(X_test)
    channels = X_train.shape[1]
    bins = X_train.shape[2]
    steps = X_train.shape[3]

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1] * X_train.shape[2], X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1] * X_test.shape[2], X_test.shape[3])
    Y_train = np_utils.to_categorical(Y_train, 2)

    model = Sequential()
    seq1 = noise.GaussianNoise(setting.noise, input_shape=(1, channels * bins, steps))
    seq2 = Convolution2D(nb_filters, channels * bins, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * bins, steps),
                           activation="relu"
                           )
    seq3 = Dropout(setting.dropout)
    seq4 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )
    seq5 = Dropout(setting.dropout)
    seq6 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )
    seq7 = Dropout(setting.dropout)
    seq8 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )
    seq9 = Dropout(setting.dropout)
    seq10 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )
    seq11 = Flatten()
    seq12 = Dense(setting.output1, activation="tanh")

    seq13 = Dense(512, activation="tanh")
    seq14 = Dense(256, activation="tanh")
    seq15 = Dense(128, activation="tanh")
    seq16 = Dense(2, activation="softmax", name="output")

    model.add(seq1)
    model.add(seq2)
    model.add(seq3)
    model.add(seq4)
    model.add(seq5)
    model.add(seq6)
    model.add(seq7)
    model.add(seq8)
    model.add(seq9)
    model.add(seq10)
    model.add(seq11)
    model.add(seq12)
    model.add(seq13)
    model.add(seq14)
    model.add(seq15)
    model.add(seq16)

    plot(model,to_file = featureName+".png", show_shapes = True)
    sgd = SGD(lr = 0.01)
    model.compile(loss='binary_crossentropy', optimizer = sgd)
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, verbose = 1, batch_size = batch_size)
    predictions = model.predict(X_test)
    output = predictions[:,1]
    output = output.tolist()
    ans = zip(Y_test,output)
    dataFrame = DataFrame(data=ans, columns=["clip", "preictal"])
    dataFrame.to_csv(setting.savePath + featureName + ".csv", index=False, header = True)

subjectList = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
setting = Setting(path = "kaggleSolution/kaggleSettings.yml")
for subject in subjectList:
    run(setting.loadSettings(name=subject))
