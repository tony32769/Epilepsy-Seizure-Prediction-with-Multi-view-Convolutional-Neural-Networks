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
from keras.optimizers import SGD, Adadelta
from multiprocessing import Pool
from Setting import *
from utils.IntervalEvaluation import *
from keras.utils.visualize_util import plot


def run(setting):
    nb_filters = setting.nb_filter
    batch_size = setting.batch_size
    nb_epoch = setting.nb_epoch
    featureName = setting.name
    feature = Feature(featureName, "kaggleSolution/kaggleSettings.yml")
    X_train, Y_train = feature.loadFromDisk("pca", "train")
    X_train, Y_train = feature.overlapInEachHour()
    X_train, _ = feature.scaleAcrossTime(X_train)
    X_test,  Y_test = feature.loadFromDisk("pca", "test")
    X_test, _ = feature.scaleAcrossTime(X_test)
    channels = X_train.shape[1]
    bins = X_train.shape[2]
    steps = X_train.shape[3]
    X_pca_train, Y_pca_train = feature.loadFromDisk("fft", "train")
    X_pca_train, Y_pca_train = feature.overlapInEachHour()
    X_pca_train, _ = feature.scaleAcrossTime(X_pca_train)
    X_pca_test, Y_pca_test = feature.loadFromDisk("fft", "test")
    X_pca_test, _ = feature.scaleAcrossTime(X_pca_test)
    X_train, X_pca_train, Y_train = feature.shuffle(X_train, X_pca_train, Y_train)
    print X_train.shape

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1] * X_train.shape[2], X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1] * X_test.shape[2], X_test.shape[3])
    X_pca_train = X_pca_train.reshape(X_pca_train.shape[0], 1, X_pca_train.shape[1] * X_pca_train.shape[2], X_pca_train.shape[3])
    X_pca_test = X_pca_test.reshape(X_pca_test.shape[0], 1, X_pca_test.shape[1] * X_pca_test.shape[2], X_pca_test.shape[3])
    Y_train = np_utils.to_categorical(Y_train, 2)

    input1 = Input(shape = (1, channels * bins, steps), name="input1")
    seq1 = Convolution2D(nb_filters, channels * bins, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * bins, steps),
                           activation="relu"
                           )(input1)
    seq1 = Dropout(setting.dropout)(seq1)
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)
    seq1 = Dropout(setting.dropout)(seq1)
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)
    seq1 = Dropout(setting.dropout)(seq1)
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)
    seq1 = Dropout(setting.dropout)(seq1)
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)
    seq1 = Flatten()(seq1)
    output1 = Dense(setting.output1, activation="tanh")(seq1)

    input2 = Input(shape=(1, channels * 9, steps), name="input2")
    seq2 = Convolution2D(nb_filters, channels * 9, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * 9, steps),
                           activation="relu"
                           )(input2)
    seq2 = Dropout(setting.dropout)(seq2)
    seq2 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq2)
    seq2 = Dropout(setting.dropout)(seq2)
    seq2 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq2)
    seq2 = Dropout(setting.dropout)(seq2)
    seq2 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq2)
    seq2 = Dropout(setting.dropout)(seq2)
    seq2 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq2)
    seq2 = Flatten()(seq2)
    output2 = Dense(setting.output2, activation="tanh")(seq2)

    merged = merge([output1, output2], mode="concat")
    merged = Dense(512, activation="tanh")(merged)
    merged = Dense(256, activation="tanh")(merged)
    if str(setting.name) != "Dog_5":
        merged = Dense(128, activation="tanh")(merged)
    output = Dense(2, activation="softmax", name="output")(merged)
    model = Model(input = [input1, input2], output=[output])
    sgd = SGD(lr = setting.lr)
    model.compile(loss='binary_crossentropy', optimizer = "sgd")
    model.load_weights("kaggleSolution/weights/" + str(setting.name) + ".h5")
    #history = model.fit({'input1':X_train, 'input2':X_pca_train}, {'output':Y_train}, nb_epoch= nb_epoch, verbose = 1, batch_size = batch_size)
    #model.save_weights("kaggleSolution/weights/" + str(setting.name) + ".h5")

    plot(model, to_file="kaggleSolution/visualization/"+ str(setting.name) + ".png", show_shapes = True)
    predictions = model.predict({'input1':X_test, 'input2':X_pca_test})
    output = predictions[:,1]
    output = output.tolist()
    ans = zip(Y_test,output)
    dataFrame = DataFrame(data=ans, columns=["clip", "preictal"])
    dataFrame.to_csv(setting.savePath + featureName + ".csv", index=False, header = True)

subjectList = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
setting = Setting(path = "kaggleSolution/kaggleSettings.yml")
for subject in subjectList:
    run(setting.loadSettings(name=subject))
