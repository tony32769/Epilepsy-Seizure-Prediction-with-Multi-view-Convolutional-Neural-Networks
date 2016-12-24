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

def run(setting, X_train, Y_train, X_test, X_pca_train, Y_pca_train, X_pca_test):

    nb_filters = setting.nb_filter
    batch_size = setting.batch_size
    nb_epoch = setting.nb_epoch
    channels = X_train.shape[1]
    bins = X_train.shape[2]
    steps = X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1] * X_train.shape[2], X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1] * X_test.shape[2], X_test.shape[3])
    X_pca_train = X_pca_train.reshape(X_pca_train.shape[0], 1, X_pca_train.shape[1] * X_pca_train.shape[2], X_pca_train.shape[3])
    X_pca_test = X_pca_test.reshape(X_pca_test.shape[0], 1, X_pca_test.shape[1] * X_pca_test.shape[2], X_pca_test.shape[3])
    Y_train = np_utils.to_categorical(Y_train, 2)

    input1 = Input(name="input1", shape = (1, channels * bins, steps))
    seq1 = noise.GaussianNoise(setting.noise, input_shape=(1, channels * bins, steps))(input1)
    seq1 = Convolution2D(nb_filters, channels * bins, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * bins, steps),
                           activation="relu"
                           )(seq1)
    #seq1.add(Dropout(0.1))
    seq1 = Convolution2D(nb_filters, 1, 3,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           activation="relu"
                           )(seq1)

    seq1 = Flatten()(seq1)
    output1 = Dense(setting.output1, activation="tanh")(seq1)

    input2 = Input(name="input2", shape=(1, channels * 9, steps))
    seq2 = noise.GaussianNoise(setting.noise, input_shape=(1, channels * 9, steps))(input2)
    seq2 = Convolution2D(nb_filters, channels * 9, 1,
                           #init="uniform",
                           W_regularizer=l2(l=setting.l2),
                           input_shape=(1, channels * 9, steps),
                           activation="relu"
                           )(seq2)

    #seq2.add(Dropout(0.1))
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
    #merged = Dense(128, activation="tanh")(merged)

    output = Dense(2, activation="softmax", name="output")(merged)
    model = Model(input = [input1, input2], output = [output])
    sgd = SGD(lr = 0.01)
    model.compile(loss="binary_crossentropy", optimizer = sgd)

    history = model.fit({'input1':X_train, 'input2':X_pca_train}, {'output':Y_train}, nb_epoch=nb_epoch, verbose = 1, batch_size = batch_size)
    predictions = model.predict({'input1':X_test, 'input2':X_pca_test})
    output = predictions[:,1]
    outputList = []
    for i in xrange(output.shape[0]):
        if output[i] >= 0.5:
            outputList.append(1)
        else:
            outputList.append(0)
    output = numpy.array(outputList)

    return output

#nameList = ["chb01", "chb05", "chb06"]
nameList = ["chb06"]
setting = Setting("chbmitSolution/chbmitsettings.yml")
for name in nameList:
    setting = setting.loadSettings(name = name)
    feature = Feature(setting.name, "chbmitSolution/chbmitsettings.yml")
    X_train, Y_train = feature.loadFromDisk("mitpca","train")
    X_train, Y_train = feature.overlapInEachHour()
    X_train, _ = feature.scaleAcrossTime(X_train)
    X_pca_train, Y_pca_train = feature.loadFromDisk("mitfft","train")
    X_pca_train, Y_pca_train = feature.overlapInEachHour()
    X_pca_train, _ = feature.scaleAcrossTime(X_pca_train)

    aucList = []
    ssList = []
    spList = []
    acList = []

    for j in xrange(1):
        print j
        cv = StratifiedKFold(Y_train, n_folds = 3, shuffle=True)
        for i, (train, test) in enumerate(cv):
            prob = run(setting, X_train[train], Y_train[train], X_train[test], X_pca_train[train], Y_pca_train[train], X_pca_train[test])
            y = Y_train[test]
            matrix = confusion_matrix(y, prob)

            TP = matrix[0][0]
            FN = matrix[0][1]
            FP = matrix[1][0]
            TN = matrix[1][1]
            print matrix

            AC = (TP + TN)/float(TP+FP+TN+FN)
            acList.append(AC)
            print "Accuracy calculated by matrix:" + str(AC)
            SS = TP / float(TP + FN)
            ssList.append(SS)
            print "Sensitivity calculated by matrix:" + str(SS)
            SP = TN / float(TN + FP)
            spList.append(SP)
            print "Specificity calculated by matrix:" + str(SP)

            fpr, tpr, thresholds = roc_curve(y, prob, pos_label = 1)
            roc_auc = auc(fpr, tpr)
            aucList.append(roc_auc)
            print "AUC:" + str(roc_auc)

    print "mean auc:" + str(numpy.mean(aucList))
    print "mean SS:" + str(numpy.mean(ssList))
    print "mean SP:" + str(numpy.mean(spList))
    print "mean AC:" + str(numpy.mean(acList))
