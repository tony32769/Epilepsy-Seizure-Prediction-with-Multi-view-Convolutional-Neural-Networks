import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import StratifiedKFold, train_test_split,KFold
from preprocessing.Processor import *
from preprocessing.Feature import *
from pandas import DataFrame
from multiprocessing import Pool
from sklearn import svm

def run(setting):
    feature = Feature(setting.name)
    X_train, Y_train = feature.loadFromDisk("train","4D")
    X_train, Y_train = feature.overlapInEachHour()
    X_train, _ = feature.scaleAcrossTime(X_train)
    X_test,  Y_test = feature.loadFromDisk("test","4D")
    X_test, _ = feature.scaleAcrossTime(X_test)

    channels = X_train.shape[1]
    bins = X_train.shape[2]
    steps = X_train.shape[3]
    X_pca_train, Y_pca_train = feature.loadFromDisk("train","PCA")
    X_pca_train, Y_pca_train = feature.overlapInEachHour()
    X_pca_train, _ = feature.scaleAcrossTime(X_pca_train)
    X_pca_test, Y_pca_test = feature.loadFromDisk("test","PCA")
    X_pca_test, _ = feature.scaleAcrossTime(X_pca_test)
    X_train, X_pca_train, Y_train = feature.shuffle(X_train, X_pca_train, Y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0],  X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
    X_pca_train = X_pca_train.reshape(X_pca_train.shape[0], X_pca_train.shape[1] * X_pca_train.shape[2] * X_pca_train.shape[3])
    X_pca_test = X_pca_test.reshape(X_pca_test.shape[0], X_pca_test.shape[1] * X_pca_test.shape[2] * X_pca_test.shape[3])

    X = numpy.concatenate( (X_train, X_pca_train), axis = 1)
    y = Y_train
    X_test = numpy.concatenate((X_test,X_pca_test), axis = 1)
    zeros = numpy.where(y == 0)
    zeros = len(zeros[0])
    ones = numpy.where(y == 1)
    ones = len(ones[0])
    cv = StratifiedKFold(y, n_folds = 3, shuffle=True)
    clf = svm.SVC(probability = True, class_weight={1: zeros / ones})

    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0,1,100)
    for i, (train, test) in enumerate(cv):
        clf.fit(X[train], y[train])
        prob = clf.predict(X[test])
        accuracy = accuracy_score(y[test], prob)
        print "accuracy:" + str(accuracy)
        fpr, tpr, thresholds = roc_curve(y[test], prob)
        #TPR is also known as sensitivity
        #FPR is one minus the specificity or true negative rate.
        print "iteration:" + str(i)
        print "fpr:" + str(fpr)
        print "tpr:" + str(tpr)
        print auc(fpr, tpr)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print "mean auc " + str(mean_auc)
    clf.fit(X,y)
    result = clf.predict_proba(X_test)

    output = result[:,1]
    ans = zip(Y_test,output)
    dataFrame = DataFrame(data=ans, columns=["clip", "preictal"])
    dataFrame.to_csv(setting.savePath + featureName + ".csv", index=False, header = True)
featureName = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
pool = Pool(7)
pool.map(run, featureName)
