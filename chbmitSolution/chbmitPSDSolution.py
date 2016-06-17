import numpy
from sklearn.metrics import roc_curve, auc, accuracy_score,confusion_matrix
from sklearn.cross_validation import StratifiedKFold, train_test_split,KFold
from preprocessing.Processor import *
from preprocessing.Feature import *
from pandas import DataFrame
from multiprocessing import Pool
from sklearn import svm

def run(featureName):
    feature = Feature(featureName, "chbmitSolution/chbmitsettings.yml")
    X_train, Y_train = feature.loadFromDisk("mitpsd", "train")
    X_train, Y_train = feature.overlapInEachHour()
    X_train, _ = feature.scaleAcrossTime(X_train)

    channels = X_train.shape[1]
    bins = X_train.shape[2]
    steps = X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    X = X_train
    y = Y_train
    zeros = numpy.where(y == 0)
    zeros = len(zeros[0])
    ones = numpy.where(y == 1)
    ones = len(ones[0])

    aucList = []
    ssList = []
    spList = []
    acList = []

    for j in xrange(100):
        cv = StratifiedKFold(y, n_folds = 3, shuffle = True)
        clf = svm.SVC(probability = True, class_weight={ 1: 3})
        for i, (train, test) in enumerate(cv):
            clf.fit(X[train], y[train])
            prob = clf.predict(X[test])

            matrix = confusion_matrix(y[test],prob)

            TP = matrix[0][0]
            FN = matrix[0][1]
            FP = matrix[1][0]
            TN = matrix[1][1]
            AC = (TP + TN) / float(TP + FP + TN + FN)
            acList.append(AC)
            print "Accuracy calculated by matrix:" + str(AC)
            SS = TP / float(TP + FN)
            ssList.append(SS)
            print "Sensitivity calcuated by matrix:" + str(SS)
            SP = TN / float(TN + FP)
            spList.append(SP)
            print "Specificity calcuated by matrix:" + str(SP)
            fpr, tpr, thresholds = roc_curve(y[test], prob)
            roc_auc = auc(fpr, tpr)
            aucList.append(roc_auc)
            print "AUC:" + str(roc_auc)

    print "featureName:"
    print "mean auc: " + str(numpy.mean(aucList))
    print "mean SS:" + str(numpy.mean(ssList))
    print "mean SP:" + str(numpy.mean(spList))
    print "mean AC:" + str(numpy.mean(acList))

featureName = ["chb01", "chb05", "chb06"]
for name in featureName:
    run(name)
#pool = Pool(3)
#pool.map(run, featureName)
