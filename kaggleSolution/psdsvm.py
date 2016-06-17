import numpy
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import StratifiedKFold, train_test_split,KFold
from preprocessing.Processor import *
from preprocessing.Feature import *
from pandas import DataFrame
from multiprocessing import Pool
from sklearn import svm

def run(featureName):
    feature = Feature(featureName)
    X_train, Y_train = feature.loadFromDisk("train","psd")
    X_train, Y_train = feature.overlapInEachHour()
    X_test,  Y_test = feature.loadFromDisk("test","psd")
    X_test, _ = feature.scaleAcrossTime(X_test)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0],  X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
    X = X_train
    X_test = X_test
    y = Y_train
    zeros = numpy.where(y == 0)
    zeros = len(zeros[0])
    ones = numpy.where(y == 1)
    ones = len(ones[0])

    clf = svm.SVC(probability = True, class_weight={1: zeros / ones})
    clf.fit(X,y)
    result = clf.predict_proba(X_test)

    output = result[:,1]
    ans = zip(Y_test,output)
    dataFrame = DataFrame(data=ans, columns=["clip", "preictal"])
    dataFrame.to_csv("/home/xiaobin/Disk/PSDSVMResult/" + featureName + ".csv", index=False, header = True)

featureName = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
pool = Pool(7)
pool.map(run, featureName)
