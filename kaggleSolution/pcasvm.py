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
    feature = Feature(setting.name, "kaggleSolution/kaggleSettings.yml")
    X_pca_train, Y_pca_train = feature.loadFromDisk("PCA","train")
    X_pca_train, Y_pca_train = feature.overlapInEachHour(shuffle = True)
    X_pca_train, _ = feature.scaleAcrossTime(X_pca_train)
    X_pca_test, Y_pca_test = feature.loadFromDisk("PCA","test")
    X_pca_test, _ = feature.scaleAcrossTime(X_pca_test)

    channels = X_pca_train.shape[1]
    bins = X_pca_train.shape[2]
    steps = X_pca_train.shape[3]
    X_pca_train = X_pca_train.reshape(X_pca_train.shape[0], X_pca_train.shape[1] * X_pca_train.shape[2] * X_pca_train.shape[3])
    X_pca_test = X_pca_test.reshape(X_pca_test.shape[0], X_pca_test.shape[1] * X_pca_test.shape[2] * X_pca_test.shape[3])

    X = X_pca_train
    y = Y_pca_train
    X_test = X_pca_test
    Y_test = Y_pca_test
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
    dataFrame.to_csv(setting.savePath + setting.name + ".csv", index=False, header = True)
featureName = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
settingList = []
for subject in featureName:
    setting = Setting(path = "kaggleSolution/kaggleSettings.yml")
    setting.loadSettings(subject)
    settingList.append(setting)
    run(setting)
#pool = Pool(7)
#pool.map(run, settingList)
