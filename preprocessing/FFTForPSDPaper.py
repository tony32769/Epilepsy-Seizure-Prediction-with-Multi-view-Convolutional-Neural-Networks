from Processor import *
from Feature import *

for j in xrange(7):

    featureName = None
    if j <= 4:
        featureName = "Dog_" + str(j + 1)
    else:
        featureName = "Patient_" + str(j - 4)

    feature = Feature(featureName)
    for i in xrange(10):
        #featureName = "Patient_2"
        #feature = Feature(featureName)
        processor = Processor()
        basePath = "/home/xiaobin/kaggle-seizure-prediction/data/raw_data/"+feature.subjectName
        print basePath

        X_train, y_train = processor.processDataPerSubject(basePath, trainOrTest="train", splitNum=10, sequence = i)
        if i == 9:
            #X_train, y_train = feature.featureSelection(X_train, y_train, samplingRate = 256, winLengthSec = 5, strideSec = 5, isEnd = True)
            X_train, y_train = feature.featureSelection(X_train, y_train, samplingRate = 400, bandNum = 8, winLengthSec = 30, strideSec=30)
        else:
            #X_train, y_train = feature.featureSelection(X_train, y_train, samplingRate = 256, winLengthSec = 5, strideSec = 5)
            X_train, y_train = feature.featureSelection(X_train, y_train, samplingRate = 400, bandNum = 8, winLengthSec = 30, strideSec=30)
            print "X_train shape" + str(X_train.shape)
        feature.saveToDisk(trainOrTest = "train", name= str(i))

        #X_test, y_test = processor.processDataPerSubject(basePath, trainOrTest="test",splitNum=10, sequence = i)
        #X_test, y_test = feature.featureSelection(X_test, y_test, samplingRate = 256, winLengthSec = 5, stride = 5)
        #X_train, y_train = feature.pca(X_test, y_test, winLengthSec = 120)
        #X_train, y_train = feature.ica(X_test, y_test)
        #X_train, y_train = feature.randomProjection(X_test, y_test)
        #X_test, y_test = feature.transform4D(X_test, y_test, channels = channels, timeSlotNum = 20)

        #feature.saveToDisk(trainOrTest="test", name = str(i))
