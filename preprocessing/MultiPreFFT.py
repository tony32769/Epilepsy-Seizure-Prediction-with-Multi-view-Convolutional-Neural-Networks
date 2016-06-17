from Processor import *
from Feature import *
from multiprocessing import Pool

def run(j):
#for j in xrange(7):
    for i in xrange(10):
        featureName = None
        channels = None
        if j in range(5):
            featureName = "Dog_" + str(j + 1)
            if j == 4:
                channels = 15
            else:
                channels = 16
        else:
            if j == 6:
                channels = 24
            else:
                channels = 15
            featureName = "Patient_" + str(j - 4)
        feature = Feature(featureName)
        processor = Processor()
        basePath = "/home/xiaobin/Disk/kaggle-seizure-prediction/data/raw_data/"+feature.subjectName
        print basePath
        X_train, y_train = processor.processDataPerSubject(basePath, trainOrTest="train", splitNum=10, sequence = i)
        X_train, y_train = feature.pca(X_train, y_train)
        #X_train, y_train = feature.fft(X_train, y_train)
        #X_train, y_train = feature.transform4D(X_train, y_train, channels = channels )
        print "X_train shape" + str(X_train.shape)
        feature.saveToDisk(trainOrTest = "train", name= str(i))

        X_test, y_test = processor.processDataPerSubject(basePath, trainOrTest="test",splitNum=10, sequence = i)
        #X_test, y_test = feature.fft(X_test, y_test )
        X_train, y_train = feature.pca(X_test, y_test)
        #X_test, y_test = feature.transform4D(X_test, y_test, channels = channels, timeSlotNum = 20)

        feature.saveToDisk(trainOrTest="test", name = str(i))

pool = Pool(7)
pool.map(run,range(7))
