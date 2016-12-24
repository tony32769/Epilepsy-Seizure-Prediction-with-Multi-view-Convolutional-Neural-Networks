from Processor import *
from Feature import *
from Setting import *

setting = Setting(path = "kaggleSolution/kaggleSettings.yml")

for j in xrange(7):
    for i in xrange(setting.splitNum):
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

        setting.loadSettings(name = featureName)
        feature = Feature(featureName, "kaggleSolution/kaggleSettings.yml")
        processor = Processor()
        basePath = setting.rawDataPath + feature.subjectName
        setting.loadSettings(name = feature.subjectName)

        X_train, y_train = processor.processDataPerSubject(basePath, trainOrTest="train", splitNum=setting.splitNum, sequence = i)
        #X_train, y_train = feature.fft(X_train, y_train, winLengthSec = 120)
        #feature.saveToDisk(trainOrTest = "train", name= str(i), featureName = "fft")
        X_train, y_train = feature.pca(X_train, y_train,winLengthSec = 120)
        feature.saveToDisk(trainOrTest = "train", name= str(i), featureName = "pca")

        X_test, y_test = processor.processDataPerSubject(basePath, trainOrTest="test",splitNum=setting.splitNum, sequence = i)
        #X_test, y_test = feature.fft(X_test, y_test, winLengthSec = 120)
        #feature.saveToDisk(trainOrTest="test", name = str(i), featureName = "fft")
        X_test, y_test = feature.pca(X_test, y_test, winLengthSec = 120)
        feature.saveToDisk(trainOrTest="test", name = str(i), featureName = "pca")
