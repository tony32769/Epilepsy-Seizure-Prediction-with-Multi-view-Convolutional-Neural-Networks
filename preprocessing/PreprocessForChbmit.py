from Processor import *
from Feature import *
from Setting import *

setting = Setting("chbmitSolution/chbmitsettings.yml")
subjectNameList = ["chb01", "chb05", "chb06"]
for subjectName in subjectNameList:
        setting.loadSettings(name = subjectName)
        feature = Feature(subjectName)
        basePath = setting.rawDataPath + feature.subjectName
        X_train, y_train = processor.processDataPerSubject(basePath, trainOrTest="train", splitNum=setting.splitNum, sequence = i)
        X_train, y_train = feature.fft(X_train, y_train, samplingRate = setting.resampleFrequency)
        feature.saveToDisk(trainOrTest = "train", name= str(i), featureName = "fft")
        X_train, y_train = feature.pca(X_train, y_train,samplingRate = setting.resampleFrequency)
        feature.saveToDisk(trainOrTest = "train", name= str(i), featureName = "pca")
