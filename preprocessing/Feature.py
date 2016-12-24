import pandas
import random
import numpy
import glob
import itertools
from scipy import signal
from ussociety.MatFile import *
from Processor import *
from Setting import *
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.random_projection

class Feature:
    def __init__(self, subjectName, settingPath):
        self.subjectName = subjectName
        self.setting = Setting(settingPath).loadSettings(name = subjectName)
        self.resultArrayX = None
        self.resultArrayY = None
        self.classZeroHistogram = None #for paper "Epileptic seizure prediction using relative spectral power features"
        self.classOneHistogram = None #for paper "Epileptic seizure prediction using relative spectral power features"

    def groupIntoBands(self, fftData, fftFrequency, bandNum):
        bands = None
        if bandNum == 5:#for comparsion purpose, used to reimplement the paper called "Epileptic seizure prediction using relative spectral power features"
            bands = [0.5, 4, 8, 15, 30, 128]
        if bandNum == 8:
            bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
        if bandNum == 80:
            bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 128]
        if bandNum == 10:
            bands = [0.1, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        if bandNum == 16:
            bands = [0.1, 2, 4, 6, 8, 10, 12, 21, 30, 40, 50, 60, 70, 85, 100, 140, 180]
        if bandNum == 18:
            bands = [0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

        frequencyBands = numpy.digitize(fftFrequency, bands)
        if fftData.ndim > 1:
            channels = fftData.shape[0]
            result = []
            for i in xrange(channels):
                dataFrame = pandas.DataFrame({"fft": fftData[i], "band": frequencyBands})
                dataFrame = dataFrame.groupby("band").mean()
                result.append(dataFrame.fft[1: -1])
            return result
        dataFrame = pandas.DataFrame({"fft": fftData, "band": frequencyBands})
        dataFrame = dataFrame.groupby("band").mean()

        return dataFrame.fft[1: -1]

    def fft(self, dataArrayX, dataArrayY, bandNum=8, samplingRate=400, winLengthSec = 30, strideSec = 30):
        #dataArrayX's shape is matFileNumber * channels * matdata
        print "In function fft:"
        print dataArrayX.shape

        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1
        newArray = numpy.zeros((dataArrayX.shape[0], channels, bandNum + 1, steps))
        size = dataArrayX.shape[0]
        for i in xrange(size):
            for j in xrange(channels):
                for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                    data = dataArrayX[i, j, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                    fftData = numpy.log10(numpy.absolute(numpy.fft.rfft(data)))
                    fftFrequency = numpy.fft.rfftfreq(n = data.shape[-1], d = 1.0 / samplingRate)
                    #newArray[i, j, : bandNum, frameIndex] = self.groupIntoBands(fftData, fftFrequency, bandNum = self.setting.bandNum)
                    newArray[i, j, : bandNum, frameIndex] = self.groupIntoBands(fftData, fftFrequency, bandNum = 8)
                    newArray[i, j, -1, frameIndex] = numpy.std(data)

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY

        return self.resultArrayX, self.resultArrayY

    def daubWavelet(self, dataArrayX, dataArrayY, bandNum=8, waveletMode = 1, samplingRate=400, winLengthSec = 30, strideSec = 30):
        #dataArrayX's shape is matFileNumber * channels * matdata
        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1

        size = dataArrayX.shape[0]
        waveLength = 4 * (waveletMode * 2 + 1)
        newArray = numpy.zeros((dataArrayX.shape[0], channels, waveLength, steps))

        for i in xrange(size):
            for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                data = dataArrayX[i, :, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                newWavelet = numpy.zeros(( channels, waveLength ))
                for j in xrange(data.shape[0]):
                    waveletData = pywt.wavedec(data[j, :], "db%d" % waveletMode, level = waveletMode *2)
                    for offset, x in enumerate(waveletData):
                        newWavelet[j, offset * 4] = numpy.mean(x)
                        newWavelet[j, offset * 4 + 1] = numpy.std(x)
                        newWavelet[j, offset * 4 + 2] = numpy.min(x)
                        newWavelet[j, offset * 4 + 3] = numpy.max(x)

                #waveletData = numpy.log10(numpy.absolute(newWavelet))
                waveletData = numpy.absolute(newWavelet)
                newArray[i, :, :, frameIndex] = waveletData

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY

        return self.resultArrayX, self.resultArrayY

    def saveToDisk(self, name, featureName, trainOrTest = "train"):
        if self.resultArrayX is None or self.resultArrayY is None:
            raise "no data to save, fft first"

        savePath = self.setting.processedDataPath + featureName
        if trainOrTest == "train":
            numpy.save(savePath + "/" + self.subjectName + "/" + name + "_trainX", self.resultArrayX)
            numpy.save(savePath + "/" + self.subjectName + "/" + name + "_trainY", self.resultArrayY)
        else:
            numpy.save(savePath + "/" + self.subjectName + "/" + name + "_testX", self.resultArrayX)
            numpy.save(savePath + "/" + self.subjectName + "/" + name + "_testY", self.resultArrayY)

    def loadFromDisk(self, featureName, trainOrTest = "train"):
        savePath = self.setting.processedDataPath + featureName
        files = None
        if trainOrTest == "train":
            files = glob.glob(savePath+  "/"+ self.subjectName+ "/*trainX.npy")
        elif trainOrTest == "test":
            files = glob.glob(savePath+ "/"+ self.subjectName+ "/*testX.npy")

        files = sorted(files)
        trainX = None
        trainY = None
        testX = None
        testY = None

        if trainOrTest == "train":
            trainX = numpy.load(files[0])
            fileName = files[0].replace("trainX", "trainY")
            trainY = numpy.load(fileName)
        elif trainOrTest == "test":
            testX = numpy.load(files[0])
            fileName = files[0].replace("testX", "testY")
            testY = numpy.load(fileName)

        for f in files[1:]:
            if trainOrTest == "train":
                temp = numpy.load(f)
                trainX = numpy.concatenate((trainX, numpy.load(f)), axis = 0)
                f = f.replace("trainX","trainY")
                trainY = numpy.concatenate((trainY, numpy.load(f)), axis = 0)
            elif trainOrTest == "test":
                testX = numpy.concatenate((testX, numpy.load(f)), axis = 0)
                f = f.replace("testX","testY")
                testY = numpy.concatenate((testY, numpy.load(f)), axis = 0)
        if trainOrTest == "train":
            self.resultArrayX = trainX
            self.resultArrayY = trainY
            return trainX, trainY
        elif trainOrTest == "test":
            self.resultArrayX = testX
            self.resultArrayY = testY
            return testX, testY

    def overlapInEachHour(self, shuffle=False, smote= False):
        shapeX = self.resultArrayX.shape
        shapeY = self.resultArrayY.shape

        zeroIndics = numpy.where(self.resultArrayY == 0)
        oneIndics = numpy.where(self.resultArrayY == 1)

        zeroIndics = numpy.array(zeroIndics[0]).tolist()
        oneIndics = numpy.array(oneIndics[0]).tolist()

        firstPart = shapeX[3] / 2
        numpy.newaxis
        tempArrayX = numpy.concatenate((self.resultArrayX[zeroIndics[0], :, :, :firstPart], self.resultArrayX[zeroIndics[1], :, :, firstPart:]), axis = 2)
        tempArrayX = tempArrayX.reshape(1, tempArrayX.shape[0], tempArrayX.shape[1], tempArrayX.shape[2])
        tempArrayY = []
        tempArrayY.append(0)

        for i in xrange(1,len(zeroIndics) - 1):
            if (i + 1) % 6 != 0:#one hour
                temp = numpy.concatenate((self.resultArrayX[zeroIndics[i], :, :, :firstPart ], self.resultArrayX[zeroIndics[i + 1], :, :, firstPart:]), axis = 2)
                temp = temp.reshape(1, temp.shape[0], temp.shape[1], temp.shape[2])
                tempArrayX = numpy.concatenate((tempArrayX, temp), axis = 0)
                tempArrayY.append(0)
        for i in xrange(len(oneIndics) - 1):
            if (i + 1) % 6 != 0:
                temp = numpy.concatenate((self.resultArrayX[oneIndics[i], :, :, :firstPart], self.resultArrayX[oneIndics[i + 1], :, :, firstPart :]), axis = 2)
                temp = temp.reshape(1, temp.shape[0], temp.shape[1], temp.shape[2])
                tempArrayX = numpy.concatenate((tempArrayX, temp), axis = 0)
                tempArrayY.append(1)

        self.resultArrayX = numpy.concatenate((self.resultArrayX, tempArrayX),axis = 0)
        self.resultArrayY = numpy.concatenate((self.resultArrayY, numpy.asarray(tempArrayY)), axis = 0)

        if smote == True:
            zeros = numpy.where(self.resultArrayY == 0)
            ones = numpy.where(self.resultArrayY == 1)
            ratio = float(len(ones[0])) / float(len(zeros[0]))
            verbose = False
            ratio = 1 / ratio / 3
            smote = SMOTE(ratio = ratio, verbose=verbose, kind='regular')
            xshape = self.resultArrayX.shape
            smox, smoy = smote.fit_transform(self.resultArrayX.reshape((xshape[0],xshape[1] * xshape[2] * xshape[3])), self.resultArrayY)
            self.resultArrayX = smox.reshape((smox.shape[0],xshape[1], xshape[2], xshape[3]))
            self.resultArrayY = smoy

        if shuffle == True:
            resultArray = zip(self.resultArrayX, self.resultArrayY)
            random.shuffle(resultArray)
            tempX, tempY = zip(*resultArray)
            self.resultArrayX = numpy.array(tempX, dtype="float32")
            self.resultArrayY = numpy.array(tempY, dtype="int8")

        return self.resultArrayX, self.resultArrayY

    def shuffle(self,dataX1, dataX2, dataY):

        resultArray = zip(dataX1, dataX2, dataY)
        random.shuffle(resultArray)
        tempX1, tempX2, tempY = zip(*resultArray)
        tempX1 = numpy.array(tempX1, dtype="float32")
        tempX2 = numpy.array(tempX2, dtype="float32")
        tempY = numpy.array(tempY, dtype="int8")

        return tempX1, tempX2, tempY

    def sampling(self, dataX1, dataX2, dataY):
        verbose = False
        ratio = 2
        smote = SMOTE(ratio = ratio, verbose=verbose, kind='regular')
        #smote = UnderSampler(verbose = verbose)
        #smote = ClusterCentroids(verbose = verbose)
        xshape = dataX1.shape
        smox1, smoy1 = smote.fit_transform(dataX1.reshape((xshape[0],xshape[1] * xshape[2] * xshape[3])), dataY)
        dataX1 = smox1.reshape((smox1.shape[0],xshape[1], xshape[2], xshape[3]))

        xshape = dataX2.shape
        smox2, smoy2 = smote.fit_transform(dataX2.reshape((xshape[0],xshape[1] * xshape[2] * xshape[3])), dataY)
        dataX2 = smox2.reshape((smox2.shape[0],xshape[1], xshape[2], xshape[3]))

        if numpy.array_equal(smoy1, smoy2):
            dataY = smoy1
            zeros = numpy.where(smoy1 == 0)
            ones = numpy.where(smoy1 == 1)
            ratio = float(len(ones[0])) / float(len(zeros[0]))
            print ratio
        else:
            exit()

        return dataX1, dataX2, dataY

    def shuffle3(self,dataX1, dataX2, dataX3, dataY):
        resultArray = zip(dataX1, dataX2,dataX3, dataY)
        random.shuffle(resultArray)
        tempX1, tempX2, tempX3, tempY = zip(*resultArray)
        tempX1 = numpy.array(tempX1, dtype="float32")
        tempX2 = numpy.array(tempX2, dtype="float32")
        tempX3 = numpy.array(tempX3, dtype="float32")
        tempY = numpy.array(tempY, dtype="int8")

        return tempX1, tempX2, tempX3, tempY

    def overlap(self):
        shapeX = self.resultArrayX.shape
        shapeY = self.resultArrayY.shape
        firstPart = shapeX[2] / 2
        tempArrayX = numpy.concatenate((self.resultArrayX[0, :, :firstPart],self.resultArrayX[1, :, firstPart:]),axis = 1)
        tempArrayX = tempArrayX.reshape(1,tempArrayX.shape[0],tempArrayX.shape[1])
        tempArrayY = []
        if self.resultArrayY[0] == 1 or self.resultArrayY[1] == 1:
            tempArrayY.append(1)
        else:
            tempArrayY.append(0)

        for i in xrange(1,shapeX[0] - 1):
            temp = numpy.concatenate((self.resultArrayX[i, :, :firstPart ], self.resultArrayX[i + 1, :, firstPart:]), axis = 1)
            temp = temp.reshape(1,temp.shape[0],temp.shape[1])
            tempArrayX = numpy.concatenate((tempArrayX, temp), axis = 0)
            if self.resultArrayY[i] == 1 or self.resultArrayY[i + 1] == 1:
                tempArrayY.append(1)
            else:
                tempArrayY.append(0)

        self.resultArrayX = numpy.concatenate((self.resultArrayX, tempArrayX),axis = 0)
        self.resultArrayY = numpy.concatenate((self.resultArrayY, numpy.asarray(tempArrayY)), axis = 0)

        zeros = numpy.where(self.resultArrayY == 0)
        ones = numpy.where(self.resultArrayY == 1)
        ratio = float(len(ones[0])) / float(len(zeros[0]))
        verbose = False
        smote = SMOTE(ratio = 1 / ratio - 5 * ratio, verbose=verbose, kind='regular')

        xshape = self.resultArrayX.shape
        smox, smoy = smote.fit_transform(self.resultArrayX.reshape((xshape[0],xshape[1] * xshape[2] * xshape[3])), self.resultArrayY)
        self.resultArrayX = smox.reshape((smox.shape[0],xshape[1], xshape[2], xshape[3]))
        self.resultArrayY = smoy
        zeros = numpy.where(self.resultArrayY == 0)
        ones = numpy.where(self.resultArrayY == 1)
        ratio = float(len(ones[0])) / float(len(zeros[0]))

        resultArray = zip(self.resultArrayX, self.resultArrayY)
        random.shuffle(resultArray)
        tempX, tempY = zip(*resultArray)
        self.resultArrayX = numpy.array(tempX, dtype="float32")
        self.resultArrayY = numpy.array(tempY, dtype="int8")

        return self.resultArrayX ,self.resultArrayY

    def pca(self, dataArrayX, dataArrayY, bandNum=8, samplingRate=400, winLengthSec = 30, strideSec = 30):

        #dataArrayX's shape is matFileNumber * channels * matdata
        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1

        newArray = numpy.zeros((dataArrayX.shape[0], channels, channels, steps))
        pca = sklearn.decomposition.PCA()
        size = dataArrayX.shape[0]

        for i in xrange(size):
            for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                data = dataArrayX[i, :, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                pcaData = pca.fit_transform(data)
                pcaData = numpy.log10(numpy.absolute(pcaData))
                newArray[i, :, :, frameIndex] = pcaData

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY

        return self.resultArrayX, self.resultArrayY

    def ica(self, dataArrayX, dataArrayY, bandNum = 8, samplingRate=400, winLengthSec = 30, strideSec = 30):

        #dataArrayX's shape is matFileNumber * channels * matdata
        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1

        newArray = numpy.zeros((dataArrayX.shape[0], channels, 16, steps))
        ica  = sklearn.decomposition.FastICA(16)
        size = dataArrayX.shape[0]

        for i in xrange(size):
            for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                data = dataArrayX[i, :, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                icaData = ica.fit_transform(data)
                icaData = numpy.log10(numpy.absolute(icaData))
                #icaData = numpy.absolute(icaData)
                newArray[i, :, :, frameIndex] = icaData

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY
        #output shape is MatFileNumber * channels * pcaed data * stride

        return self.resultArrayX, self.resultArrayY

    def randomProjection(self, dataArrayX, dataArrayY, bandNum = 8, samplingRate=400, winLengthSec = 30, strideSec = 30):

        #dataArrayX's shape is matFileNumber * channels * matdata
        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1

        newArray = numpy.zeros((dataArrayX.shape[0], channels, channels, steps))
        projection = sklearn.random_projection.GaussianRandomProjection(channels)
        size = dataArrayX.shape[0]

        for i in xrange(size):
            for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                data = dataArrayX[i, :, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                icaData = projection.fit_transform(data)
                icaData = numpy.log10(numpy.absolute(icaData))
                #icaData = numpy.absolute(icaData)
                newArray[i, :, :, frameIndex] = icaData

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY

        return self.resultArrayX, self.resultArrayY

    def scaleAcrossFeature(self,dataTest=None,scalers=None):
        data = self.resultArrayX
        channelNum = data.shape[1]
        binNum = data.shape[2]
        timeStepNum = data.shape[3]
        flatten_dim = channelNum * binNum * timeStepNum
        data = data.reshape(data.shape[0], flatten_dim)

        if dataTest is not None:
            dataComplete = numpy.vstack((data, dataTest.reshape(dataTest.shape[0],flatten_dim)))
        else:
            dataComplete = data

        if scalers is None:
            scalers = sklearn.preprocessing.StandardScaler()
            scalers.fit(dataComplete)

        data = scalers.transform(data)
        data = data.reshape(data.shape[0],channelNum,binNum,timeStepNum)

        return data, scalers

    def scaleAcrossTime(self,dataTest = None, scalers = None):

        data = self.resultArrayX
        exampleNum = data.shape[0]
        channelNum = data.shape[1]
        binNum = data.shape[2]
        timeStepNum = data.shape[3]

        if scalers is None:
            scalers = [None] * channelNum

        for i in range(channelNum):
            dataI = numpy.transpose(data[:, i, :, :], axes=(0, 2, 1))
            dataI = dataI.reshape((exampleNum * timeStepNum, binNum))

            if dataTest is not None:
                dataITest = numpy.transpose(dataTest[:, i, :, :], axes = (0 ,2, 1))
                dataITest = dataITest.reshape(dataTest.shape[0] * timeStepNum, binNum)
                dataIComplete = numpy.vstack((dataI, dataITest))
            else:
                dataIComplete = dataI

            if scalers[i] is None:
                scalers[i] = sklearn.preprocessing.StandardScaler()
                scalers[i].fit(dataIComplete)

            dataI = scalers[i].transform(dataI)
            dataI = dataI.reshape((exampleNum, timeStepNum, binNum))
            dataI = numpy.transpose(dataI, axes=(0, 2, 1))
            data[:, i, :, :] = dataI

        return data, scalers

    # this is for comparsion from paper "Epileptic seizure prediction using
    # relative spectral power"
    def psd(self, dataArrayX, dataArrayY, bandNum=8, samplingRate=256, winLengthSec = 5, strideSec = 5):

        #dataArrayX's shape is matFileNumber * channels * matdata
        #bands = [0.5, 4, 8, 15, 30, 128]
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
        channels = dataArrayX.shape[1]
        dataLengthSec = dataArrayX.shape[2] / samplingRate
        steps = (dataLengthSec - winLengthSec) / strideSec + 1
        newArray = numpy.zeros((dataArrayX.shape[0], channels, bandNum, steps))
        size = dataArrayX.shape[0]
        for i in xrange(size):
            for j in xrange(channels):
                for frameIndex, windowIndex in enumerate(range(0, dataLengthSec - winLengthSec + 1, strideSec)):
                    data = dataArrayX[i, j, windowIndex * samplingRate:(windowIndex + winLengthSec) * samplingRate]
                    freq, power = signal.welch(data, fs=samplingRate)
                    power = numpy.log10(numpy.abs(power))

                    frequencyBands = numpy.digitize(freq, bands)
                    dataFrame = pandas.DataFrame({"fft": power, "band": frequencyBands})

                    dataFrame = dataFrame.groupby("band").sum()
                    result = dataFrame.fft[1: -1]
                    newArray[i, j, :, frameIndex] = result

                    #newArray[i, j, -1, frameIndex] = numpy.std(data)

        #normalize
        shape = newArray.shape
        for i in range(shape[0]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    newArray[i, :, j, k] = newArray[i, :, j, k] / newArray[i, :, j ,k].sum()

        self.resultArrayX = newArray
        self.resultArrayY = dataArrayY
        return self.resultArrayX, self.resultArrayY

    # Relative spectral power, from paper "Epileptic seizure prediction using
    # relative spectral power features"
    def rsp(self, dataArrayX, dataArrayY, bandNum=5, samplingRate=256, winLengthSec = 5, strideSec = 5):
        x, y = self.psd(dataArrayX, dataArrayY, bandNum = bandNum, samplingRate=samplingRate, winLengthSec = winLengthSec, strideSec = strideSec)
        shape = x.shape
        x = x.reshape(shape[0], shape[1] * shape[2], shape[3])
        newArray = numpy.zeros((shape[0], shape[1] * shape[2] * (shape[1] *shape[2]  -1) / 2, shape[3]))
        comlist = []
        for i in xrange(shape[0]):
            for j in xrange(shape[3]):
                combinations = itertools.combinations(x[i ,: ,j],2)
                for k in combinations:
                    comlist.append(k[0] / k[1])
                newArray[i,:,j] = numpy.array(comlist)
                comlist = []

        self.resultArrayX = newArray
        self.resultArrayY = y
        return self.resultArrayX, self.resultArrayY
    #this is for paper "Epileptic seizure prediction using relative spectral power features"
    def movingAverage(self, dataArrayX, dataArrayY, bandNum=5, samplingRate=256, winLengthSec = 5, strideSec = 5):
        x, y = self.rsp(dataArrayX, dataArrayY, bandNum = bandNum, samplingRate=samplingRate, winLengthSec = winLengthSec, strideSec = strideSec)
        shape = x.shape

        n = 12
        newArray = numpy.zeros((shape[0], shape[1] - n + 1, shape[2]))
        for i in xrange(shape[0]):
            for j in xrange(shape[2]):
                ret = numpy.cumsum(x[i, :, j], dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                average = ret[n-1:] / n
                normAverage = (average - average.min()) / (average.max() - average.min())
                newArray[i,:,j] = normAverage

        self.resultArrayX = newArray
        self.resultArrayY = y

        return self.resultArrayX, self.resultArrayY

    #this is for paper "Epileptic seizure prediction using relative spectral power features"
    def featureSelection(self, dataArrayX, dataArrayY, bandNum=5, samplingRate=256, winLengthSec = 5, strideSec = 5, isEnd = False):

        x, y = self.movingAverage(dataArrayX, dataArrayY, bandNum = bandNum, samplingRate=samplingRate, winLengthSec = winLengthSec, strideSec = strideSec)

        shape = x.shape
        x = x.reshape(shape[0] * shape[2], shape[1])
        for i in xrange(shape[0]):
            if self.classZeroHistogram is None and y[i] == 0:
                self.classZeroHistogram = numpy.histogram(x[i, :], 100, range=(0.0, 1.0))[0]
            elif self.classOneHistogram is None and y[i] == 1:
                self.classOneHistogram = numpy.histogram(x[i, :], 100, range=(0.0, 1.0))[0]
            elif y[i] == 0:
                self.classZeroHistogram = self.classZeroHistogram + numpy.histogram(x[i, :], 100, range=(0.0, 1.0))[0]
            elif y[i] == 1:
                self.classOneHistogram = self.classOneHistogram + numpy.histogram(x[i, :], 100, range=(0.0, 1.0))[0]

        if isEnd == True:
            if self.classZeroHistogram is not  None:
                classZeroSum = self.classZeroHistogram.sum()
                self.classZeroHistogram = self.classZeroHistogram / float(classZeroSum * 0.01)
            if self.classOneHistogram is not None:
                classOneSum = self.classOneHistogram.sum()
                self.classOneHistogram = self.classOneHistogram / float(classOneSum * 0.01)

        return x, y

