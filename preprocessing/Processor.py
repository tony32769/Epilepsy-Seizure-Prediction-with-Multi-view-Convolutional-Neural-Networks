from ussociety.MatFile import *
from ussociety.Subject import *
from SignalUtils import *
from unbalanced_dataset import SMOTE
from Setting import *
import glob

class Processor:
    def __init__(self):
        self.setting = Setting()
        self.matFileInstance = MatFile()
        self.subjectInstance = Subject()
        self.signalUtilInstance = SignalUtils()
        self.trainDir = "/home/xiaobin/Disk/trainData"
        #self.testDir = "/home/xiaobin/Disk/testData"
        self.subjectName = ""

    def processDataPerMatFile(self,dataFile,trainOrTest="train"):
        self.matFileInstance.readMat(dataFile)
        self.matFileInstance.name = dataFile
        self.matFileInstance.samplingRate = self.setting.resampleFrequency
        self.matFileInstance.data = self.signalUtilInstance.resample(self.matFileInstance.data, self.setting.resampleFrequency)
        self.matFileInstance.data = self.signalUtilInstance.butterWorthBandpassFilter(self.matFileInstance.data, band=[0.1, 180], frequency = 400)
        #this is for comparison solution
        self.matFileInstance.data = self.signalUtilInstance.butterWorthBandpassFilter(self.matFileInstance.data, band=[0.5, 128], frequency = 256)

        matPerSlot = self.matFileInstance.getDataListPerTimeSlot(timeslot = self.matFileInstance.timeLength)
        size = matPerSlot.shape
        size = size[0]

        #self.matFileInstance.data = self.signalUtilInstance.resample(self.matFileInstance.data, 400)
        #self.matFileInstance.data = self.signalUtilInstance.butterWorthBandpassFilter(matPerSlot[0].data, band=[0.1, 180], frequency = 400)

        #fft
        #matFileInstance.data = util.fft(matFileInstance.data)
        #self.matFileInstance.data = self.signalUtilInstance.daubWavelet(self.matFileInstance.data)

        #self.matFileInstance.data = self.signalUtilInstance.resample(self.matFileInstance.data, 400)
        sizex,sizey = matPerSlot[0].data.shape

        matBagX = numpy.zeros((size,sizex,sizey))
        matBagY =  []

        matBagX[0, :, :] = matPerSlot[0].data

        if trainOrTest == "train":
            matBagY.append(self.matFileInstance.getSeizureOnsetLabel())
        elif trainOrTest == "test":
            matBagY.append(self.matFileInstance.name.split("/")[-1])

        for i in xrange(1,size):
            #self.matFileInstance.data = self.signalUtilInstance.resample(self.matFileInstance.data, 400)
            #self.matFileInstance.data = self.signalUtilInstance.butterWorthBandpassFilter(matPerSlot[i].data, band=[0.1, 180], frequency = 400)
            #self.matFileInstance.data = self.signalUtilInstance.daubWavelet(self.matFileInstance.data)
            matBagX[i, :, :] = matPerSlot[i].data
            if trainOrTest == "train":
                #matBagY[i, :, :] = self.matFileInstance.getSeizureOnsetLabel()
                matBagY.append(self.matFileInstance.getSeizureOnsetLabel())
            elif trainOrTest == "test":
                matBagY.append(self.matFileInstance.name.split("/")[-1])

        return matBagX, matBagY

    def processDataPerSubject(self,subject, splitNum = 10, sequence = 0, trainOrTest = "train"):
        dataList = None
        if trainOrTest == "train":
            dataList = self.subjectInstance.getTrainFileList(subject)
        elif trainOrTest == "test":
            dataList = self.subjectInstance.getTestFileList(subject)
        else:
            raise "trainOrTest error"

        if subject.find("Dog") != -1:
            index =  subject.find("Dog")
            self.subjectName =  subject[index : index + len("Dog") + 2]
        else:
            index = subject.find("Patient")
            self.subjectName = subject[index : index + len("Patient") + 2 ]
        amount = None
        if len(dataList) % splitNum == 0:
            amount = len(dataList) / splitNum
        else:
            amount = len(dataList) / splitNum + 1
        for i in xrange(splitNum):
            if i == sequence:
                return self.processData(dataList[i * amount: (i + 1)*amount],sequence = sequence, trainOrTest = trainOrTest)

    def processData(self,trainList,sequence = 0,trainOrTest = "train"):

        dim0  = len(trainList)
        if dim0 == 0:
            raise "trainList is empty"
        X,Y = self.processDataPerMatFile(trainList[0],trainOrTest = trainOrTest)
        xdim0,xdim1,xdim2 = X.shape
        dim0 = dim0 * xdim0

        trainDataX = numpy.zeros((dim0, xdim1, xdim2))
        trainDataY = []
        trainDataX[0:xdim0,:,:] = X
        trainDataY += Y
        #if trainOrTest == "train":
        #    #trainDataY = numpy.zeros_like(trainDataX)
        #    trainDataX[0:xdim0,:,:] = X
        #    trainDataY[0:xdim0,:,:] = Y
        #elif trainOrTest == "test":
        #    trainDataY = []
        #    trainDataX[0:xdim0,:,:] = X
        #    trainDataY += Y
        for i in xrange(1,len(trainList)):
            tempX, tempY = self.processDataPerMatFile(trainList[i],trainOrTest = trainOrTest)
            trainDataX[i * xdim0: (i + 1) * xdim0, :, :] = tempX
            #if trainOrTest == "train":
            #    trainDataY[i * xdim0: (i + 1) * xdim0,:,:] = tempY
            #elif trainOrTest == "test":
            trainDataY += tempY
        #self.saveDataToDisk(trainDataX, trainDataY, sequence, trainOrTest)

        return trainDataX,trainDataY

    def saveDataToDisk(self,trainDataX,trainDataY,sequence = 0, trainOrTest = "train"):

        if trainOrTest == "train":
            numpy.save(self.trainDir + "/" + self.subjectName + "/trainX_" + str(sequence), trainDataX)
            numpy.save(self.trainDir + "/" + self.subjectName + "/trainY_" + str(sequence), trainDataY)
        elif trainOrTest == "test":
            numpy.save(self.trainDir + "/" + self.subjectName + "/testX_" + str(sequence), trainDataX)
            numpy.save(self.trainDir + "/" + self.subjectName + "/testY_" + str(sequence), numpy.array(trainDataY))
        else:
            raise "save error : train or test"
        #scipy.io.savemat(self.trainDir + "/trainX_" + timeStr,{"data":trainDataX})
        #scipy.io.savemat(self.trainDir + "/trainY_" + timeStr,{"data":trainDataY})
        #del trainDataX
        #del trainDataY

    def loadDataFromDisk(self,sequence = 0, trainOrTest = "train"):
        trainX = None
        trainY = None
        #files = os.listdir(self.trainDir + "/" + self.subjectName)
        files =  glob.glob(self.trainDir + "/" + self.subjectName + "/*.npy")
        files = sorted(files)
        count = 0
        for f in files:
            if trainOrTest == "train":
                if "trainX" in f and count == sequence:
                    #trainX = numpy.load(self.trainDir + "/" + self.subjectName + "/" + f)
                    trainX = numpy.load(f)
                    #trainX = scipy.io.loadmat(f)["data"]
                    f = f.replace("trainX","trainY")
                    #trainY = numpy.load(self.trainDir + "/" + self.subjectName + "/" + f)
                    trainY = numpy.load(f)
                    #trainY = scipy.io.loadmat(f)["data"]
                    count += count
                    return trainX, trainY
            elif trainOrTest == "test":
                if "testX" in f and count == sequence:
                    testX = numpy.load(f)
                    f = f.replace("testX","testY")
                    testY = numpy.load(f)
                    count += count
                    return testX, testY

        return None
    def rebalanceData(self,x, y,mode="SMOTE", trainOrTest = "train", rebalanceInstance = None):
        if mode == "SMOTE":
            if trainOrTest == "train":
                verbose = False
                ratio = float(numpy.count_nonzero( y == 1)) / float(numpy.count_nonzero(y == 0))
                smoteInstance = SMOTE(ratio = ratio, verbose = verbose,kind = "regular")
                smoteDataX, smoteDataY = smote.fit_transform(x, y)

                return smoteInstance, smoteDataX, smoteDataY
            elif trainOrTest == "test":
                if rebalanceInstance is None:
                    raise "rebalanceInstance can not be none when the data is for testing"
                else:
                    smoteDataX = smoteInstance.transform(x)
                    return smoteDataX

