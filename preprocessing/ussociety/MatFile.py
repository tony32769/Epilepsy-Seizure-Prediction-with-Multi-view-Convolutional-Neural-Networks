import numpy
import scipy.io
import gc

class MatFile:
    def __init__(self,name=""):
        self.subject = ""
        self.name = name
        self.data = None
        self.channels = []
        self.samplingRate = 400
        #time in seconds
        self.timeLength = 0
        self.sequence = 0
        self.matType = ""

    def readMat(self,fileName):
        if self.name == "" or fileName != "":
            self.name = fileName

        index = self.name.find("Dog")
        if index == -1:
            index == self.name.find("Patient")
            self.subject = self.name[index:index + len("Patient") + 2]
        else:
            self.subject = self.name[index:index + len("Dog") + 2]
        #rawData = scipy.io.loadmat(fileName)

        rawData = scipy.io.loadmat(fileName,squeeze_me = True)

        print fileName
        if "test" in fileName:
            self.matType = "test"
        elif "preictal" in fileName:
            self.matType = "preictal"
        else:
            self.matType = "interictal"
        for key in rawData.keys():
            if key != "__version__" and key != "__globals__" and key != "__header__":
                rawData = rawData[key]
                break
        rawData = rawData.tolist()
        self.data = numpy.array(rawData[0],dtype="float32")
        #channels need to be tested
        self.timeLength = int(rawData[1])
        self.samplingRate = int(rawData[2])
        self.channels = rawData[3]
        if self.matType != "test":
            self.sequence = int(rawData[4])

    def getSeizureOnsetLabel(self):
        if self.matType == "preictal":
            return 1
        elif self.matType == "interictal":
            return 0
        else:
            return None
    def getDataListPerTimeSlot(self,timeslot = 60):
        if self.timeLength == timeslot:
            return numpy.array([self],dtype=numpy.object)
        else:
            #matList = []
            matList = numpy.ndarray((self.timeLength / timeslot, ), dtype=numpy.object)
            for i in range(self.timeLength / timeslot):
                mat = MatFile()
                mat.channels = self.channels
                mat.samplingRate = self.samplingRate
                mat.data = self.data[:, timeslot * i * self.samplingRate: timeslot * (i+1) * self.samplingRate]
                mat.matType = self.matType
                mat.name = self.name +"_" +str(i)
                mat.sequence = self.sequence
                mat.subject = self.subject
                mat.timeLength = self.timeLength
                matList[i] = mat

            return matList

