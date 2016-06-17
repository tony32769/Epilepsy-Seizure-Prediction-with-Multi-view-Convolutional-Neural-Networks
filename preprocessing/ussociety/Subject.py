from MatFile import *
import os
import glob

class Subject:
    def __init__(self,name=""):
        self.name = name
        self.matTrainFileList = []
        self.matTestFileList = []

    def getTrainFileList(self,name=""):
        self.matTrainFileList = []
        if self.name == "":
            self.name = name
        #trainFiles = os.listdir(self.name)
        trainFiles = glob.glob(self.name + "/*.mat")
        trainFiles = sorted(trainFiles)
        for trainFile in trainFiles:
            if "test" not in trainFile:
                #matFile = MatFile()
                #matFile.readMat(self.name + "/" + trainFile)
                #self.matTrainFileList.append(self.name + "/" + trainFile)
                self.matTrainFileList.append(trainFile)

        return self.matTrainFileList

    def getTestFileList(self,name=""):
        self.getTestFileList = []
        if self.name == "":
            self.name = name
        #testFiles = os.listdir(self.name)
        testFiles = glob.glob(self.name + "/*.mat")
        testFiles = sorted(testFiles)
        for testFile in testFiles:
            if "test" in testFile:
                #matFile = MatFile()
                #matFile.readMat(self.name + "/" + testFile)
                #self.matTestFileList.append(self.name + "/" + testFile)
                self.matTestFileList.append(testFile)

        return self.matTestFileList
