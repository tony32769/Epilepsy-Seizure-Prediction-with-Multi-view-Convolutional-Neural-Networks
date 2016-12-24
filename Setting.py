import yaml
import os
class Setting:
    def __init__(self, path = "settings.yml"):
        #data process related
        self.splitNum = 10
        self.timeSlotLength = 60 # in seconds
        self.timeSlotNum = 600 / self.timeSlotLength
        self.rawDataPath = None
        self.processedDataPath = None
        self.savePath = None
        self.resampleFrequency = None

        #cnn model related
        self.name = None
        self.nb_filter = None
        self.nb_epoch = None
        self.batch_size = None
        self.output1 = None
        self.output2 = None
        self.lr = None
        self.dropout = None
        self.l2 = None

        if path != "":
            self.path = path
        #    if os.path.isfile(path):
        #        self.loadSettings(self.path)

    def loadSettings(self, name = "Dog_1", path = ""):
        with open(self.path, "r") as yfile:
            yfile =  yfile.read()
            fileList = yfile.split("################################################")
            for tmpfile in fileList:
                yamlFile = yaml.load(tmpfile)
                if yamlFile["Setting"]["name"] == name:
                    self.name = yamlFile["Setting"]["name"]
                    self.resampleFrequency = yamlFile["Setting"]["DataProcess"]["resampleFrequency"]
                    self.rawDataPath = yamlFile["Setting"]["DataProcess"]["rawDataPath"]
                    self.processedDataPath = yamlFile["Setting"]["DataProcess"]["processedDataPath"]
                    self.splitNum = yamlFile["Setting"]["DataProcess"]["splitNum"]
                    self.timeSlotLength = yamlFile["Setting"]["DataProcess"]["timeSlotLength"]
                    self.timeSlotNum = yamlFile["Setting"]["DataProcess"]["timeSlotNum"]
                    self.nb_filter = yamlFile["Setting"]["Model"]["nb_filter"]
                    self.nb_epoch = yamlFile["Setting"]["Model"]["nb_epoch"]
                    self.batch_size = yamlFile["Setting"]["Model"]["batch_size"]
                    self.output1 = yamlFile["Setting"]["Model"]["output1"]
                    self.output2 = yamlFile["Setting"]["Model"]["output2"]
                    self.lr = yamlFile["Setting"]["Model"]["lr"]
                    self.dropout = yamlFile["Setting"]["Model"]["dropout"]
                    self.l2 = yamlFile["Setting"]["Model"]["l2"]
                    self.savePath = yamlFile["Setting"]["DataProcess"]["savePath"]

            return self
    def saveSettings(self, path = ""):
        data = dict(Setting = dict(name = self.name,
                                   DataProcess = dict(rawDataPath = self.rawDataPath,
                                                      processedDataPath = self.processedDataPath,
                                                      splitNum = self.splitNum,
                                                      timeSlotLength = self.timeSlotLength,
                                                      timeSlotNum = self.timeSlotNum,
                                                      savePath = self.savePath
                                                      ),
                                   Model = dict(nb_filter = self.nb_filter,
                                                nb_epoch = self.nb_epoch,
                                                batch_size = self.batch_size,
                                                output1 = self.output1,
                                                output2 = self.output2,
                                                lr = self.lr,
                                                dropout = self.dropout,
                                                l2 = self.l2)
                                   ))
        if path != "":
            self.path = path
        with open("settings.yml", "a") as outfile:
            outfile.write(yaml.dump(data, default_flow_style=False))
