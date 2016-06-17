import scipy.signal
import sklearn
import numpy
import pywt

class SignalUtils:
    def __init__(self, name=""):
        self.name = name

    def resample(self, data, newSamplingRate, timeLength = 600):

        return scipy.signal.resample(data,newSamplingRate * timeLength, axis = 1)

    def butterWorthBandpassFilter(self,data, order=5, band = [1, 47], frequency = 400):

        b, a  = scipy.signal.butter(order, numpy.array(band) / (frequency/2.0), btype="band")

        return scipy.signal.lfilter(b, a, data, axis=1)

    def fft(self, data):
        axis = data.ndim - 1

        return numpy.fft.rfft(data, axis = axis)

    #def firLowpassFilter(self,data,frequency=256):
    #    pass

    def magnitude(self, data):
        return numpy.absolute(data)

    def log10(self, data):

        index = numpy.where(data < 0)
        data[index] = numpy.max(data)
        data[index] = numpy.min(data) * 0.1

        return numpy.log10(data)

    # more functions on wavelet refering
    # http://www.pybytes.com/pywavelets/ref/dwt-discrete-wavelet-transform.html
    def daubWavelet(self, data, mode = 1):
        shape = data.shape
        wavelet = numpy.empty((shape[0], 4 * (mode * 2 + 1)), dtype=numpy.float64)
        for i in range(len(data)):
            waveleti = wavelet[i]
            newData = pywt.wavedec(data[i], "db%d" % mode, level = mode * 2)
            for offset, x in enumerate(newData):
                waveleti[offset * 4] = numpy.mean(x)
                waveleti[offset * 4 + 1] = numpy.std(x)
                waveleti[offset * 4 + 2] = numpy.min(x)
                waveleti[offset * 4 + 3] = numpy.max(x)

        return wavelet

    #def scale(self, data):

    #    return sklearn.preprocessing.scale(data, axis = 0)

    def correlationMatrix(self,data):

        return numpy.corrcoef(data)

    def eigenvalues(self,data):
        w, v = numpy.linalg.eig(data)
        w = numpy.absolute(w)
        w.sort()

        return w

    def upperRightTriangle(self, matrix):
        accum = []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                accum.append(matrix[i, j])

        return numpy.array(acuum)

    #def frequencyCorrelation(self, data, start, end, withFFT = False, withCorrelation = True, withEigen = True):


