import numpy as np
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from preprocessing.Processor import *
from preprocessing.Feature import *
from tsne import bh_sne

#data = OfficialImageClassification(x_dtype="float32")
#x_data = data.all_images
#y_data = data.all_labels

feature = Feature("Dog_5")
x_data, y_data = feature.loadFromDisk("train","PCA")

x_data = np.asanyarray(x_data).astype("float64")
print x_data.shape
print y_data.shape
x_data = x_data.reshape((x_data.shape[0], -1))
print x_data.shape

#n = 200
#x_data = x_data[:n]
#y_data = y_data[:n]
#print x_data.shape
#print y_data.shape

vis_data = bh_sne(x_data,perplexity=15)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
cm = plt.cm.get_cmap("cool")
#plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet"))
plt.scatter(vis_x, vis_y, c=y_data, cmap=cm)
plt.colorbar(ticks=range(2))
#plt.clim(-0.5, 9.5)
plt.show()
