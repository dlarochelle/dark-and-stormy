
import requests
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib as mp
import scipy as sp
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import *

from scipy import interpolate
import csv


# File input
link = 'http://cluster-7-slave-06.sl.hackreduce.net:8098/buckets/dark-stormy-stock-state/keys?keys=true'
r = requests.get(link)
ks =r.json().values()
bucket = 'http://cluster-7-slave-06.sl.hackreduce.net:8098/buckets/dark-stormy-stock-state/keys/'

def bucket_address(bucket, ks):
    bucket_addy = []
    dataDict={}
    for strA in ks:
        bucketVal= bucket + strA
        theData=requests.get(bucketVal)
        # returns a dict with KEY and VALUE
        dataDict[strA]=theData.content        
    return dataDict

# Visualization
def update_line(num, data, line):
    line.set_data(data[...,:num])
    return line,


def test_pic():
    import numpy as np
    import matplotlib.pyplot as plt

    fig1 = plt.figure()

    data = np.random.rand(2, 25)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),interval=50, blit=True)
    #line_ani.save('lines.mp4')

    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

        im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,blit=True)
        #im_ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()

def stock_data(myDict):
    # 50 numbers between -5 sigma and 5 sigma
    myVals=myDict.values()
    x = np.array(myVals).astype(np.float)
    # Standard normal PDF
    hist, bin_edges = np.histogram(x,bins=50)
    y = stats.norm.pdf(hist)
    # Logistic PDF
    yb = stats.logistic.pdf(x)
    
    # Plot logistic PDF
    plt.plot(np.arange(len(yb)),yb, color="blue", label="logistic PDF")
    # Plot Standard normal PDF
    #plt.plot(x,y, color="red", label="standard normal PDF")
    # Label for x axis
    plt.xlabel("z")
    # Label for title of graph
   # plt.title("PDF for Gaussian of mean = {0} & std. deviation = {1}".format(
   #            mean, std))
    # Legend
   # legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    # This launches the graph when function is called
    plt.draw() # changed from plt.draw() to plt.show()
    plt.savefig('test.png')
def rand_PDF_dist(mean, std):
    # 50 numbers between -5 sigma and 5 sigma
    x = sp.linspace(-5*std, 5*std, 50)
    # Standard normal PDF
    y = stats.norm.pdf(x, loc=mean, scale=std)
    # Logistic PDF
    yb = stats.logistic.pdf(x, loc=mean, scale=std)
    
    # Plot logistic PDF
    plt.plot(x,yb, color="blue", label="logistic PDF")
    # Plot Standard normal PDF
    plt.plot(x,y, color="red", label="standard normal PDF")
    # Label for x axis
    plt.xlabel("z")
    # Label for title of graph
    plt.title("PDF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    # Legend
    legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    # This launches the graph when function is called
    plt.draw() # changed from plt.draw() to plt.show()
    plt.savefig('test.png')

def sort_stock_data():
    pass

if __name__ == "__main__":
    test = bucket_address(bucket, ks[0][0:100])
   # print test
    stock_data(test)
    #rand_PDF_dist(mean=0.0, std=1.0)

    




