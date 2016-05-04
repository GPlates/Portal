import matplotlib
matplotlib.use('Agg')
import logging, pickle
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as stats
from revrand_map import classification, basis_functions
# NOTE: uses commit b5f23b7eac6e84e22a9d223ad9556630b618268d from revrand

# Helper function
def normaliseInputs(trainingX):
    Xmean = np.mean(trainingX, axis=0)
    X_norm = trainingX - Xmean
    Xmin = np.min(X_norm, axis=0); Xmax = np.max(X_norm, axis=0)
    Xrange = Xmax-Xmin
    X_norm = X_norm/Xrange
    return X_norm, Xmean, Xrange


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# A la Carte classifier setting
nbases = 2500
lenscale = 0.1
reg = 1000.

# method = 'SGD'
method = 'SGD'
batchsize = 100
rate = 0.9
eta = 1e-6
passes = 10

# Pre-learnt lengthscales
xlen = 0.01333521
ylen = 0.03162278
lenscales = (xlen,ylen)


def trainAndSaveModel():
    #collate and normalise data
    print('Training model...please wait')
    data_path = '/notebooks/'
    data = np.load(data_path + 'seafloorData_final.npz')
    lat = data['lat']
    lon = data['lon']
    labels = data['labels'].ravel().astype(int)
    ndata = len(labels)


    #Pad the boundaries
    lonPos170180Ind = (lon>170)*(lon<180)
    lonNeg170180Ind = (lon<-170)*(lon>-180)

    padPosLon = lon[lonNeg170180Ind]+360
    padPosLat = lat[lonNeg170180Ind]
    padPosLabel = labels[lonNeg170180Ind[:,0]]

    padNegLon = lon[lonPos170180Ind]-360
    padNegLat = lat[lonPos170180Ind]
    padNegLabel = labels[lonPos170180Ind[:,0]]

    lon = np.append(np.append(lon,padPosLon), padNegLon)
    lat = np.append(np.append(lat, padPosLat), padNegLat)
    labels = np.append(np.append(labels, padPosLabel), padNegLabel)

    trainingPoints = np.array([lon,lat]).T

    # Create test data
    xeva, yeva = np.meshgrid(np.linspace(-180,180,361),np.linspace(-90,90,181))

    xeva = xeva.T
    yeva = yeva.T
    testPoints = np.c_[xeva.ravel(), yeva.ravel()]

    X, Xmean, Xrange = normaliseInputs(trainingPoints)
    Xs = (testPoints - Xmean)/Xrange
    Y = labels

    # Plot scatter
    Phi = basis_functions.RandomRBF_ARD(nbases, X.shape[1])
    #pl.figure()

    # learn weights
    weights, labels = classification.learn_sgd(X, Y, Phi, lenscales,
                               regulariser=reg, eta=eta, batchsize=batchsize,
                               rate=rate, passes=passes)
    

    with open("Phi.pkl",'wb') as file:
        pickle.dump(Phi, file)
    np.savez('data',weights=weights, Xmean=Xmean, Xrange=Xrange)
    print('Model has been trained and saved')
    
def loadModelAndPredict(x,y):
    xeva = x.T
    yeva = y.T
    
    testPoints = np.c_[xeva.ravel(), yeva.ravel()]
    data = np.load('data.npz')
    Xs = (testPoints - data['Xmean'])/data['Xrange']
    Phi = pickle.load(open('Phi.pkl', 'rb'))
    return classification.predict(Xs, data['weights'], Phi, lenscales)
