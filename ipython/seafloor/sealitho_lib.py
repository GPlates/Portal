
# coding: utf-8

# In[5]:

# Python multivariate analysis (Support Vector Classification) script
# written by Simon O'Callaghan NICTA
# 14 May 2015

import matplotlib
matplotlib.use('Agg')
rootPath='/notebooks/'
import sys
sys.path.append(rootPath+'bdkd-external-devel/rocks/seafloorLitho/')
sys.path.append(rootPath+'revrand-master/')
sys.path.append(rootPath+'bdkd-external-devel')
sys.path.append("/usr/local/lib/python3.4/site-packages/")

import os
import pandas as pd
import numpy as np
import utils
from sklearn.svm import SVC
from sklearn import cross_validation
from matplotlib import pyplot as pl
get_ipython().magic('matplotlib inline')
from multiprocessing import Pool
import multiprocessing
import scipy.linalg as linalg

def loadData(path):
    data = pd.read_csv(path)
    for i in range(len(data.columns)):
      data  = data[np.isfinite(data[data.columns[i]])]  # Remove any nans

    # Set up the features (X) and the targets or labels (y)
    X = data[data.columns[3:]].values
    y = data.lithology.values
    lonlat = data[data.columns[:2]].values
    return X-np.min(X,axis=0),y, lonlat


def preprocessData(X):
    X[:, 4] = np.log(X[:, 4]+1e-20)
    X[:, 6] = np.log(X[:, 6]+1e-20)
    return X

def normaliseData(X):
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0)
    X = (X - means)/std_devs
    return X

def printMostUsefulFeatures(X, y, nRandomFeatures, featOnListArray, resultsList,maxSampPerClass, lithoLabel):
    results = np.asarray(resultsList).T
    nFeatures = X.shape[1]
    nTrueFeatures = nFeatures - nRandomFeatures
    classifierScoreList=[]

    XCull, yCull = utils.cullData(X,y,maxSampPerClass)
    featureAccuracy = np.zeros((nTrueFeatures,results.shape[1]))
    for nFeaturesIter in (np.arange(nTrueFeatures)+1):
        classifierScore = np.mean(results[np.sum(featOnListArray, axis=1)==nFeaturesIter], axis=1)
        classifierScoreList.append(np.max(classifierScore).astype(float))
        activeFeatures = featOnListArray[np.sum(featOnListArray,axis=1)
                                             ==nFeaturesIter,:]
        bestFeatures = np.arange(nFeatures)[activeFeatures[np.argmax(classifierScore)]==1]+1
        bestFeatureID = np.argmax(classifierScore)
        featureAccuracy[(nFeaturesIter-1),:] = results[np.sum(featOnListArray, axis=1)==nFeaturesIter][bestFeatureID,:]
        print('For a classifier using %d features,' %(nFeaturesIter))
        print('the most informative features are:')
        print(bestFeatures.astype(list), 'with a score of %f' %(np.max(classifierScore)))
        scoreList = []
        for iter in range(results.shape[1]):
            Xrand = np.random.randn(XCull.shape[0], nFeaturesIter)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
              Xrand, yCull, test_size=0.2)
            svc = SVC(kernel='rbf', probability=True)
            svc.fit(X_train,y_train)
            scoreList.append(svc.score(X_test, y_test))

        print('%d random features have a score of: %f' %(nFeaturesIter, np.mean(scoreList)))
        print()

    pl.figure()
    pl.errorbar((np.arange(nTrueFeatures)+1).astype(int), np.mean(featureAccuracy,axis=1), np.std(featureAccuracy,axis=1))
    pl.title('Area Under ROC Curve vs Number of Features')
    pl.xlabel('Number of features')
    pl.ylabel('AUC')
    xint = range(0,nTrueFeatures+2)
    pl.xticks(xint)
    ax = pl.gca()
    ax.set_xlim((0,nTrueFeatures+1))
    pl.savefig('auc'+str(lithoLabel)+'.pdf')



def genFeatureFreq(nFeatures, nRandomFeatures, results,featOnArray):
    nTrueFeatures = nFeatures - nRandomFeatures
    featureFreq = np.zeros([nTrueFeatures, nFeatures])

    for scoreList in results:
        maxFeatSortedID = np.argsort(scoreList)[::-1]
        maxFeatSorted = featOnArray[maxFeatSortedID, :]

        for nFeaturesIter in (np.arange(nTrueFeatures)+1):
              indexBoole = np.sum(maxFeatSorted,axis=1)==nFeaturesIter
              featureFreq[nFeaturesIter-1,:] = featureFreq[nFeaturesIter-1,:] +                                                maxFeatSorted[indexBoole,:][0,:]

    return featureFreq

def genAUCMatrix(nFeatures, nRandomFeatures, results,featOnArray):

    results = np.mean(results,axis=0)

    nTrueFeatures = nFeatures - nRandomFeatures
    featureFreq = np.zeros([nTrueFeatures, nFeatures])

    for i, activeFeats in enumerate(featOnArray[:-1,:]):
        for j in np.arange(np.sum(activeFeats)):
            col_id = np.where(activeFeats==1.)[0]
            featureFreq[np.sum(activeFeats)-1,col_id[j]] = results[i]

    return featureFreq


def plotMatrix(featureFreq,lithoLabel):
    nTrueFeatures = featureFreq.shape[0]
    nFeatures = featureFreq.shape[1]
    pl.figure()
    pl.imshow(featureFreq, interpolation='nearest', extent=(0.5, nFeatures + 0.5, nTrueFeatures + 0.5, 0.5))
    pl.xlabel('Feature ID')
    pl.ylabel('Number of Features')
    pl.colorbar()
    ax = pl.gca()
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    pl.savefig('balancedmultvarAnalysisLabel'+str(lithoLabel)+'Seafloor.pdf')


def removeCorrelatedFeatures(X):
    keepdims = [0,6,4,5,7, -1]
    X = X[:,keepdims]

    return X

def removeHemisphere(X, y_map, lonlat):
    keeppoints_id= np.where(lonlat[:,1]<0)[0]

    return X[keeppoints_id,:], y_map[keeppoints_id]

def cullData(X,y,maxSampPerClass):

    XCull = X.copy()
    yCull = y.copy()

    for i in [0, 1]:
        if np.sum(y == i)>maxSampPerClass:
              index = np.arange(yCull.shape[0])[yCull==i]
              np.random.shuffle(index)
              XCull = np.delete(XCull, index[maxSampPerClass:], axis=0)
              yCull = np.delete(yCull, index[maxSampPerClass:])
    return XCull, yCull

def balanceData(X,y):
    '''
    assumes spacing is uniform between labels
    '''

    xBal = X.copy()
    yBal = y.copy()

    nPerGroup, bins = np.histogram(yBal, np.unique(yBal).shape[0])
    maxSampPerClass = np.min(nPerGroup)

    for i in np.unique(yBal):
        if np.sum(y == i)>maxSampPerClass:
              index = np.arange(yBal.shape[0])[yBal==i]
              np.random.shuffle(index)
              xBal = np.delete(xBal, index[maxSampPerClass:], axis=0)
              yBal = np.delete(yBal, index[maxSampPerClass:])
    return xBal, yBal



def calculateScore(X_Train_Selected, y_train, X_Test_Selected, y_test): 
    from computers import gp

    def kerneldef(h, k):
        return h(1e-3, 1e5, 1) * k('matern3on2', np.asarray([h(5e-2, 1e1, 1)
                                          for _ in range(X_Train_Selected.shape[1])]))

    
    #Evaluate accuracy using these features,
            
    # Intialise constants
    WALLTIME = 1000.0
    N_CORES = 1

    # Modeling options
    WHITEFN = 'pca'
    APPROX_METHOD = 'laplace'       # 'laplace' or 'pls'
    MULTI_METHOD = 'OVA'            # 'AVA' or 'OVA'
    FUSE_METHOD = 'EXCLUSION'       # 'MODE' or 'EXCLUSION'
    RESPONSE_NAME = 'probit'        # 'probit' or 'logistic'

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = WALLTIME
    responsefunction = gp.classifier.responses.get(RESPONSE_NAME)


    learned_classifier = gp.classifier.learn(X_Train_Selected, y_train, kerneldef,
        responsefunction, optimiser_config,
        multimethod=MULTI_METHOD, approxmethod=APPROX_METHOD,
        train=True, ftol=1e-6, processes=N_CORES)

    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, learned_classifier, np.unique(y_train))

    import logging
    # Print the matrix of learned classifier hyperparameters
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    # predictor_plt = gp.classifier.query(learned_classifier, X)


    predictor_plt = gp.classifier.query(learned_classifier, X_Test_Selected)
    exp_plt = gp.classifier.expectance(learned_classifier, predictor_plt)
    var_plt = gp.classifier.variance(learned_classifier, predictor_plt)
    yq_lde_plt = gp.classifier.linearised_model_differential_entropy(exp_plt, var_plt,
        learned_classifier)
    yq_sd_plt = gp.classifier.equivalent_standard_deviation(yq_lde_plt)
    yq_prob_plt = gp.classifier.predict_from_latent(exp_plt, var_plt,
        learned_classifier, fusemethod=FUSE_METHOD)


    import sklearn.metrics as metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_test, yq_prob_plt)
    return metrics.auc(fpr, tpr)

import itertools
def evalFeature_(X, y, feature_num):
    maxSampPerClass = np.min([2000, int(np.sum(y))])
    XCull, yCull = cullData(X, y, maxSampPerClass)
    
    totalFeatureNum = X.shape[1]
    if feature_num > totalFeatureNum:
        feature_num = totalFeatureNum
    
    np.random.seed()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
          XCull, yCull, test_size=0.2, random_state=np.round(np.random.random()*1e5).astype(int))
    scoreList = []
    featureList = []
    
    a = range(1,totalFeatureNum+1)
    for i in itertools.combinations(a, feature_num):
        X_Train_Selected = X_train[:, [item-1 for item in i]]
        X_Test_Selected = X_test[:, [item-1 for item in i]]
        #print(i,end="...") 
        featureList.append(i)
        scoreList.append(
            calculateScore(
                X_Train_Selected,
                y_train,
                X_Test_Selected,
                y_test))
    return featureList, scoreList

def prepareData(lithoLabel):
    # Set up Variables
    path = rootPath+'bdkd-external-devel/rocks/seafloorLitho/seafloor_lith_data_all_features.csv'
    nRandomFeatures = 1
    removeLabels = [9,10,11,12]

    # Load Data
    X,y, lonlat = loadData(path)

    # Remove a hemisphere from the data to analyse effects
    X, y = removeHemisphere(X, y, lonlat)

    # Preprocess the data
    X = preprocessData(X)

    # Adding random features to X
    X = np.append(X, np.random.random([X.shape[0], nRandomFeatures]), axis=1)

    # Normalise Data
    X = normaliseData(X)
            # Remove correlated features
    X = removeCorrelatedFeatures(X)

    # Map labels to groups
    mapping = {'1': 1, '2': 1, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 5,
               '9': 9, '10': 10, '11': 11, '12': 12, '13': 3}
    y_map = np.zeros(y.shape)
    for i in range(y.shape[0]):
        y_map[i] = mapping[str(int(y[i]))]

    # Remove labels as requested by geoscientist
    for remLab in removeLabels:
        X = X[y_map != remLab,:]
        y_map = y_map[y_map != remLab]

    X, y_map = utils.balanceData(X, y_map)

    y_map[y_map != lithoLabel] = 0
    y_map[y_map == lithoLabel] = 1
    return X, y_map


# In[6]:

import numpy as np

featureMap = {
    1: 'bathymetry',
    2: 'silicate',
    3: 'productivity',
    4: 'salinity',
    5: 'temperature',
    6: 'random'}

def featureToString(features):
    ret = ()
    for f in features:
        if f in featureMap:
            ret+=(featureMap[f],)
        else:
            ret+=('Unknown Feature',)
    return ret

def averageScores(scoreList):
    s = set([i[0] for i in scoreList])
    #print(s)
    averageScores = []
    for i in s:
        score = np.mean([item[1] for item in scoreList if item[0]==i])
        stdScore = np.std([item[1] for item in scoreList if item[0]==i])
        averageScores.append((i,score,stdScore))
    #print(averageScores)
    return averageScores


'''
The input 'data' is a list of scores like the list returned from evalFeature_().
The function returns a list of best mean scores and their standard deviation for each length of features.
    [
        (0.78440830919243076, 0.039494486987150655, (1,)), 
        (0.80520222723360158, 0.029503010975171547, (1, 2)), 
        (0.8237200195396438, 0.030222509761139552, (1, 2, 3)), 
        (0.82893379888111907, 0.033162962670966828, (0, 1, 2, 3)), 
        (0.82579654303048822, 0.028347542152092554, (0, 1, 2, 3, 4))
    ]
'''
def bestMeanScores(data):
    featureNum = len(max(data, key = lambda x: len(x[0]))[0]) #get the max number of features
    bestMean=[None]*(featureNum)
    
    for i in range(1,featureNum+1):
        featureScores = [item for item in data if len(item[0])==i]
        if not len(featureScores):
            #print('No model has been trained with {0} feature(s).'.format(i))
            pass
        else:
            meanScores = averageScores(featureScores)
            meanScores.sort(key = lambda x: x[1],reverse=True)
            #print('The highest mean score is {1}. The STD is {2}.' \
            #      .format(i, meanScores[0][1], meanScores[0][2]))
            bestMean[i-1] = (meanScores[0][1], meanScores[0][2],meanScores[0][0])
            #print('The feature(s): {}.'.format(featureToString(meanScores[0][0])))
    return bestMean

def printFeatureEvalMatrix(data):
    bestScores=[]
    for d in data: # for each evaluation iteration
        bestScores.append(bestMeanScores(d))
    #print(bestScores)
    maxUsedFeatureNum = 0
    allFeatures=set()
    for i in bestScores: #for each iteration
        for s in i: #for each score
            maxUsedFeatureNum = max(maxUsedFeatureNum, len(s[2]))
            for f in s[2]:
                allFeatures.add(f)
    
    iterationNum = len(data)
    matrix = np.zeros((maxUsedFeatureNum,len(allFeatures)))
    #print(len(allFeatures),maxUsedFeatureNum)
    for i in range(len(allFeatures)): #feature id
        for j in range(maxUsedFeatureNum): #feature number
            cnt=0
            for n in range(iterationNum):
                if i+1 in bestScores[n][j][2]:
                    cnt+=1
            matrix[j][i]=cnt/iterationNum
    #print(matrix)
    pl.figure()
    pl.imshow(matrix, interpolation='nearest', extent=(0.5, len(allFeatures) + 0.5, maxUsedFeatureNum + 0.5, 0.5))
    pl.xlabel('Feature ID')
    pl.ylabel('Number of Features')
    xint = range(1,len(allFeatures)+1)
    pl.xticks(xint)
    yint = range(1,maxUsedFeatureNum+1)
    pl.yticks(yint)
    pl.colorbar()
    ax = pl.gca()
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
                    
            
    
def printResults(data):
    #The data is a list of results from each evaluation iteration.
    #for example, if we have evaluated the features in 5 iterations, 
    #the data will look like data[0]=[...], data[1]=[...], ..., data[4]=[...].
    #Each iteration data is a list of ((features), score), for example ((1,2,3), 0.8723432).
    results=[]
    for r in data:
        results+=r
    
    highestScore = max(results, key = lambda x: x[1])
    print("The highest score of all models is {}.".format(highestScore[1]))
    print('The feature(s) that has/have been used to train the model are {}.'          .format(featureToString(highestScore[0])))
    print()
    
    featureNum = len(max(results, key = lambda x: len(x[0]))[0])
    #print('The max number of features that have been used to train the models is {}.'.format(featureNum))
    #print()
    
    bestMean=[None]*(featureNum)
    
    for i in range(1,featureNum+1):
        featureScores = [item for item in results if len(item[0])==i]
        if not len(featureScores):
            print('No model has been trained with {0} feature(s).'.format(i))
        else:
            featureScores.sort(key = lambda x: x[1],reverse=True)
            featureHighestScore = featureScores[0]
            print('The number of models which are trained with {0} feature(s) is {1}.'                   .format(i, len(featureScores)))
            print('The highest score is {1}.'                   .format(i, featureHighestScore[1]))
            print('The feature(s): {}.'.format(featureToString(featureHighestScore[0])))
            meanScores = averageScores(featureScores)
            meanScores.sort(key = lambda x: x[1],reverse=True)
            print('The highest mean score is {1}. The STD is {2}.'                   .format(i, meanScores[0][1], meanScores[0][2]))
            bestMean[i-1] = (meanScores[0][1], meanScores[0][2])
            print('The feature(s): {}.'.format(featureToString(meanScores[0][0])))
            
        print()

#The input 'data' is a list of lists which are returned from evalFeature_(). 
#The length of 'data' is the number of iterations that the script has run.
#Typically, the 'data' has a shape of (the number of iterations, 62, 2).
#The second dimension has a length of 62 because we use 5 real feature + 1 random feature.
#The number of all possible combinations for 6 features is 2**8=64.
#We don't care the empty set and the full set. So, it gives us 64-2=62.
#Basically, it means all numbers between 0b000001 and 0b111110.
#The second dimension does not have to be 62 as long as it is a list of scores of evaluated features.
#The third dimension has a length of 2. The first item is a tuple of features which are evaluated.
#The second item is the score. For example, ((1,2,3), 0.80497592295345111) means feature 1, 2 and 3 
#are evaluated and the score is 0.80497592295345111.
def printFeatureEvalPlot(data):
    bestMean = bestMeanScores([item for sublist in data for item in sublist])
    featureNum = 0
    for i in bestMean: #for each iteration
        featureNum = max(featureNum, len(i[2]))
    #print(featureNum, bestMean)
    #print([i[0] for i in bestMean])
    pl.figure()
    pl.errorbar(range(1,featureNum+1),[i[0] for i in bestMean],[i[1] for i in bestMean])
    pl.title('Area Under ROC Curve vs Number of Features')
    pl.xlabel('Number of features')
    pl.ylabel('AUC')
    xint = range(0,featureNum+2)
    pl.xticks(xint)
    ax = pl.gca()
    ax.set_xlim((0,featureNum+1))
    #ax.set_ylim((0.7,0.9))
    pl.show()


from multiprocessing import Pool
import multiprocessing
import time

def runEvaluation(arg, **kwarg):
    return FeatureEvaluator._evaluate(*arg, **kwarg)

class FeatureEvaluator():
    def __init__(self, label, featureNumber=6):
        self.label = label
        self.featureNumber = featureNumber
        self.results = []
        self.iterationNumber = 0
    
    def getLabel(self):
        return self.label
    
    def getFeatureNumber(self):
        return self.featureNumber
    
    def _evaluate(self, idx):
        #print('inddex: {}'.format(idx))
        output=[]
        for featureNum in range(1, self.featureNumber):
            X,y_map=prepareData(self.label)
            features, scores = evalFeature_(X,y_map,featureNum)
            output.extend(zip(features, scores))
            #output.extend([1,2])
        #print(output)
        return output
        
    def doFeatureEvaluation(self, iterationNum):
        self.iterationNumber += iterationNum
        import datetime
        print('This process might take a while. Please be patient...')
        print("begin: ",datetime.datetime.now())
        pool = Pool(processes=(multiprocessing.cpu_count()-1))
        rets = pool.map_async(runEvaluation, zip([self]*iterationNum, range(iterationNum)))         
        while not rets.ready():
            print('***',end='|')
            time.sleep(5)
            
        #print('results: {}'.format(rets.get()))
        self.results.extend(rets.get())
        print()
        print("end: ",datetime.datetime.now())
    
    def getResults(self):
        return self.results
    
    def printFeatureEvalPlot(self):
        printFeatureEvalPlot(self.results)
        
    def printFeatureEvalMatrix(self):
        printFeatureEvalMatrix(self.results)

    def getMeanScores(self):
        ret=[]
        s = set([i[0] for iteration in self.results for i in iteration])#get all unique feature combinations
        for featureCombination in s:
            #get all scores of this feature combination
            tmp = [item[1] for iteration in self.results for item in iteration if item[0]==featureCombination]
            score = np.mean(tmp)
            stdScore = np.std(tmp)
            ret.append((featureCombination, score, stdScore))
        return ret
    
    def getHighestMeanScores(self):
        ret=[]
        scores = self.getMeanScores()
        for featureNumber in range(1, self.featureNumber):
             ret.append(max([i for i in scores if len(i[0])==featureNumber],key = lambda x: x[1]))
        return ret
            
    def getHighestScores(self):
        ret = []
        for iteration in self.results:
            highestScores=[]
            for featureNumber in range(1, self.featureNumber):
                highestScores.append(max([i for i in iteration if len(i[0])==featureNumber],key = lambda x: x[1]))
            ret.append(highestScores)
        return ret






