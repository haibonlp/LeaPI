# coding:utf-8

import sys
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

import numpy as np




class SKLearnClassifier:
  def __init__(self, clfName='logit'):
    #self.arg = arg

    self.vectorizer = DictVectorizer()


    if clfName == 'logit':
      self.basemodel = LogisticRegression( multi_class="auto",  solver='newton-cg')
    elif clfName == 'nb':
      self.basemodel = MultinomialNB(alpha=0.1)
      #self.basemodel = BernoulliNB(alpha=.01)
    elif clfName == 'svm':
      self.basemodel = SVC(kernel='linear', probability=True )



  def train(self, trainFeatList, trainY ):
    #print( 'start train ....' )

    trainX = self.vectorizer.fit_transform(trainFeatList)

    self.basemodel.fit(trainX, trainY)
    pY = self.predict( trainFeatList )
    num = len( [ i for i, j in zip(pY, trainY) if i ==j] )
    #print( ' --- train acc : %.2f '% ( 100* float(num)/len(trainY) ) )



  def predict(self, testFeatList):
    #testFeatList = []
    #for ev in testlist:
    #  testFeatList.append( ev.feat2val )

    testX = self.vectorizer.transform( testFeatList )

    #Y_new = self.basemodel.predict_proba(X)
    Y_new  = self.basemodel.predict( testX)

    return Y_new


  def predict_proba(self, testFeatList):
    #testFeatList = []
    #for ev in testlist:
    #  testFeatList.append( ev.feat2val )

    testX = self.vectorizer.transform( testFeatList )

    probY = self.basemodel.predict_proba(testX)

    #print( 'feat testX :  ', type(testX), testX.shape )
    #print ' unlabeled, probY : ',  type(probY), probY.shape
    #print probY[0]

    return probY









