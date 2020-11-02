# coding:utf-8

import sys, json
#from event_info import CEventInfo
from time import time


#sys.path.append("/ssd-onefish/hbding/researchOnefish/affective-events/data-for-new-methods-for-human-needs/2019-01-methods-and-experiments/learn-human-needs/logistic-regression/sklearn-classifiers/")
#from featextractors.semantic_field_predict_feature_v2 import SemanticPredictFieldFeatureV2



class EventFeatureExtractor:
  def __init__(self, arg):
    self.arg = arg
    self.resource = self.arg.resource





  def extract_event_features(self, event):
    feat2val =  {}

    if len(self.arg.featlist) <=0:
      raise Exception( 'Error! Did not use any features!!' )

    for featType in self.arg.featlist:
      if featType == 'BOW':
        self.extract_event_feature_bowFeat( event, feat2val )

      elif featType == 'AvgVec':
        self.extract_event_feature_avgvec( event, feat2val )

      else:
        raise Exception( f"Error! Not found featType: [{featType}] !!" )

    return feat2val




  def extract_event_feature_bowFeat(self, event, feat2val):

    pass



  def get_word_ngram(self, words):
    feat2val = {}
    NgramSize = self.arg.ngram_size
    wordNum = len(words)
    for i in xrange(wordNum):
      suffixWord = ''
      for n in xrange(NgramSize):
        fkey = 'ng' + str(n+1)
        if n <=0:
          ngram = words[i].lower()
        else:
          pi = i - n
          if pi >=0 and pi < wordNum:
            ngram = words[pi].lower()  + '/' + suffixWord
          else:
            ngram = '<S>/'+ suffixWord
        fstr = fkey + '=' + ngram
        feat2val[fstr] = 1.0
        suffixWord = ngram
    #print words
    #print '     feat2val :::: ', sorted(feat2val.items(), key=lambda x :x[0] )
    return feat2val





  def extract_event_feature_avgvec(self, event, feat2val ):
    eventEmbedData = self.resource.get_event_embedding_data()

    vec_size = eventEmbedData.get_event_embeddin_dimension()
    #evlist = self.trainlist + self.testlist
    evkey = '$$event$$_'+str(event.evid)
    if not eventEmbedData.has_embedding(evkey):
      print( 'Error! not found ev : [%s]'% evkey )
      return
    vec = eventEmbedData.get_embedding_from_key(evkey)
    #print type(vec), vec.shape, vec
    for i in range(vec_size):
      fkey = 'AvgEmbed:d'+str(i)
      feat2val[fkey] = vec[i]
    #ev.feat2val = feat2val
    #print  '\n', ev.evid, sorted(feat2val.items(), key=lambda x:x[0] )




