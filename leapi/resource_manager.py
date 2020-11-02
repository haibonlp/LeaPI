# coding:utf-8

## this class will manage all resources used in this project

#import redis
from data.event_dict_data import EventDictData
from data.probase_db import ProbaseDB
from data.auto_labeled_events_pool import AutoLabeledEventsPool
from weaklabeldata.auto_labeled_events_pool_snorkel import AutoLabeledEventsPoolSnorkel
from data.event_embedding_data import  EventEmbeddingData
from data.human_need_gold_data import HumanNeedGoldData
import util

class ResourceManager(object):
  def __init__(self, arg):
    self.arg = arg

    self.probaseDB = ProbaseDB()

    self.has_event_dict_data = False
    self.has_labeled_events_pool = False
    self.has_event_embedding_data = False
    self.has_human_need_gold_data = False
    self.has_word2vec_data = False



  def get_event_dict_data(self):
    if not self.has_event_dict_data:
      self.event_dict_data = EventDictData(self.arg)
      self.has_event_dict_data = True
    return self.event_dict_data


  def get_probase_db(self):
    return self.probaseDB


  def get_labeled_event_pool(self, pooltype='basic'):
      if not self.has_labeled_events_pool:
          self.has_labeled_events_pool = True
          if pooltype == 'basic':
              self.labeled_event_pool = AutoLabeledEventsPool(self.arg)
          elif pooltype == 'Lsnorkel':
              self.labeled_event_pool = AutoLabeledEventsPoolSnorkel(self.arg)
          else:
              self.labeled_event_pool = None
      return self.labeled_event_pool


  def get_event_embedding_data(self):
    if not self.has_event_embedding_data:
      self.has_event_embedding_data = True
      self.event_embed_data = EventEmbeddingData()
      self.event_embed_data.load_event_embedding_data(self.arg.event_vec_fpath)

    return self.event_embed_data


  def get_word2vec_data(self):
    if not self.has_word2vec_data:
        self.has_word2vec_data = True
        w2vfpath = self.arg.word2vecfpath
        self.word2vec =  util.load_gensim_word2vec_model(w2vfpath)

    return self.word2vec


  def get_human_need_gold_data(self):
    if not self.has_human_need_gold_data:
      self.has_human_need_gold_data = True
      self.hneed_gold_data = HumanNeedGoldData(self.arg)
    return self.hneed_gold_data




