# coding:utf-8

import util


class EventEmbeddingData(object):
  def __init__(self):
    pass


  def load_event_embedding_data(self, infpath):
    self.event2vec =  util.load_gensim_word2vec_model(infpath)


  def get_event_embeddin_dimension(self):
    return self.event2vec.vector_size


  def has_embedding(self, key):
    return key in self.event2vec


  def get_embedding_from_key(self, key):
    return self.event2vec[key]




