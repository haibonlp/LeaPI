# coding:utf-8

import numpy as np
from snorkel_labeler import hneedlabels


class LabelAggregator(object):
    def __init__(self, arg):
        self.arg = arg
        self.aggregator = AggregatorAverageVote(arg)


    def main_aggregate_labels(self):
        #print(f"#### start to aggregate labels : ")
        self.aggregator.aggregate_labels()
        #print(f"--- label information after aggregation... ")
        self.arg.resource.get_labeled_event_pool().show_info()


    def main_aggregate_labels_selected(self):
        #print(f"#### start to aggregate labels : ")
        self.aggregator.aggregate_labels_selected()
        #print(f"--- label information after aggregation... ")
        self.arg.resource.get_labeled_event_pool().show_info()




"""
class AggregatorFilterMultiLabels:
  '''
  This class tries to select a label for each event
  by filtering all events that were assigned with multi-labels.
  '''
  def __init__(self, arg):
    self.resource = arg.resource
    self.labeled_event_pool = self.resource.get_labeled_event_pool()



  def aggregate_labels(self):
    filtered_evids = []
    for evid in self.labeled_event_pool.evid2labelpool:
      labelpool = self.labeled_event_pool.evid2labelpool[evid]
      labelset = set()
      for linfo in labelpool.get_weak_labels():
        labelset.add( linfo.class_name )

      if len(labelset)>1:
        filtered_evids.append(evid)
        labelpool.set_aggregated_label( None)
        #labelpool.aggregated_label = None
      else:
        #labelpool.aggregated_label = list(labelset)[0]
        labelpool.set_aggregated_label( list(labelset)[0] )

    ### remove all filtered labeled events
    for evid in filtered_evids:
      del self.labeled_event_pool.evid2labelpool[evid]
"""



class AggregatorAverageVote():
    def __init__(self, arg):
        self.arg = arg
        num_classes = hneedlabels.get_num_of_classes()
        self.cardinality = num_classes
        self.labeled_event_pool = self.arg.resource.get_labeled_event_pool('Lsnorkel')

    def compute_vote_normalized(self, L):
        n, m = L.shape
        Y_p = np.zeros((n, self.cardinality), dtype=float)
        for i in range(n):
            #counts = np.zeros(self.cardinality)
            for j in range(m):
                if L[i, j] != -1:
                    #counts[L[i, j]] += 1
                    Y_p[i, L[i,j] ] += 1
            #Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_sum = Y_p.sum(axis=1).reshape(-1, 1)
        Y_sum[ Y_sum==0 ] = 1e-8
        Y_p /= Y_sum
        return Y_p


    def aggregate_labels(self):
        L_matrix = self.labeled_event_pool.get_label_matrix()

        L_probs = self.compute_vote_normalized(L_matrix)
        self.labeled_event_pool.set_weak_label_probs(L_probs)



    def aggregate_labels_selected(self):

        L_matrix = self.labeled_event_pool.get_label_matrix_selected()

        L_probs = self.compute_vote_normalized(L_matrix)
        self.labeled_event_pool.set_weak_label_probs(L_probs)






"""
class AggregatorMajorityVote(object):
  '''
  This class choose the human need label for an event
  by using majority voting.
  '''
  def __init__(self):
    pass


  def aggregate_labels(self):
    pass
"""


