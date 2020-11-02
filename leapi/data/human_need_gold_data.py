# coding:utf-8

import json
from data.event_info import CEventInfo

class HumanNeedGoldData(object):
  def __init__(self, arg):
    self.arg = arg

    self.set_label_index()

    self.official_evaluation_events = self.load_train_test_by_foldId(1)
    print(f" ==== #Official Evaluation Event size : {len(self.official_evaluation_events)}")


  def set_label_index(self):
    self.tag2id = {'Physiological': 0, 'Health': 1,  'Leisure':2 , 'Social':3,
                  'Finance': 4, 'Cognition':5, 'Mental': 6,  'None': 7 }
    self.id2tag = { 0: 'Physiological', 1:'Health',  2:'Leisure', 3:'Social',
                       4: 'Finance', 5:'Cognition',  6:'Mental',  7:'None' }


  def get_class_names(self):
    return self.tag2id.keys()


  def get_num_of_classes(self):
    return len(self.tag2id)


  def get_label_from_id(self, tid):
    label = self.id2tag.get(tid, None)
    if label is None:
      raise Exception(f"Error! Not found class-id :{tid} !!")
    return label


  def get_labelId_by_label(self, label):
    labelId = self.tag2id.get(label, None)
    if labelId is None:
      raise Exception(f"Error! The query label [{label}] not found!")
    return labelId



  def load_train_test_by_foldId(self, foldId):

    trainfpath = self.arg.hneed_gold_data_fprefix+ '/fold-'+str(foldId)+'/official-humanNeed-fold-'+str(foldId)+'-train.json'
    testfpath = self.arg.hneed_gold_data_fprefix+ '/fold-'+str(foldId)+'/official-humanNeed-fold-'+str(foldId)+'-test.json'

    ## load train data
    self.trainlist = self.load_event_info(trainfpath)
    self.testlist  = self.load_event_info(testfpath)

    #print(f"### Fold ID : {foldId}  ###")
    #print( 'Loading train size : ', len(self.trainlist) )
    #print( 'Loading test  size : ', len(self.testlist) )

    all_gold_events = self.trainlist + self.testlist
    return all_gold_events


  def load_dev_data(self):
      devfpath = self.arg.hneed_dev_data_fpath
      self.devlist = self.load_event_info(devfpath)
      #print(f"   ==DevSet size : {len(self.devlist)}")
      return self.devlist


  def get_dev_set(self):
      return self.devlist


  def get_test_set(self):
    return self.testlist


  def load_event_info(self, infpath):
    with open(infpath) as InFile:
      tmplist = json.load(InFile)
    eventlist = []
    for (ev4field, evtup, evid, ptag, htag ) in tmplist:
      event = CEventInfo()
      event.feat2val = {}
      event.ev4field = ev4field
      event.evtup = evtup
      event.evid  = evid
      event.ptag  = ptag
      event.htag  = htag
      eventlist.append( event )
    return eventlist


