# coding:utf-8

from data.auto_label_info import AutoLabelInfo, AutoLabelPool
import random

class AutoLabeledEventsPool(object):
  def __init__(self, arg):
    self.arg = arg
    self.evid2labelpool = {}

    ## label function to ID
    self.lfname2id = {}


  #def get_label_pool(self):
  #  return self.evid2labelpool.items()


  def get_label_function_id(self, lfname):
    return self.lfname2id.get(lfname)

  def get_num_of_labeled_insts(self):
    return len(self.evid2labelpool)


  def get_num_of_label_functions(self):
    return len(self.lfname2id)


  def add_label_function(self, lfname):
    if lfname in self.lfname2id:
      return
    lid = len(self.lfname2id)
    self.lfname2id[lfname] = lid


  def get_label_pools(self):
    pools = []
    for evid in self.evid2labelpool:
      labelpool = self.evid2labelpool[evid]
      pools.append(labelpool)
    return pools


  def add_labeled_event(self, event, lfname, hneed):
    if event.evid not in self.evid2labelpool:
      self.evid2labelpool[event.evid] = AutoLabelPool(event)

    #labelinfo = AutoLabelInfo(event, lfname, hneed)
    labelpool = self.evid2labelpool[event.evid]
    labelpool.add_weak_label(lfname, hneed)

    ### add label function
    self.add_label_function(lfname)
    #self.evid2labelpool[event.evid].append( labelinfo )
    #print(f"add label event : \n     ", labelinfo)


  def show_info(self):
    total = len(self.evid2labelpool)
    hneed2freq = {}
    num_labels = 0
    for evid in self.evid2labelpool:
      labelpool = self.evid2labelpool[evid]
      num_labels += len(labelpool.weak_label_list)
      for linfo in labelpool.weak_label_list:
        if linfo.class_name not in hneed2freq:
          hneed2freq[linfo.class_name] = 0
        hneed2freq[linfo.class_name] += 1
    print(f"\n----- total labeled events: {total}")
    print(f"        total labels : {num_labels} ")
    for hneed in hneed2freq:
      print(f"           {hneed:15}   freq: {hneed2freq[hneed]}")




########### Prepare training event samples ... #####

  def prepare_train_set(self):
    '''
    obtain train samples for downstream event-human-need classifier
    '''
    traindata = [] ### each element is a tuple (event, label)

    evidset = set()
    labelset = set()
    for evid, labelpool in self.evid2labelpool.items():
      evidset.add(evid)
      event = labelpool.event
      label = labelpool.get_aggregated_label()
      inst = (event, label)
      traindata.append(inst)

      labelset.add(label)
    num_inst = len(traindata)
    num_class = len(labelset)
    num_inst = int(num_inst/num_class)

    self.select_samples_for_None_class(traindata, num_inst, evidset)

    self.print_train_set_info(traindata)

    return traindata


  def print_train_set_info(self, traindata):
    print('\n', '##'*10, ' Train Info ', '##'*10)
    label2freq = {}
    print(f"   Total train size : {len(traindata)} ")
    for event, label in traindata:
      if label not in label2freq:
        label2freq[label] = 0
      label2freq[label] += 1

    for label in label2freq:
      print(f'       label : {label:20}   num : {label2freq[label]}')

    print('\n')



  def select_samples_for_None_class(self, traindata, num_inst, hnEvidset ):
    '''
    randomly select num_inst events that were not previously selected
    '''
    instlist = []
    eventDictData = self.arg.resource.get_event_dict_data()
    for event in eventDictData.get_all_events():
      if event.evid in hnEvidset:
        continue
      inst = (event, 'None')
      instlist.append(inst)

    random.shuffle(instlist)
    retlist = instlist[:num_inst]

    print(f"  Select None class num: {len(retlist)}")

    traindata += retlist





