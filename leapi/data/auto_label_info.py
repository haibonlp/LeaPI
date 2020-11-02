# coding:utf-8


class AutoLabelInfo(object):
  def __init__(self, label_function_name, class_name):
    self.lfname = label_function_name
    self.class_name = class_name


  def __str__(self):
    return f"{self.class_name:15}  {self.lfname:15}   "



class AutoLabelPool(object):
  def __init__(self, event):
    self.event = event
    self.weak_label_list = []


  def set_aggregated_label(self, label):
    self.aggregated_label = label


  def get_aggregated_label(self):
    return self.aggregated_label


  def get_weak_labels(self):
    return self.weak_label_list


  def add_weak_label(self, label_function_name, class_name):
    labelinfo = AutoLabelInfo(label_function_name, class_name)
    self.weak_label_list.append(labelinfo)





