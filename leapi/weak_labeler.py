# coding:utf-8


from snorkel_labeler.base_snorkel_labeler import BaseLabeler


class WeakLabeler(object):
  def __init__(self, arg ):
    self.arg = arg
    self.resource = arg.resource

    self.labeler = BaseLabeler(arg)


  def main_collect_weak_labels(self):
    self.labeler.collect_weak_labels()




