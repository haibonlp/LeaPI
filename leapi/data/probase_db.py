# coding:utf-8

import json
import redis


class ProbaseDB(object):
    def __init__(self):

        self.db = redis.Redis("localhost", db=1)

    def get_concept_freqs_by_word(self, word):
        ret = self.db.hget("Probase_word2concept2freq", word)
        if ret is None:
            return None
        ret = json.loads(ret)
        return ret

    def get_word_freqs_by_concept(self, concept):
        ret = self.db.hget("Probase_concept2word2freq", concept)
        if ret is None:
            return None
        ret = json.loads(ret)
        return ret

    def get_topK_insts_by_concept(self, concept, topK):
        inst2freq = self.get_word_freqs_by_concept(concept)
        if inst2freq is None:
            #print(f"Error! not found concept : [{concept}]")
            return []
        tmplist = sorted(inst2freq.items(), key=lambda x:int(x[1]), reverse=True)
        ret = [ x[0] for x in tmplist[ : topK] ]
        return ret


    def get_topK_insts_freq_by_concept(self, concept, topK):
        inst2freq = self.get_word_freqs_by_concept(concept)
        if inst2freq is None:
            print(f"Error! not found concept : [{concept}]")
            return []
        tmplist = sorted(inst2freq.items(), key=lambda x:int(x[1]), reverse=True)
        #ret = [ x[0] for x in tmplist[ : topK] ]
        ret = tmplist[:topK]
        return ret


def test():
  db = ProbaseDB()
  wordlist = ['apple', 'love', 'happy', 'be', 'was', 'learning', 'study']

  print(f"***** word -> concepts *****")
  for w in wordlist:
    c2f = db.get_concept_freqs_by_word(w)
    print(f"\n=== {w}")
    if c2f is None:
      print(c2f)
      continue
    ret = sorted(c2f.items(), key=lambda x:int(x[1]), reverse=True)
    print('     ', ret[:15])


  clist = ['eat', 'food', 'equipment', 'leisure', 'drink', 'health', 'death', 'medical equipment',
      'entertainment', 'employment', 'business', 'learning', 'study', 'health problem', 'health' ]
  print(f"***** concept --> words *****")

  for c in clist:
    w2f = db.get_word_freqs_by_concept(c)
    ret = None
    if w2f:
      ret = sorted(w2f.items(), key=lambda x:int(x[1]), reverse=True)
      ret = ret[:15]
    print(f"\n {c:15}  -->   {ret} ")






if __name__ == '__main__':

  #db = ProbaseDB()
  #print(db.get_concept_freqs_by_word("apple"))
  test()


