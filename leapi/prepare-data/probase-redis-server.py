# coding:utf-8

import redis
import json

class ProbaseRedisServer:
  def __init__(self, fpath):
    try:
      #conn = redis.StrictRedis(
      #    host='localhost',
      #    port=6379,
      #    password='')
      db = redis.Redis(host='localhost', port=6379, db=1)   ### db number is 1
      print(db)
      db.ping()
      print('Connected!')
    except Exception as ex:
      print('Error:', ex)
      exit('Failed to connect, terminating.')
    self.db = db


    #self.probase_fpath = "./data-concept/data-concept-instance-relations.txt"
    #self.probase_fpath = "/home/haibo/haiboWorkSpace/bigdisk1/data/research-data/probase/data-concept/data-concept-instance-relations.txt" # rtcw1
    #self.probase_fpath = "../bigdata/probase/data-concept/data-concept-instance-relations.txt"
    self.probase_fpath = fpath
    self.minfreq = 2



  def add_key_value(self, key, value):
    self.db.set(key, value)

    #print(db.keys())

    #db.set("a", 1)
    #db.set("a", 2)
    #print(db.get("a"))


  def load_probase_dict(self):
    word2concept2freq = {}
    concept2word2freq = {}
    num_pairs = 0
    with open(self.probase_fpath) as InFile:
      for line in InFile:
        line = line.strip()
        termlist = line.split('\t')
        if len(termlist) != 3:
          print(f"Error format :: {line}")
          continue
        ##(concept, instance, freq  )
        concept, word, freq = termlist[0], termlist[1], termlist[2]
        if int(freq) < self.minfreq:
          continue

        if concept not in concept2word2freq:
          concept2word2freq[concept] = {}
        concept2word2freq[concept][word]= freq
        if word not in word2concept2freq:
          word2concept2freq[word] = {}
        word2concept2freq[word][concept] = freq

        #if concept not in concept2word2freq:
        #  concept2word2freq[concept] = []
        #concept2word2freq[concept].append( word + '\t' + freq  )
        #if word not in word2concept2freq:
        #  word2concept2freq[word] = []
        #word2concept2freq[word].append( concept+'\t'+freq )

        num_pairs += 1

    print(f"minfreq: {self.minfreq}  unique concepts: {len(concept2word2freq)}   unique words: {len(word2concept2freq)}   num_pairs : {num_pairs} ")




    return word2concept2freq, concept2word2freq




  def main_create_redis_server(self):

    word2concept2freq, concept2word2freq = self.load_probase_dict()

    ## convert to json string
    for word in word2concept2freq:
      tmpstr = json.dumps(word2concept2freq[word])
      self.db.hset('Probase_word2concept2freq', word, tmpstr)
      #word2concept2freq[word] = tmpstr

    for concept in concept2word2freq:
      tmpstr = json.dumps(concept2word2freq[concept])
      #concept2word2freq[concept] = tmpstr
      self.db.hset('Probase_concept2word2freq', concept, tmpstr)

    #self.db.hmset('Probase_word2conceptFreqs', word2concept2freq)
    #self.db.hmset('Probase_concept2wordFreqs', concept2word2freq)


    pass




if __name__ == '__main__':

  #probase_fpath = "../bigdata/probase/data-concept/data-concept-instance-relations.txt"
  probase_fpath = "../../resources/probase-data-concept-instance-relations.txt"
  server = ProbaseRedisServer(probase_fpath)
  server.main_create_redis_server()



