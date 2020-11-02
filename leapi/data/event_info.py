# coding:utf-8

import collections

wordMap = { "@fps@"   : "i",
             "@fps@$"  : "my",
             "@fpp@"   : "we",
             "@fpp@$"  : "our",
             "@sp@"    : "you",
             "@sp@$"   : "your",
             "@tpm@"   : "he",
             "@tpm@$"  : "his",
             "@tpf@"   : "she",
             "@tpf@$"  : "her",
             "@tpp@"   : "they",
             "@tpp@$"  : "their"
     }

class CEventInfo(object):
    def __init__(self):
        #self.evid = None
        #self.evstr = None
        #self.evtup = None ### event representation
        #self.evptag = None
        #self.evhtag = None
        #self.feat2val = {}

        self.field_names = ['Agent', 'Predicate', 'Theme', 'PP']

        self.field2concepts = collections.OrderedDict()


    def __repr__(self):
        return f"<ID:{self.evid} <{self.Agent}, {self.Predicate}, {self.Theme}, {self.PP}> >  "




    def load_event_info_from_7fields_str(self, evtup, evinfo):
        self.evtup      = evtup
        self.evid       = evinfo['evId']
        self.ev4field   = self.convert_7fields_to_4fields(evtup)
        self.idxStr     = evinfo['docs']

        self.field2value = self.get_field2value(self.ev4field)
        self.field2head  = self.get_field2head_words()


    def get_field2value(self, ev4field):
        name2value = {}
        for name, value in zip(self.field_names, ev4field):
            name2value[name] = value
        self.Agent      = name2value['Agent']
        self.Predicate  = name2value['Predicate']
        self.Theme      = name2value['Theme']
        self.PP         = name2value['PP']
        return name2value


    def get_value_by_field(self, fieldname):
        return self.field2value[fieldname]

    def get_head_by_field(self, fieldname):
        return self.field2head[fieldname]


    def remove_preposition_in_PP(self):
        PP = ' '.join( self.PP.split()[1:] )
        return PP

    def remove_possessive_pronoun(self, text):
        words = text.strip().split()
        if len(words) <= 1:
            return text
        if words[0].endswith("@$"):
            text = " ".join( words[1:] )
            #print(f"##possessive pronoun : [{text}]  <==  {words}")
        return text



    def convert_7fields_to_4fields(self, evstr):
      terms = evstr.split('===')
      if len(terms) != 7:
        print( 'Error, foramt : ', evstr )
        return evstr
      terms = [t.strip() for t in terms]
      newtermlist = []
      for t in terms:
        t = t.strip()
        wlist = t.split()
        words = []
        for w in wlist:
          #if w in wordMap:
          #  w = wordMap[w]
          words.append(w)
        t = ' '.join(words)
        newtermlist.append(t)

      (agent, negator, verb, theme, infverb, infverbObj, prep) = newtermlist

      fields = []
      agentField = agent.strip()
      fields.append(agentField)

      verbField = negator +' '+ verb
      if len(infverb) >0:
        verbField += ' ' + infverb
      verbField = ' '.join( verbField.strip().split() )
      fields.append( verbField.strip() )

      themeField = theme + ' ' + infverbObj
      themeField = ' '.join( themeField.strip().split() )
      fields.append( themeField.strip() )
      prepField = prep.strip()
      fields.append(prepField)

      # <Agent, Predicat, Theme, PP>
      return fields


    def get_field2head_words(self):
      ''' Use last word as head word '''
      field2head = {}
      for field, wstr in zip(self.field_names, self.ev4field):
        words = wstr.strip().split()
        if len(words) <=0:
          continue
        head = words[-1]
        field2head[field] = head
      return field2head




    def get_event_words(self, evtup):
      terms = evtup.split('===')
      words = []
      for t in terms:
        t = t.strip()
        itemlist = t.split()
        for item in itemlist:
          item = item.strip()
          wlist = item.split('_')
          for w in wlist:
            w = w.strip()
            if w in wordMap:
                w = wordMap[w]
            if len(w) > 0:
              words.append(w)
      #print 'evtup : ', evtup, words
      return words






