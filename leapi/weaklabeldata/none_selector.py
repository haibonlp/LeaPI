import random
import numpy as np
from scipy.spatial.distance import cdist

class NoneClassSelector:
    def __init__(self, arg, eventlist, orgtrain):
        self.arg = arg
        self.eventlist = eventlist
        self.orgtrain = orgtrain

        self.orgtrain_evids = self.get_event_ids(orgtrain)
        #print(f" ###orgtrain_evids set : {len(self.orgtrain_evids)}")


    def get_event_ids(self, instlist):
        evids = set()
        for inst in instlist:
            evids.add( inst['event'].evid )
        return evids


    def select_none_examples(self, size):
        nonedata = []

        nonedata = self.select_none_class_by_random(size)
        #nonedata = self.select_none_class_by_distance(size)

        #print(f" ===== select NONE class num: {len(nonedata)}")

        nonedata = self.convert_to_train_inst(nonedata)

        return nonedata

    def convert_to_train_inst(self, tmplist):
        retlist = []
        for event in tmplist:
            inst = {}
            inst['event'] = event
            inst['label'] = 'None'
            retlist.append(inst)
        return retlist


    def select_none_class_by_random(self, size):
        '''
        randomly select
        '''
        tmplist = []
        for event in self.eventlist:
            if event.evid not in self.orgtrain_evids:
                tmplist.append(event)
        retlist = tmplist[:size]

        return retlist


    def select_none_class_by_distance(self, size):
        '''
        select based on the average distance to all other classes
        '''


        eventEmbs = self.arg.resource.get_event_embedding_data()
        class2embs = {}
        class_names = set( [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental' ] )
        for inst in self.orgtrain:
            event = inst['event']
            label = inst['label']
            if label == 'None':
                continue
            evkey = '$$event$$_'+str(event.evid)
            v = eventEmbs.get_embedding_from_key(evkey)
            if label not in class2embs:
                class2embs[label] = []
            class2embs[label].append(v)

        other_class_embs = []
        for label in class2embs:
            emb = np.mean(class2embs[label], axis=0)
            other_class_embs.append(emb)
            print(f" Class ProtoAvgEmbs: {label:<15} {emb}")
        other_class_embs = np.array(other_class_embs)
        print(f"other_class_embs: {other_class_embs.shape}")
        cand_embs = []
        cand_events = []
        for ev in self.eventlist:
            if ev.evid in self.orgtrain_evids:
                continue
            evkey = '$$event$$_'+str(ev.evid)
            v = eventEmbs.get_embedding_from_key(evkey)
            cand_embs.append(v)
            cand_events.append(ev)
        cand_embs = np.array(cand_embs)
        print("#selectNone, cand_embs shape: {cand_embs.shape}, cand_events:{len(cand_events)}")
        dist = cdist(cand_embs, other_class_embs, 'cosine')
        dist = np.mean(dist, axis=1)
        terms = [(e, dist[i]) for i, e in enumerate(cand_events) ]

        terms = sorted(terms, key=lambda x:x[1], reverse=True)
        selected_none_events = []
        for t in terms[:size]:
            selected_none_events.append( t[0] )
            #[t[0] for t in terms[: size]]
        return selected_none_events




