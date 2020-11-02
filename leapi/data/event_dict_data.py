# coding:utf-8

import redis
import json, gzip
from data.event_info import CEventInfo


class EventDictData(object):
    def __init__(self, arg):
        self.arg = arg

        self.evid2event = {}

        #self.eventlist = self.load_all_dict_events_from_redis()
        self.eventlist = self.load_all_dict_events()


    def get_event_by_id(self, evid):
        return self.evid2event[evid]



    def load_all_dict_events(self ):
        with gzip.open(self.arg.eventdictfpath) as InFile:
            ev2info = json.load(InFile)
        eventlist = []
        for evtup in ev2info:
            evid      = ev2info[evtup]['evId']
            evinfo    = ev2info[evtup]
            event     = CEventInfo()
            event.feat2val = {}
            event.load_event_info_from_7fields_str(evtup, evinfo)

            eventlist.append(event)
            self.evid2event[evid] = event
        print(f"### load event-dict data, num of events: {len(eventlist)}")
        return eventlist


    def load_all_dict_events_from_redis(self ):
        db = redis.Redis("localhost", db=1)
        ev2info = json.loads( db.get("EventDictData") )
        eventlist = []
        for evtup in ev2info:
            evid      = ev2info[evtup]['evId']
            evinfo    = ev2info[evtup]
            event     = CEventInfo()
            event.feat2val = {}
            event.load_event_info_from_7fields_str(evtup, evinfo)

            eventlist.append(event)
            self.evid2event[evid] = event

        print(f"### load event-dict data, num of events: {len(eventlist)}")
        return eventlist


    def get_all_events(self):
        return self.eventlist


    def get_all_unlabeled_events(self):
        evaldata = self.arg.resource.get_human_need_gold_data()
        eval_evids = set()
        for ev in evaldata.official_evaluation_events:
            eval_evids.add(ev.evid)

        unlabel_events = []
        for ev in self.eventlist:
            if ev.evid not in eval_evids:
                unlabel_events.append(ev)
        print(f"####all unlabeled events  size : {len(unlabel_events)}")
        return unlabel_events






