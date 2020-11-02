
from snorkel_labeler.lfs.lf_generator import LFGenerator
from snorkel.labeling import LFApplier
from snorkel.labeling import LFAnalysis

from concepts.concept_recognizer import ConceptRecognizer

class BaseLabeler(object):
    def __init__(self, arg):
        self.arg = arg
        self.resource = arg.resource

        self.labeled_events_pool = self.resource.get_labeled_event_pool('Lsnorkel') ## used to store weakly labeled events



    def generate_labeling_functions(self):
        lfgenerator = LFGenerator(self.arg)
        lfs = lfgenerator.generate_lfs()
        print(f"### total lfs num: {len(lfs)}")
        return lfs



    def collect_weak_labels(self):
        lfs = self.generate_labeling_functions()
        applier = LFApplier(lfs=lfs)
        eventData = self.resource.get_event_dict_data()
        eventlist = eventData.get_all_unlabeled_events()
        eventlist = eventlist[:30000]

        crecognizer = ConceptRecognizer(self.arg)
        crecognizer.recognize_event_concepts(eventlist)

        L_train = applier.apply(eventlist, progress_bar=False)

        total_num = len(eventlist)
        #print("coverage : ", coverage)
        summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
        #print(f"\nsummary : total candidates: {len(eventlist)} \n", type(summary))
        for idx, r in summary.iterrows():
            coverNum = int(r.Coverage*total_num)
            #print(f"{idx:60}  j:{r.j:5} Polarity:{str(r.Polarity):5} Coverage:{r.Coverage:>.5f}({coverNum:>10}) Overlaps:{r.Overlaps:0.5f} Conflicts:{r.Conflicts:>.5f}")
        #print(f"\n")


        self.labeled_events_pool.set_init_label_matrix( L_train, eventlist )
        #self.labeled_events_pool.set_init_label_data( eventlist  )

        #self.arg.conceptsetdata.collect_valid_concepts( lfs, L_train )
        self.arg.crdata.collect_valid_concept_rules(lfs, L_train)


        #self.labeled_events_pool.show_train_label_examples()

        #print()
        #print(f" ###Unlabel Instances : {len(eventlist)}")
        print(f" ###LFs num: {len(lfs)}")
        print(f" ###Concept Rule Candidate Size : {len(self.arg.crdata.concept_rule_candidates)}")
        #print(f" ###L_matrix shape : {L_train.shape}")


