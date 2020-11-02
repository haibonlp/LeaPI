import json

from snorkel.labeling import LFAnalysis
from collections import OrderedDict


class ConceptRuleData:
    def __init__(self, arg):
        self.arg = arg

        self.concept_rule_candidates = []
        self.concept2instances = {}
        self.inst2concepts = {}
        self.idx2concept_rule = {}

        self.selected_concept_rules = []

        self.load_concept_rule_candidates()



    def load_concept_rule_candidates(self):
        fpath = self.arg.concept_cand_fpath
        with open(fpath) as InFile:
            for line in InFile:
                tmp = json.loads(line)
                if 'concept_rule' in tmp:
                    self.concept_rule_candidates.append(tmp)
                elif 'concept2instances' in tmp:
                    concept = tmp['concept2instances']['concept']
                    insts = tmp['concept2instances']['instances']
                    self.concept2instances[concept] = insts

        for idx, r in enumerate(self.concept_rule_candidates):
            r['index'] = idx
            self.idx2concept_rule[idx] = r

        for concept in self.concept2instances:
            for inst in self.concept2instances[concept]:
                if inst not in self.inst2concepts:
                    self.inst2concepts[inst] = set()
                self.inst2concepts[inst].add( concept)



    def collect_valid_concept_rules(self, lfs, L):
        '''
        lfs: list of labeling functions
        L  : labeling matrix
        '''
        analysis = LFAnalysis(L=L, lfs=lfs)
        coverages = analysis.lf_coverages()
        conflicts = analysis.lf_conflicts()

        print(f"### LFs Coverages shape: {coverages.shape}")

        self.valid_concept_rules = []
        self.valid_idx2concept_rule = OrderedDict()
        for i, lf in enumerate(lfs):
            r = lf.concept_rule
            if coverages[i] <= 0:
                continue
            if coverages[i]>=0.5 or conflicts[i]>=0.5:
                continue

            self.valid_concept_rules.append(r)
            self.valid_idx2concept_rule[ r['index'] ] = r

        print(f"  ###=== Selected ##Valid Concept Rules size : {len(self.valid_concept_rules)} ")






