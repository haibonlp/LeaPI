import json
from snorkel.labeling import LabelingFunction
from snorkel_labeler import hneedlabels

class LFGeneratorByCandidateConcepts:
    def __init__(self, arg):
        self.arg = arg
        self.lfs = []

    def add_lf(self, lf):
        self.lfs.append(lf)


    def generate_lfs(self):
        for cr in self.arg.crdata.concept_rule_candidates:
            lf = self.make_lf_from_concept_rule(cr)
            lf.concept_rule = cr

            self.add_lf(lf)

        return self.lfs



    def make_lf_from_concept_rule(self, concept_rule):

        #lf = make_lf_event_contain_concept(concept)
        lf = make_lf_event_concept_rule(concept_rule)

        return lf


def make_lf_event_concept_rule(rule):
    label = rule['label']
    rule_name = ', '.join( [ f"{r['op']}_{r['concept']}" for r in rule['concept_rule'] ] )
    lfname = f"LF_{label} <= '{rule_name}'"

    lf = LabelingFunction(name=lfname, f=op_event_satisfy_concept_rule, resources=dict(label=label, conditions=rule['concept_rule']) )

    return lf


def op_event_satisfy_concept_rule(x, label, conditions):
    op2concept2freq = {}
    for r in conditions:
        op = r['op']
        concept = r['concept']
        if op not in op2concept2freq:
            op2concept2freq[op] = {}
        if concept not in op2concept2freq[op]:
            op2concept2freq[op][concept] = 0
        op2concept2freq[op][concept] += 1

    for op, concept2freq in op2concept2freq.items():
        if not event_satisfy_condition(x, op, concept2freq):
            return -1
    labelid = hneedlabels.get_id_by_label(label)
    return labelid


def event_satisfy_condition(x, op, concept2freq):
    assert (op in {'has', 'only has', 'not has'})
    if op == 'has':
        return satisfy_condition_has(x, concept2freq)
    elif op == 'only has':
        return satisfy_condition_onlyhas(x, concept2freq)
    elif op == 'not has':
        return satisfy_condition_nothas(x, concept2freq)
    else:
        raise Exception(f"Error! Error condition : {op} {concept2freq}")


def satisfy_condition_has(x, concept2freq):
    c2f = {}
    fieldnamelist = ['Agent', 'Predicate', 'Theme', 'PP']
    for field in fieldnamelist:
        for c in x.field2concepts[field]:
            if c not in c2f:
                c2f[c] = 0
            c2f[c] += 1
    if set( concept2freq.items() ).issubset( set(c2f.items())  ):
        return True
    return False


def satisfy_condition_onlyhas(x, concept2freq):
    fieldnamelist = ['Agent', 'Predicate', 'Theme', 'PP']
    c2f = {}
    for fieldname in fieldnamelist:
        text = x.get_value_by_field(fieldname)
        #if text.startswith("@fp"):  # if is FirstPerson, skip 1st person
        #if len(text.split())==1 and text.startswith("@"):  # if is pronouns, skip pronouns
        if text.startswith("@"):  # if is pronouns, skip pronouns
            continue
        for c in x.field2concepts[fieldname]:
            if c not in c2f:
                c2f[c] = 0
            c2f[c] += 1
    if concept2freq == c2f:
        return True
    return False



def satisfy_condition_nothas(x, r):
    concept = r['concept']
    fieldnamelist = ['Agent', 'Predicate', 'Theme', 'PP']
    for field in fieldnamelist:
        if concept in x.field2concepts[field]:
            return False
    return True




def make_lf_event_contain_concept(concept):
    concept_name = concept['concept']
    label = concept['label']
    concept_insts = set(concept['instances'])

    lfname= f"LF_Contain_'{concept_name}'->{label}"
    lf = LabelingFunction(name=lfname, f=op_event_Contain_concept_instance, resources=dict(instset=concept_insts, label=label) )

    return lf


def op_event_Contain_concept_instance(x, instset, label):
    labelid = hneedlabels.get_id_by_label(label)
    fieldnamelist = ['Agent', 'Predicate', 'Theme', 'PP']
    for fieldname in fieldnamelist:
        text = x.get_value_by_field(fieldname)
        if text in instset:
            return labelid
    return -1





