
from snorkel_labeler.lfs.lf_generator_by_candidate_concepts import LFGeneratorByCandidateConcepts
#from snorkel_labeler.lfs.lf_generator_by_specific_rules import LFGeneratorBySpecificRules
#from snorkel_labeler.lfs.lf_generator_by_concepts import LFGeneratorByConcepts


class LFGenerator:
    def __init__(self, arg):
        self.arg = arg
        self.lfs = []

    def add_lf(self, lf):
        self.lfs.append(lf)


    def generate_lfs(self):

        self.generate_lfs_by_candidate_concepts()

        return self.lfs

    def generate_lfs_by_candidate_concepts(self):
        generator = LFGeneratorByCandidateConcepts(self.arg)
        lfs = generator.generate_lfs()
        self.lfs += lfs











