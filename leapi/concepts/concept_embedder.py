
import numpy as np


class LabelEmbedder:
    def __init__(self, arg):
        self.class_names = [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental', 'None' ]
        self.arg = arg
        self.label2emb = {}
        self.embed_all_labels()

        #print(f"\n\n###LabelEmbeds : ")
        #print(self.label2emb)


    def get_label_emb(self, label_name):
        return self.label2emb[label_name]

    def embed_all_labels(self):
        self.embed_by_label_name()


    def embed_by_label_name(self):
        w2v = self.arg.resource.get_word2vec_data()
        for label in self.class_names:
            w = label.strip().lower()
            if w in w2v:
                self.label2emb[label] = w2v[w]







class ConceptEmbedder:
    def __init__(self, arg):
        self.arg = arg
        self.resource = arg.resource

        self.label_embedder = LabelEmbedder(arg)


    def get_avg_word_emb(self, concept):
        words = concept.split()
        w2v = self.resource.get_word2vec_data()
        vecs = []
        dim = w2v.vector_size
        for w in words:
            if w not in w2v:
                continue
            vecs.append( w2v[w] )

        if len(vecs) <=0:
            return None
        else:
            return np.mean(vecs, axis=0)




    def embed_concept(self, concept):

        emb = self.get_avg_word_emb(concept)

        return emb

    def get_cand_concept_embeds(self, candlist):
        embs = []
        for cand in candlist:
            concept = cand['concept']
            v = self.embed_concept(concept)
            embs.append(v)
        return embs



    def get_concept_label_emb(self, concept):
        label = concept['label']
        return self.label_embedder.get_label_emb(label)




    def embed_concept_by_its_insts(self, concept, c2insts):
        vecs = []
        for inst in c2insts[concept]:
            v = self.get_avg_word_emb(inst)
            vecs.append(v)

        w2v = self.resource.get_word2vec_data()
        dim = w2v.vector_size
        if len(vecs) <=0:
            return np.zeros(dim)
        else:
            return np.mean(vecs, axis=0)


    def embed_concept_by_its_text(self, concept):

        emb = self.get_avg_word_emb(concept)

        return emb



    def get_cand_concept_rule_embs(self, candlist, crdata):

        w2v = self.resource.get_word2vec_data()
        dim = w2v.vector_size

        cand_embs = []
        for cand in candlist:
            c_embs = []
            for r in cand['concept_rule']:
                c = r['concept']
                #v = self.embed_concept_by_its_insts(c, crdata.concept2instances)
                v = self.embed_concept_by_its_text(c)
                if v is not None:
                    c_embs.append(v)
            if len(c_embs) <=0:
                cv= np.zeros(dim)
            else:
                cv = np.mean(c_embs, axis=0)
            cand_embs.append( cv )

            #print(f"#cand-emb, mean: {np.mean(cv):.5f} std: {np.std(cv):.5f}  {cand['index']}  {r}   emb: {cv}")

        return cand_embs






