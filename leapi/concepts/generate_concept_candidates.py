import sys
import json
import numpy as np
import pprint

sys.path.append("../")
import util
from data.probase_db import ProbaseDB
from gensim.models import KeyedVectors
from collections import OrderedDict
from scipy.spatial import distance


def preprocess_guildeline():
    guideline_fpath = "../resources/human-needs-anno-guildelines/human-need-real-anno-guidelines-v3.json"
    with open(guideline_fpath) as InFile:
        class2guideline = json.load(InFile)

    outfpath = "./candidate_results/preprocessed-guidelines-v3.json"

    post_class2guide = {}
    from stanza.server import CoreNLPClient
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'depparse'], timeout=30000, memory='16G') as client:
        for cname in class2guideline:
          outdict = {}
          postguides = []
          for text in class2guideline[cname]['guidelines']:
            docj = client.annotate(text, output_format="json")
            #guide = {'orgtext': text, 'tokens': tokens, 'json': docj }
            guide = {'orgtext': text,  'json': docj }
            postguides.append( guide )
          outdict['guidelines'] = postguides  ###
          #examples = []
          #for exinfo in class2guideline[cname]['examples']:
          #  text = exinfo['explanation']
          #  docj = client.annotate(text, output_format="json")
          #  post_ex = { 'event': exinfo['event'],
          #      'explanation': {'orgtext': text,
          #        #'tokens': tokens,
          #        'json' : docj
          #        }  }
          #  examples.append( post_ex )
          #outdict['examples'] = examples

          post_class2guide[cname] = outdict

    with open(outfpath, 'w') as OutFile:
        json.dump(post_class2guide, OutFile, indent=1)


def load_word2vec(w2vfpath):
    word2vec =  util.load_gensim_word2vec_model(w2vfpath)
    return word2vec


class ProbaseConceptRetriever:
    def __init__(self, w2vfpath, probasefpath):
        self.probasefpath = probasefpath

        self.probaseDB = ProbaseDB()
        self.word2vec = load_word2vec(w2vfpath)

        self.probase_concepts, self.probase_concept_embs = self.load_probase_concepts()

        self.cache_keyword2concepts = {}
        self.cache_keyword2mainConcept = {}
        self.probase_concept_content_embs = None



    def collect_concepts(self):
        #fpath = "../../bigdata/probase/data-concept/data-concept-instance-relations.txt"
        fpath = self.probasefpath
        c2f = {}

        with open(fpath) as InFile:
            for line in InFile:
                terms = line.strip().split('\t')
                if len(terms) != 3:
                    continue
                c = terms[0]
                f = int(terms[2])
                if c not in c2f:
                    c2f[c] = 0
                c2f[c] += f

        outfpath = "../tmpdata/probase_concept_freq.json"

        with open(outfpath, 'w') as OutFile:
            for c, f in sorted(c2f.items(), key=lambda x:x[1], reverse=True):
                if f < 10:
                    break
                out = {'concept': c, 'freq': f}
                json.dump(out, OutFile)
                OutFile.write('\n')


    def load_probase_concepts(self):
        self.collect_concepts()
        fpath = "../tmpdata/probase_concept_freq.json"
        conceptlist = []
        concept_embs = []
        topK = 10000
        with open(fpath) as InFile:
            for i, line in enumerate(InFile):
                tmp = json.loads(line)
                concept = tmp['concept']
                freq = tmp['freq']

                emb = self.get_avg_words_emb(concept)
                if emb is None:
                    continue
                conceptlist.append(concept)
                concept_embs.append(emb)
                if len(conceptlist) >= topK:
                    break
        concept_embs = np.array(concept_embs)
        return conceptlist, concept_embs


    def get_avg_words_emb(self, wordstr):
        words = wordstr.strip().split()
        vecs = []
        for w in words:
            if w in self.word2vec:
                v = self.word2vec[w]
                vecs.append(v)
            else:  ## to make sure that each word is a common word
                return None
        emb = None
        if len(vecs)>0:
            emb = np.mean(vecs, axis=0)
        return emb


    def get_similar_probase_concepts(self, kw):

        emb = self.get_avg_words_emb(kw)
        if emb is None:
            return []

        topK = self.concept_name_sim_topK
        emb = np.array([emb])
        sims = 1- distance.cdist(emb, self.probase_concept_embs, metric="cosine")
        sims = list(sims[0])
        terms = zip(self.probase_concepts, sims )

        sim_concepts = []
        for t in sorted(terms, key=lambda x:x[1], reverse=True)[:topK]:
            if self.debug > 0:
                print(f"kw :  {kw:<20}  sim: {t}")
            sim_concepts.append(t[0])


        return sim_concepts


    def retrieve_probase_concepts(self, keywords):
        #kw2concepts = OrderedDict()
        keyconcepts = []
        for kw in keywords:
            #ws = kw.split()
            concepts = self.get_similar_probase_concepts(kw)
            if len(concepts) <=0:
                continue
            mainConcept = concepts[0]
            self.cache_keyword2mainConcept[ kw ] = mainConcept

            concepts = self.expand_concepts_by_contents(concepts)
            #kw2concepts[kw] = concepts

            self.cache_keyword2concepts[kw] = concepts

            keyconcepts += concepts


        keyconcepts = sorted( list(set( keyconcepts )) )
        #print(f"##keyconcepts probase")
        #print(keyconcepts)
        #pprint.pprint(keyconcepts)

        return keyconcepts


    def get_concept_content_embs(self ):
        topn = 10
        content_embs = KeyedVectors(self.word2vec.vector_size)
        for concept in self.probase_concepts:
            insts = self.probaseDB.get_topK_insts_by_concept(concept, topn)
            vecs = []
            for inst in insts:
                v = self.get_avg_words_emb(inst)
                if v is not None:
                    vecs.append(v)
            if len(vecs) >0:
                emb = np.mean(vecs, axis=0)
                content_embs.add(concept, emb)
        return content_embs




    def expand_concepts_by_contents(self, initset):
        if self.probase_concept_content_embs is None:
            self.probase_concept_content_embs = self.get_concept_content_embs()
        #print(f"self.probase_concept_content_embs : { self.probase_concept_content_embs.shape }")
        topK = self.concept_content_exp_topK

        if topK <=0 :
            return initset

        expset = set( initset )
        for c in initset:
            if c not in self.probase_concept_content_embs:
                continue
            ret = self.probase_concept_content_embs.similar_by_word(c, topn=topK)
            for sim_c, sim_score in ret:
                expset.add(sim_c)
                if self.debug > 0:
                    print(f"#-expand: {c:<15}  {sim_c:<20} {sim_score:.5f} ")

        if self.debug > 0:
            print(f"####Expand set: {expset}")
            #pprint.pprint(expset)
        return expset




class CandidateConceptGenerator:
    def __init__(self, w2vfpath, guidefpath, probasefpath):
        self.w2vfpath = w2vfpath
        self.guidefpath = guidefpath
        self.probasefpath = probasefpath

        self.retriever = None

        self.probaseDB = ProbaseDB()
        #self.load_stopwords()

        #self.word2vec = self.load_word2vec()
        ##self.wordvec_size = self.word2vec.vector_size

    def load_stopwords(self):
        fpath = "../resources/stopwords.txt"
        stopwords = set()
        with open(fpath) as f:
            for w in f:
                w = w.strip()
                if len(w) <=0:
                    continue
                stopwords.add(w)
        self.stopwords = stopwords
        print(f" ## total stopwords : {len(stopwords)}")

    def is_stopword(self, word):
        if word in self.stopwords:
            return True
        return False


    def build_dep_map(self, sent):
        rels = sent['enhancedPlusPlusDependencies']
        w2dep2w = {}
        for r in rels:
            dep = r['dep'].split(":")[0]
            gw = (r['governor']-1, r['governorGloss'])
            dw = (r['dependent']-1, r['dependentGloss'])
            #gw = r['governor']-1
            #dw = r['dependent']-1
            if gw not in w2dep2w:
                w2dep2w[gw] = {}
            gdep = ('g', dep)
            if gdep not in w2dep2w[gw]:
                w2dep2w[gw][ gdep ] = []
            w2dep2w[gw][ gdep ].append(dw)

            if dw not in  w2dep2w:
                w2dep2w[dw] = {}
            ddep = ('d', dep)
            if ddep not in w2dep2w[dw]:
                w2dep2w[dw][ ddep ] = []
            w2dep2w[dw][ ddep ].append(gw)

        #pprint.pprint(w2dep2w)
        return w2dep2w


    def get_compound_mod(self, depmap, dw):
        if dw not in depmap:
            return None
        gdep = ('g', 'compound')
        if gdep in depmap[dw]:
            return depmap[dw][gdep][0]
        return None


    def get_subj(self, depmap, idx, token, tokenlist):
        gw = (idx, token['word'])
        if gw in depmap:
            gdep = ('g', 'nsubj')
            if gdep in depmap[gw]:
                dwlist = depmap[gw][gdep]
                tmplist = []
                for dw in dwlist:
                    mod = self.get_compound_mod(depmap, dw)
                    #text = dw[1]
                    text = tokenlist[ dw[0] ]['lemma']
                    if mod :
                        #text = mod[1] + " " + text
                        modtext = tokenlist[ mod[0] ]['lemma']
                        text = modtext + ' ' + text

                    tmplist.append(text)
                return tmplist
        return [""]

    def get_obj(self, depmap, idx, token, tokenlist):
        gw = (idx, token['word'])
        gdep = ('g', 'obj')
        ldep = ('g', 'obl')
        if gw not in depmap:
            return ""
        if gdep in depmap[gw]:
            dwlist = depmap[gw][gdep]
            tmplist = []
            for dw in dwlist:
                mod = self.get_compound_mod(depmap, dw)
                #text = dw[1]
                text = tokenlist[ dw[0] ]['lemma']
                if mod :
                    #text = mod[1] + " " + text
                    modtext = tokenlist[ mod[0] ]['lemma']
                    text = modtext + ' ' + text
                tmplist.append(text)
            return tmplist
        elif ldep in depmap[gw]:
            dwlist = depmap[gw][ldep]
            tmplist = []
            for dw in dwlist:
                mod = self.get_compound_mod(depmap, dw)
                #text = dw[1]
                text = tokenlist[ dw[0] ]['lemma']
                if mod :
                    #text = mod[1] + " " + text
                    modtext = tokenlist[ mod[0] ]['lemma']
                    text = modtext + ' ' + text
                tmplist.append(text)
            return tmplist



        return [""]



    def generate_keyword_tups(self, desc):
        sents = desc['sentences']
        keyword_tups  = []
        for sent in sents:
            if self.debug > 1:
                print(f"")
                print( ' '.join( [w['word'] for w in sent['tokens'] ]) )
            tokens = sent['tokens']
            #for r in sent['enhancedDependencies']:
            depmap = self.build_dep_map( sent )
            if self.debug > 1:
                pprint.pprint(depmap)

            ktups = []
            for i, t in enumerate(tokens):
                if t['pos'].startswith('VB'):
                    subjlist = self.get_subj(depmap, i, t, tokens)
                    objlist = self.get_obj( depmap, i, t, tokens)
                    #pred = t['word']
                    pred = t['lemma']
                    for subj in subjlist:
                        for obj in objlist:
                            #tup = (subj, pred, obj)
                            tup = (subj, obj)
                            ktups.append(tup)

            #print(f" ktups: {ktups} ")
            #keywords += kwords
            keyword_tups += ktups

        if self.debug > 0:
            print(f"     ===  keyword_tups :  {keyword_tups}")
        return keyword_tups

    def extract_keyword_tups(self, guide, label):
        keyword_tups = []
        for desc in guide['guidelines']:
            descj = desc['json']
            keyword_tups += self.generate_keyword_tups(descj)

        tmp = OrderedDict()
        for kt in keyword_tups:
            tmp[kt] = 1

        ## add label tup
        label_tup = (label.lower(), '', '')
        tmp[label_tup] = 1

        keyword_tups = list( tmp.keys() )

        if self.debug > 0:
            print(f"\n  TTtag : {label:<20}  {keyword_tups} \n")
        return keyword_tups


    def get_keywords_from_tup(self, keyword_tups):
        keywords = set()
        for tup in keyword_tups:
            for kw in tup:
                kw = kw.strip()
                if len(kw) > 0:
                    keywords.add(kw)
        if self.debug > 0:
            print(f"\n  keywords :   {keywords} \n")
        return keywords

    def get_keyword_rules_from_tups(self, keyword_tups):

        keyword_rules = OrderedDict()
        for tup in keyword_tups:
            tup = [ kw.strip() for kw in tup ]
            #unigram
            for kw in tup:
                if len(kw) > 0:
                    r = tuple( [kw] )
                    keyword_rules[ r ] = 1
            #bigram
            for i, ki in enumerate(tup):
                if len(ki)<=0:
                    continue
                for kj in tup[i+1:]:
                    if len(kj)<=0:
                        continue
                    r = tuple( sorted( [ki, kj] ) )
                    keyword_rules[ r ] = 1
        keyword_rules = list( keyword_rules.keys() )
        #pprint.pprint( keyword_tups )
        #print(f"## keyword_rules : ")
        #pprint.pprint( keyword_rules )

        return keyword_rules


    def get_concept_rules(self, keyword_rules):
        rset = set()
        for kr in keyword_rules:
            # unigram
            if len(kr) == 1:
                kw = kr[0]
                if kw in self.retriever.cache_keyword2concepts:
                    for c in self.retriever.cache_keyword2concepts[kw]:
                        cr = tuple([c])
                        rset.add(cr)
            # bigram
            if len(kr) == 2:
                k1, k2 = kr[0], kr[1]
                if k1 in self.retriever.cache_keyword2mainConcept and k2 in self.retriever.cache_keyword2mainConcept:
                    c1 = self.retriever.cache_keyword2mainConcept[k1]
                    c2 = self.retriever.cache_keyword2mainConcept[k2]
                    cr = tuple( sorted([c1, c2]) )
                    rset.add(cr)
                #for c1 in self.retriever.cache_keyword2concepts[k1]:
                #    for c2 in self.retriever.cache_keyword2concepts[k2]:
                #        cr = tuple( sorted([c1, c2]) )
                #        rset.add(cr)
        return rset




    def generate_candidate_concepts(self, guide, label):
        keyword_tups = self.extract_keyword_tups(guide, label)

        keywords = self.get_keywords_from_tup(keyword_tups)

        if self.retriever is None:
            self.retriever = ProbaseConceptRetriever(self.w2vfpath, self.probasefpath)
        self.retriever.debug = self.debug

        self.retriever.concept_name_sim_topK = 1
        self.retriever.concept_content_exp_topK = 0

        keyconcepts = self.retriever.retrieve_probase_concepts(keywords)

        keyword_rules = self.get_keyword_rules_from_tups(keyword_tups)

        concept_rules = self.get_concept_rules( keyword_rules )


        return concept_rules




    def collect_candidates(self, label2crset):
        candlist = []
        all_concepts = set()
        for label in label2crset:
            if label in {'None'}:
                continue
            crset = label2crset[label]
            for cs in crset:
                cr = [ {'op': 'has', 'concept': c } for c in cs ]
                cand = {
                        'label' : label,
                        'concept_rule': cr
                        }
                candlist.append(cand)
                for c in cs:
                    all_concepts.add(c)
        return candlist, all_concepts

    def collect_concept_instances_from_probase(self):
        c2insts = {}
        topK = 20
        minFreq = 5
        for c in self.all_concepts:
            inst_freq = self.probaseDB.get_topK_insts_freq_by_concept(c, topK)
            insts = []
            for inst, freq in inst_freq:
                if int(freq) < minFreq:
                    break
                #print(inst, freq, end=" ")
                insts.append(inst)
            c2insts[c] = insts
        return c2insts




    def add_SemEval2015_sentiment_lexicon(self):

        fpath = self.semeval2015_sentiment_lex_fpath
        topK = 200
        all_words = []
        with open(fpath) as InFile:
            for line in InFile:
                term = json.loads(line)
                all_words.append(term)
        termlist = all_words[:topK] + all_words[-topK:]
        senti_words = []
        for term in termlist:
            if term['pos'][-1] in { 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}:
                senti_words.append( term['word'] )

        concept = 'tweet sentiment words'
        concept_rule = [ {'op': 'has', 'concept': concept}]
        cand = {
                'label' : 'Mental',
                'concept_rule': concept_rule,
                }

        self.concept2instances[concept] = senti_words
        self.candlist.append(cand)


    def add_sentiment_lexicon(self):
        self.add_SemEval2015_sentiment_lexicon()



    def save_concept_candidates(self, label2crset ):

        #candfpath = "./candidate_results/concept_candidates.auto.json"
        candfpath = f"../tmpdata/concept_candidate_simK{self.retriever.concept_name_sim_topK}_expK{self.retriever.concept_content_exp_topK}.json"

        with open(candfpath, 'w') as OutFile:
            for cand in self.candlist:
                json.dump(cand, OutFile)
                OutFile.write('\n')
            for c in self.concept2instances:
                insts = self.concept2instances[c]
                c2insts = {'concept2instances' : { 'concept': c, 'instances': insts} }
                json.dump(c2insts, OutFile)
                OutFile.write('\n')



    def main_generate(self):
        with open(self.guidefpath) as f:
            label2guide = json.load(f)
        label2crset = OrderedDict()
        for t in label2guide:
            print(f"\n\n ### Class : {t} ###")
            crset = self.generate_candidate_concepts(label2guide[t], t)
            label2crset[t] = crset


        self.candlist, self.all_concepts = self.collect_candidates(label2crset)
        self.concept2instances = self.collect_concept_instances_from_probase()

        self.add_sentiment_lexicon()

        self.save_concept_candidates( label2crset )




def generate_candidates():
    #w2vfpath = "/Users/haibo/mywork/data/embeddings/google/GoogleNews-vectors-negative300.bin"
    #guidefpath = "./candidate_results/preprocessed-guidelines-v3.json"
    #semeval2015_sentiment_lex_fpath = "../bigdata/sentiment-lexicons/SemEval2015-English-Twitter-Lexicon.pos.json"

    probasefpath = "../../resources/probase-data-concept-instance-relations.txt"
    guidefpath = "../../resources/human-needs-anno-guildelines/preprocessed-guidelines.json"
    w2vfpath   = "../../resources/google-vecs.bin"
    semeval2015_sentiment_lex_fpath = "../../resources/SemEval2015-English-Twitter-Lexicon.pos.json"


    g = CandidateConceptGenerator(w2vfpath=w2vfpath, guidefpath=guidefpath, probasefpath=probasefpath)
    g.semeval2015_sentiment_lex_fpath = semeval2015_sentiment_lex_fpath

    g.debug = 0

    g.main_generate()

if __name__ == '__main__':

    #preprocess_guildeline()
    generate_candidates()





