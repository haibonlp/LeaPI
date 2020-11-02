
import json
import numpy as np
from snorkel_labeler import hneedlabels
from weaklabeldata.none_selector import NoneClassSelector


class AutoLabeledEventsPoolSnorkel:
    def __init__(self, arg):
        self.arg = arg

        pass

    def set_init_label_matrix(self, L, data):
        self.init_L_train_data = data
        self.L_train_data = data

        L = self.remove_conflict_with_Mental(L)
        self.init_L_train_matrix = L
        self.L_train_matrix = L

    def set_init_label_data(self, data):
        self.init_L_train_data = data
        self.L_train_data = data


    def set_label_matrix(self, L_matrix):
        self.L_train_matrix = L_matrix

    def get_label_matrix(self):
        return self.L_train_matrix


    def remove_conflict_with_Mental(self, L):
        '''
            When an instance is labeled as both Mental and other classes.
            Then, Mental labels will be removed.
        '''
        mentalId= hneedlabels.get_id_by_label('Mental')
        N, M = L.shape
        for i in range(N):
            labelset = set()
            for j in range(M):
                if L[i,j] != -1:
                    labelset.add( L[i, j] )
            if len(labelset) > 1 and mentalId in labelset:
                #print(f"##remove mental label : {L[i]}")
                L[i][ L[i] == mentalId ] = -1
                #print(f"## after mental label : {L[i]}")
                #print(f"  # set {labelset},  {self.L_train_data[i].field2value}")

        return L





    def get_label_matrix_selected(self):
        #selected_concepts = self.arg.conceptsetdata.selected_concepts
        selected_concepts = self.arg.crdata.selected_concept_rules
        #print(f"## Selected Concepts size : {len(selected_concepts)}")
        selected_idx = []
        for concept in selected_concepts:
            idx = concept['index']
            selected_idx.append(idx)


        #selected_idx = np.array(selected_idx, dtype=int)

        #print(f"### Original L shape : {self.init_L_train_matrix.shape}   {type(self.init_L_train_matrix)}")
        #print(f"### Selected Idx :  {type(selected_idx)}  {selected_idx}")

        self.L_train_matrix = self.init_L_train_matrix[:, selected_idx]

        #print(f"## Original L shape : {self.init_L_train_matrix.shape}")

        #print(f"## Selected L shape : {self.L_train_matrix.shape}")

        return self.L_train_matrix



    def set_label_data(self, L_data):
        self.L_train_data = L_data

    def get_label_data(self):
        return self.L_train_data

    def set_weak_label_probs(self, L_probs):
        '''
        N * M matrix:
            N is the num of instances
            M is the num of classes
        '''
        self.L_train_probs = L_probs


    def show_info(self):
        N = self.L_train_matrix.shape[0]
        labeled = (self.L_train_matrix != -1).any(axis=1)
        num_labeled = labeled.sum()
        #print(f" num_labeled: {num_labeled}  total : {N},  ratio: {num_labeled/N: 0.3}")
        #print( f" labeled matrix :{labeled.shape} \n", labeled)

        labeled = [ ]
        for i in range(N):
            if (self.L_train_matrix[i] != -1).any():
                term = (i, self.L_train_matrix[i], self.L_train_data[i], self.L_train_probs[i])
                labeled.append(term )
        #print( labeled[:10])


    def show_train_label_examples(self):
        N = self.L_train_matrix.shape[0]
        num = 0
        for i in range(N):
            if (self.L_train_matrix[i] != -1).any():
                term = (i, self.L_train_matrix[i], self.L_train_data[i])
                print(term)
                num+=1
        print(f"==== totally labeled events : {num} !\n")

        pass



    def prepare_train_set(self):

        traindata = self.select_train_with_max_proba()

        none_selector = NoneClassSelector(self.arg, self.L_train_data, traindata)
        none_size = self.arg.none_size
        nonedata = none_selector.select_none_examples(size=none_size)
        traindata += nonedata

        if self.arg.debug > 2:
            self.save_noisy_train_set(traindata)
        #self.print_train_set_info(traindata)
        return traindata




    def print_train_set_info(self, traindata):
        print('\n', '##'*10, ' Initial Train Info ', '##'*10)
        label2freq = {}
        print(f"   Total train size : {len(traindata)} ")
        for inst in traindata:
            event = inst['event']
            label = inst['label']
            if label not in label2freq:
                label2freq[label] = 0
            label2freq[label] += 1

        #for label in label2freq:

        class_names = self.arg.class_names
        if 'None' not in class_names:
            class_names = self.arg.class_names + ['None']
        for label in class_names:
            if label not in label2freq:
                print(f'       label : {label:20}   num : 0')
            else:
                print(f'       label : {label:20}   num : {label2freq[label]}')

        print('\n')



    ######### Balance Train Data ##########

    def balance_train_set(self, traindata):
        label2insts = {}
        for inst in traindata:
            label = inst['label']
            if label not in label2insts:
                label2insts[label] = []
            label2insts[label].append(inst)

        retlist = []
        num = 500
        for label in label2insts:
            insts = label2insts[label]
            insts = sorted(insts, key=lambda x:x['score'], reverse=True)
            retlist += insts[:num]

        return retlist


    ########### train data selection ################3



    def select_train_with_max_proba(self):


        N = self.L_train_matrix.shape[0]
        labeled = (self.L_train_matrix != -1).any(axis=1)
        trainlist = []
        for i in range(N):
            if not labeled[i]:
                continue
            event = self.L_train_data[i]
            probas = self.L_train_probs[i]
            labelid = np.argmax(probas)
            score = probas[labelid]
            label = hneedlabels.get_label_by_id(labelid)

            inst = {}
            inst['event'] = event
            inst['label'] = label
            inst['score'] = score
            trainlist.append(inst)
        return trainlist





    def select_and_save_noisy_data(self):
        traindata = self.prepare_train_set()
        self.save_noisy_train_set(traindata)



    def save_noisy_train_set(self, traindata):
        label2insts = {}
        #print(f"   Total train size : {len(traindata)} ")
        for inst in traindata:
            event = inst['event']
            label = inst['label']
            if label not in label2insts:
                label2insts[label] = []
            label2insts[label].append(inst)
        outfpath = "./tmpdata/tmp_noisy_train_data.json"
        with open(outfpath, 'w') as OutFile:
            for label in label2insts:
                insts = label2insts[label]
                for inst in insts:
                    #evstr = str(inst['event'])
                    #score = str(inst.get('score', None))
                    #outstr = f"{label:15}  {score:10}  {evstr:30} "
                    event = inst['event']
                    score = inst.get('score', 0)
                    obj = {'id': event.evid, 'label': label, 'field2value': event.field2value, 'score': f"{score:<.3f}"}
                    json.dump(obj, OutFile)
                    OutFile.write("\n")



