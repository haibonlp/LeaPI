# coding:utf-8


from collections import defaultdict
from argparser import ArgumentParser
from resource_manager import ResourceManager
from weak_labeler import WeakLabeler
from label_aggregator import LabelAggregator
from human_need_classifier import HumanNeedClassifier

from concepts.concept_embedder import ConceptEmbedder
from concepts.concept_rule_data import ConceptRuleData

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import random, sys, os
import time
import json
import numpy as np
import heapq


import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



class ConceptEnvironment:
    def __init__(self, arg,  crdata, concept_embedder):
        self.arg = arg
        self.candlist = crdata.valid_concept_rules
        self.concept_embedder = concept_embedder

        self.cand_concept_embeds = concept_embedder.get_cand_concept_rule_embs(self.candlist, crdata)

        w2v = self.arg.resource.get_word2vec_data()
        wvdim = w2v.vector_size


        self.concept_emb_dim = wvdim

        ## state:  (1) selected-concepts-embeds-mean, (2) current-concept-embed
        # (3) the label embed of the current concept

        if self.arg.stateType == 'CL':
            self.state_emb_size = wvdim
        elif self.arg.stateType == 'CLH':
            #self.state_emb_size = 50 + 50
            self.state_emb_size = wvdim + wvdim


    def reset(self):
        self.selected_num = 0
        self.selected_idx = []
        self.selected_concept_mean =  np.zeros(self.concept_emb_dim)


    def take_action(self, a, curidx):
        if a == 1:
            featsum = self.selected_num*self.selected_concept_mean + self.cand_concept_embeds[curidx]
            self.selected_concept_mean = featsum/float(self.selected_num +1)
            self.selected_idx.append(curidx)
            self.selected_num += 1
        #print(f" #take-action: select_num : {self.selected_num}  {len(self.selected_idx)}" )



    def get_cur_state(self, idx):

        cand = self.candlist[idx]
        cur_concept_emb = self.cand_concept_embeds[idx]
        cur_label_emb = self.concept_embedder.get_concept_label_emb(cand)

        if self.arg.state_feat == 'avglabel':
            state = np.mean([cur_concept_emb, cur_label_emb], axis=0 ) ## add label embs
        elif self.arg.state_feat == 'difflabel':
            state = cur_concept_emb - cur_label_emb
        elif self.arg.state_feat == 'prodlabel':
            state = cur_concept_emb * cur_label_emb



        if self.arg.stateType == 'CLH':
            state = np.concatenate( [state,  self.selected_concept_mean ] )
        elif self.arg.stateType == 'CL':
            #state = np.concatenate( [state] )
            pass


        state = torch.tensor(state).float()
        return state


class Experience():
    def __init__(self, reward, selected_concepts):
        self.reward = reward
        self.selected_concept_cands = selected_concepts

    def __lt__(self, other):
        return (self.reward, str(self.selected_concept_cands) ) < (other.reward, str(self.selected_concept_cands))

    def __eq__(self, other):
        return (self.reward, str(self.selected_concept_cands) ) == (other.reward, str(self.selected_concept_cands))

    def __gt__(self, other):
        return (self.reward, str(self.selected_concept_cands) ) > (other.reward, str(self.selected_concept_cands))



class MemoryBuffer():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.itemlist = []

    def put(self, item):
        #item = (reward, exp)
        heapq.heappush(self.itemlist, item)
        if len(self.itemlist) > self.maxsize:
            heapq.heappop(self.itemlist)

    def __len__(self):
        return len(self.itemlist)

    def size(self):
        return len(self.itemlist)

    def sample(self, size):
        size = min(size, self.size())
        return random.sample(self.itemlist, size)




"""
class PolicyNet(nn.Module):
    def __init__(self, in_dim=50):
        super().__init__()

        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x
"""


class PolicyNetV2(nn.Module):
    def __init__(self, in_dim=50, hidden_size=32, num_actions=2, arg=None):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

        if arg.rl_init == 'normal0.01':
            nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
            nn.init.normal_(self.linear1.bias, mean=0, std=0.01)

            nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
            nn.init.normal_(self.linear2.bias, mean=0, std=0.01)
        elif arg.rl_init == 'normal0.1':
            nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
            nn.init.normal_(self.linear1.bias, mean=0, std=0.1)

            nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
            nn.init.normal_(self.linear2.bias, mean=0, std=0.1)


    def forward(self, x):
        x = F.relu( self.linear1(x) )
        x = F.softmax(self.linear2(x), dim=-1)

        return x





class RLConceptLearner:
    def __init__(self, arg):
        self.arg = arg
        self.resource = ResourceManager(self.arg)

        self.arg.resource = self.resource
        self.weakLabeler = WeakLabeler(self.arg )
        self.labelAggregator  = LabelAggregator(self.arg)




    def get_rewards_on_test(self, scoredict):
        #print(f"==Test_scoredict : {scoredict}")
        tag2prf = scoredict['tag2prf']
        avgF1 = scoredict['avgF']
        h2r = {}
        for h in self.class_names:
            p, r, f = tag2prf[h]
            h2r[h] = f
        return h2r, avgF1

    def get_rewards_on_dev(self, scoredict):
        #print(f"==Dev_scoredict : {scoredict}")
        tag2prf = scoredict['tag2prf']
        avgF1 = scoredict['macroF1']
        h2r = {}
        for h in self.class_names:
            p, r, f = tag2prf[h]
            h2r[h] = f
        return h2r, avgF1


    def init_estimate_hneed_performance(self):
        '''
            run prediction system with all potential concepts
        '''
        self.weakLabeler.main_collect_weak_labels()

        self.labelAggregator.main_aggregate_labels()
        self.humanNeedClassifier =  HumanNeedClassifier(self.arg)
        dev_scoredict, test_scoredict = self.humanNeedClassifier.main_evaluate_hneed_classifier(evalTest=True)

        #print(f"dev_scoredict: {dev_scoredict}")
        #print(f"test_scoredict: {test_scoredict}")

        dev_h2r, dev_avgF1 = self.get_rewards_on_dev(dev_scoredict)
        test_h2r, test_avgF1 = self.get_rewards_on_test(test_scoredict)


        return dev_h2r, dev_avgF1, test_h2r, test_avgF1



    def estimate_hneed_performance(self):
        t0 = time.time()
        self.labelAggregator.main_aggregate_labels_selected()
        t1 = time.time()
        self.humanNeedClassifier =  HumanNeedClassifier(self.arg)
        dev_scoredict, test_scoredict = self.humanNeedClassifier.main_evaluate_hneed_classifier(evalTest=True)
        #print(f"#timeClf : {time.time()-t1}")


        dev_h2r, dev_avgF1 = self.get_rewards_on_dev(dev_scoredict)
        test_h2r, test_avgF1 = self.get_rewards_on_test(test_scoredict)


        return dev_h2r, dev_avgF1, test_h2r, test_avgF1


    def select_action_with_exploration(self, state, agent):
        probs = agent(state)
        m = Categorical(probs)

        ## add random noisy to avoid local minimum
        m1 = Categorical(self.arg.random_mu*torch.ones(2) + probs )
        action = m1.sample()

        logprob = m.log_prob(action)
        prob = m.probs[action]
        #print(f"logprob :  {logprob:.5f}   requries_grad: {logprob.requires_grad}  prob: {m.probs[action]:.5f}  act: {action} ")
        return action.item(), logprob, prob

    def select_action(self, state, agent):
        probs = agent(state)
        p, a = torch.max(probs, dim=-1)
        #print(f"select, act: {a}  prob: {p:.5f} ")
        return a.item(), p


    def choose_concepts_with_exploration(self, agent, env):
        acts = []
        selected_concepts = []
        log_probs = []
        cand_num = len(env.candlist)

        for idx in range(cand_num):
        #for idx in np.random.permutation(cand_num):
            cand = env.candlist[idx]
            state = env.get_cur_state(idx)
            a, logprob, prob = self.select_action_with_exploration(state, agent)
            #print(f" explore act : {a}  logprob: {logprob:.5f}  prob:{prob:.5f} {cand['label']} {cand['index']} {cand['concept_rule']} ")
            acts.append(a)
            log_probs.append(logprob)
            if a==1:
                selected_concepts.append( cand )
            env.take_action(a, idx)
        return acts, log_probs, selected_concepts





    def choose_concepts(self, agent, env):
        acts = []
        selected_cands = []
        log_probs = []
        cand_num = len(env.candlist)
        for idx in range(cand_num):
            state = env.get_cur_state(idx)
            a, logprob = self.select_action(state, agent)

            cand = env.candlist[idx]
            #print(f"select,   act: {a}  prob: {logprob:.5f}  '{cand['label']:<12}  {cand['index']}' {cand['concept_rule']} ")
            acts.append(a)
            log_probs.append(logprob)
            if a==1:
                selected_cands.append( env.candlist[idx] )
            env.take_action(a, idx)

        return acts, log_probs, selected_cands




    def save_selected_concepts(self, h2selconcepts, selected_concepts):
        self.conceptsetdata.selected_concepts = selected_concepts




    def update_policy(self, reward, acts, agent, optimizer, log_probs):
        num = len(log_probs)
        rewardlist = [reward]*num

        log_probs = torch.stack(log_probs)
        rewards = torch.tensor( rewardlist ).float()
        policy_loss = -(log_probs*rewards).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        loss = policy_loss.item()
        #print(f" h2loss : {h2loss}")
        return loss


    def adjust_rewards_avg(self, sample_data, ep):
        N = len(sample_data)
        avgR = []
        for t, d in enumerate(sample_data):
            #print(f" ###Ep:{ep:<3} t={t} r before F1: {d['avgF1']} ")
            avgR.append( d['avgF1'] )
        avgR = np.mean(avgR)
        #print(f" ###Ep:{ep:<3} avg reward after adjust average F1 : {avgR} ")
        for t, d in enumerate(sample_data):
            d['avgF1'] = d['avgF1'] - avgR
            #print(f" ###Ep:{ep:<3} t={t} r adjust : {d['avgF1']} ")
        return sample_data


    def adjust_rewards_baseline(self, sample_data, base_avgF1):

        for t, d in enumerate(sample_data):
            h2r = d['h2r']
            #strh2r= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.5f} " for l in h2r.items() ] )
            print(f" ### t={t} r before: avgF1:{d['avgF1']:<.3f} ")

        #strh2r= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.5f} " for l in base_h2r.items() ] )
        print(f" ### base-reward, avgF1:{base_avgF1:<.3f}  ")
        for t, d in enumerate(sample_data):
            h2r = d['h2r']
            d['avgF1'] = d['avgF1'] - base_avgF1
            print(f" ### t={t} r adjust, avgF1:{d['avgF1']} ")
        return sample_data




    def get_epoch_loss(self, list_loss, ep):
        h2avgloss = {}
        for h2loss in list_loss:
            for h in h2loss:
                if h not in h2avgloss:
                    h2avgloss[h] = 0
                h2avgloss[h] += h2loss[h]
        N = len(list_loss)
        avgloss = 0
        for h in h2avgloss:
            h2avgloss[h] = h2avgloss[h]/float(N)
            avgloss += h2avgloss[h]
        avgloss = avgloss/len(h2avgloss)
        strh2avgloss= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.4f} " for l in h2avgloss.items() ] )
        print(f" ##{ep:<3} Epoch AvgLoss:{avgloss:3.3f} {strh2avgloss} ")

        return avgloss

    def run_all_concept_candidates(self ):

        dev_h2r, dev_avgF1, test_h2r, test_avgF1 = self.init_estimate_hneed_performance()
        dev_strh2r= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.4f} " for l in dev_h2r.items() ] )
        print(f"AllConcepts #Baseline Dev ,avgF1:{dev_avgF1:.3f}    {dev_strh2r}")
        test_strh2r= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.4f} " for l in test_h2r.items() ] )
        print(f"AllConcepts #Baseline Test,avgF1:{test_avgF1:.3f}   {test_strh2r}")
        return dev_h2r, dev_avgF1, test_h2r, test_avgF1


    def add_sample_data_from_buffer(self, sample_data, mbuffer, agent, env):
        sample_list = mbuffer.sample(self.arg.buffer_sample_size)
        for sample_item in sample_list:
            env.reset()
            reward, selected_concept_cands = sample_item.reward, sample_item.selected_concept_cands
            selected_concept_idxs = set([r['index'] for r in selected_concept_cands]  )

            acts = []
            log_probs = []
            for i in range( len(env.candlist) ):
                cand = env.candlist[i]
                state = env.get_cur_state(i)
                a = 0
                if cand['index']  in selected_concept_idxs:
                    a = 1
                acts.append(a)
                probs = agent(state)
                m = Categorical(probs)
                logprob = m.log_prob( torch.tensor(a) )
                log_probs.append(logprob)

                env.take_action(a, i)

            spdata = {
                       # 'h2r': dev_h2r,
                        'avgF1': reward,
                        'acts': acts,
                       # 'selected_concepts': selected_concepts,
                        'log_probs' : log_probs
                        }
            sample_data.append( spdata )



    def main_rl_concept_selection(self):

        t0 = time.time()
        base_dev_h2r, base_dev_avgF1, base_test_h2r, base_test_avgF1 = self.run_all_concept_candidates()
        #print(f"###run all concepts time## : { time.time() - t0 } ")
        #sys.exit()

        concept_embedder = ConceptEmbedder(self.arg)
        env = ConceptEnvironment(self.arg, self.crdata, concept_embedder)
        env.class_names = self.class_names

        num_epoch =  self.arg.rl_max_epoch #1
        #learning_rate = 0.01
        learning_rate = self.arg.rl_agent_lr  #1e-3
        sample_times =  self.arg.rl_sample_times #30

        #agent = PolicyNet(in_dim=env.state_emb_size)
        agent = PolicyNetV2(in_dim=env.state_emb_size, arg=self.arg)
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

        #self.mbuffer = MemoryBuffer(self.arg.buffer_max_size)
        self.expcache = {}

        epoch_rewards = []
        epoch_sample_dev_rewards = []
        epoch_selected_concepts = []

        pre_dev_avgF1 = 0

        for ep in range(num_epoch):

            t0 = time.time()
            sample_data = []
            for t in range(sample_times):
                env.reset()

                t1 = time.time()
                acts, log_probs, selected_concepts = self.choose_concepts_with_exploration(agent, env)
                #print(f"##time, Sample chose concept : { time.time()-t1}")
                self.crdata.selected_concept_rules = selected_concepts

                #print(f"- t={t} selconcepts: {h2selconcepts}")
                t2 = time.time()

                if tuple(acts) in self.expcache:
                    dev_h2r, dev_avgF1, test_h2r, test_avgF1 = self.expcache[ tuple(acts) ]
                else:
                    dev_h2r, dev_avgF1, test_h2r, test_avgF1 = self.estimate_hneed_performance()
                    self.expcache[ tuple(acts) ] = (dev_h2r, dev_avgF1, test_h2r, test_avgF1)

                #print(f"##time, Sample estimate performance : {time.time()-t2}")

                dev_strh2r= ' '.join( [ f"{l[0][:5]:<6} {l[1]:3.4f} " for l in dev_h2r.items() ] )
                #print(f" Dev #Epoch:{ep:<3} t:{t:<3} ##h2r, avgF1:{dev_avgF1:.3f}  {dev_strh2r}")
                #print(f" Dev #Epoch:{ep:<3} sample-t:{t:<3}  avgF1:{dev_avgF1:.3f} ")
                spdata = {
                        'h2r': dev_h2r,
                        'avgF1': dev_avgF1,
                        'acts': acts,
                        'selected_concepts': selected_concepts,
                        'log_probs' : log_probs
                        }
                sample_data.append( spdata )

                #print(f"\n\n")

            #print(f"##time, sampledata : {time.time()-t0}")


            sample_mean_dev_avgF1 = np.mean( [sample['avgF1'] for sample in sample_data ]  )

            epoch_sample_dev_rewards.append( {'ep': ep, 'sample_dev_f1': [ f"{sample['avgF1']:.3f}" for sample in sample_data ]} )


            t20 = time.time()
            sample_data = self.adjust_rewards_avg( sample_data, ep )
            list_loss = []
            #print(f"#updateNum : {len(sample_data)}")
            for t, d in enumerate(sample_data):
                acts, log_probs = d['acts'], d['log_probs']
                reward = d['avgF1']
                loss = self.update_policy(reward, acts, agent, optimizer, log_probs)
                list_loss.append(loss)

            #print(f"##time,update : {time.time()-t20}")

            #############################
            #####   eval on dev set
            env.reset()
            acts, log_probs, selected_cands = self.choose_concepts(agent, env)

            if len(selected_cands) < 3:
                print(f"########Epoch:{ep:<3}  DevAvgF1:{0:.3f} Error! selected_cands:{len(selected_cands)}\n\n")
                continue

            self.crdata.selected_concept_rules = selected_cands

            dev_h2r, dev_avgF1, test_h2r, test_avgF1 = self.estimate_hneed_performance()

            pre_dev_avgF1 = dev_avgF1

            ep_reward = [ ep,  f"{dev_avgF1:.3f}", f"{test_avgF1:.3f}"  ]
            epoch_rewards.append(ep_reward )
            selected_cands_idx = [ r['index'] for r in selected_cands]
            epoch_selected_concepts.append( (ep, selected_cands_idx) )

            #print(f"##time,epoch{ep}: {time.time()-t0}")
            print(f"########Epoch:{ep:<3} Selected:{len(selected_cands):<3}  DevAvgF1: {dev_avgF1:.3f}   TestAvgF1: {test_avgF1:.3f} ", flush=True)


        # ret1 = {'setting': str(self.arg.args), 'scores': epoch_rewards }
        # ret2 = {'setting': str(self.arg.args), 'all_cands': env.candlist,  'selected_concepts': epoch_selected_concepts}
        # ret3 = {'setting': str(self.arg.args), 'sample_dev_scores': epoch_sample_dev_rewards}
        # print(f"\n### Epoch ##Ep-sample-dev Scores: \n")
        # print(json.dumps(ret3))
        # print(f"\n### Epoch ##EPScores: \n")
        # print(json.dumps(ret1) )
        # print(f"\n### Epoch ##EPSelectedConcepts: \n")
        # print(json.dumps(ret2))


        print(f"\n\n ####### Results for Each Epoch ####### \n")
        bestDevf1 = 0
        bestTestf1 =0
        bestEp = 0
        for ep, devf1, testf1 in epoch_rewards:
            devf1 = float(devf1)
            testf1 = float(testf1)
            print(f"#Epoch: {ep:<3} Reward, Dev: {devf1:<.3f}  Test: {testf1:<.3f}")
            if devf1 > bestDevf1:
                bestDevf1 = devf1
                bestTestf1 = testf1
                bestEp = ep
            if (ep+1)%50 == 0:
                print(f" -------------------------- Till Epoch: {ep}, bestDevf1: {bestDevf1} bestTestf1:{bestTestf1}  bestEp:{bestEp}")

        print(f"\n######### bestDevf1: {bestDevf1:<.3f} bestTestf1:{bestTestf1:<.3f}  bestEp:{bestEp}")







    def start_rlconcept(self):
        self.class_names = [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental', 'None' ]
        self.arg.class_names = self.class_names


        self.crdata = ConceptRuleData(self.arg)
        self.arg.crdata = self.crdata

        self.main_rl_concept_selection()





def init_random_seed(arg):
    print(f"init random seed : {arg.randomSeed} ")
    os.environ['PYTHONHASHSEED']=str(arg.randomSeed)
    random.seed( arg.randomSeed)
    np.random.seed(arg.randomSeed)
    torch.manual_seed(arg.randomSeed)
    ## cuda related
    torch.cuda.manual_seed_all(arg.randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main_rl_learn():

    arg = ArgumentParser()
    arg.print_all_parameters(arg)

    init_random_seed(arg)

    learner = RLConceptLearner(arg)

    learner.start_rlconcept()



if __name__ == '__main__':

    main_rl_learn()


