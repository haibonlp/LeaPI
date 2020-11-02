# coding:utf-8

import argparse

class ArgumentParser:
  def __init__(self):

    self.max_iter = 0

    self.randomSeed = 111


    self.eventdictfpath = "../resources/event-dict.gz"
    self.word2vecfpath = "../resources/GoogleNews-vectors-negative300.bin"
    self.hneed_gold_data_fprefix = "../resources/human-needs-data/test-data/"
    self.hneed_dev_data_fpath = "../resources/human-needs-data/dev-data.json"
    self.event_vec_fpath = "../resources/event-vecs.bin"
    self.concept_cand_fpath = "../resources/concept_candidate_simK1_expK0.json"


    self.outprefix = "./tmpdata/predict-tags"

    self.featlist = [ 'AvgVec' ]

    self.stateType = 'CL'  ## CL, CLH
    self.debug = 0

    self.parse_arguments()


  def parse_arguments(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--rl-max-epoch', type=int, default=1)
    parser.add_argument('--rl-sample-times', type=int, default=3)
    parser.add_argument('--rl-agent-lr', type=float, default=1e-3)
    parser.add_argument('--max-concepts', type=int, default=3)
    parser.add_argument('--candfpath', type=str, default=self.concept_cand_fpath)
    parser.add_argument('--state-type', type=str, default='CL')
    parser.add_argument('--state-feat', type=str, default='prodlabel', choices=['avglabel', 'difflabel', 'prodlabel'])
    parser.add_argument('--random-mu', type=float, default=0)
    parser.add_argument('--none-size', type=int, default=300)

    parser.add_argument('--rl-init', type=str, default='default', choices=['default', 'normal0.01', 'normal0.1'])
    parser.add_argument('--wvfpath', type=str, default=self.word2vecfpath)


    args = parser.parse_args()

    self.args = args
    print(args)


    self.randomSeed = args.seed
    self.rl_max_epoch = args.rl_max_epoch
    self.rl_sample_times = args.rl_sample_times
    self.rl_agent_lr = args.rl_agent_lr
    self.rl_max_concepts = args.max_concepts
    #self.buffer_max_size = args.buffer_max_size
    #self.buffer_sample_size = args.buffer_sample_size
    #self.buffer_start_size = args.buffer_start_size
    self.concept_cand_fpath = args.candfpath
    self.stateType = args.state_type
    self.state_feat = args.state_feat
    self.random_mu = args.random_mu
    self.none_size = args.none_size
    self.rl_init = args.rl_init
    self.word2vecfpath = args.wvfpath




  def print_all_parameters(self, arg):
    attrs = vars(arg)
    print('##'*20)
    print('\n###########parameters##########')
    for item in attrs.items():
      print( "  --  %s: %s" % item  )
    print('##'*20, '\n\n', flush=True)





