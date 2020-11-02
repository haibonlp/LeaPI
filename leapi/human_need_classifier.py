# coding:utf-8

import util
#from data.human_need_gold_data import HumanNeedGoldData
from classifiers.sklearn_classifier import SKLearnClassifier
from featextractors.event_feature_extractor import EventFeatureExtractor


class HumanNeedClassifier(object):
    def __init__(self, arg):
        self.arg = arg
        self.resource = self.arg.resource
        self.hneedGoldData = self.resource.get_human_need_gold_data()
        self.eventFeatExtractor = EventFeatureExtractor(self.arg)


    def train_classifier_with_noisy_data(self, clf):
        labeled_event_pool = self.resource.get_labeled_event_pool('Lsnorkel')

        train_feats = []
        train_labelIds = []
        traindata = labeled_event_pool.prepare_train_set()
        for inst in traindata:
            event = inst['event']
            label = inst['label']

            feat2val = self.eventFeatExtractor.extract_event_features(event)
            labelId = self.hneedGoldData.get_labelId_by_label(label)

            train_feats.append(feat2val)
            train_labelIds.append(labelId)

        clf.train(train_feats, train_labelIds)


    def predict_on_test_set(self, clf):
        testlist = self.hneedGoldData.get_test_set()
        test_feats = []
        for event in testlist:
          feat2val = self.eventFeatExtractor.extract_event_features(event)
          test_feats.append(feat2val)
        test_predY = clf.predict(test_feats)

        eval_result = self.evaluate_predictions(testlist, test_predY)
        return eval_result



    def evaluate_predictions(self, testlist, testPredY):
        gold_evid2tag = {}
        sys_evid2tag = {}
        taglist = self.hneedGoldData.get_class_names()
        sysTagList = []
        goldTagList = []
        for i, ev in enumerate(testlist):
            gold = ev.htag
            systid = testPredY[i]
            sysTag = self.hneedGoldData.get_label_from_id(systid)
            gold_evid2tag[ ev.evid ] = gold
            sys_evid2tag[ ev.evid ] = sysTag

            sysTagList.append( sysTag )
            goldTagList.append( gold)


            if self.arg.debug> 2 and gold != sysTag:
                print(f"Error, Gold: {gold:<15}  Predict: {sysTag:<15} EventId: {ev.evid:>10}  Event: {ev.ev4field}")

        #print( '\n #######testfpath : ', self.arg.testfpath )
        scoredict = util.compute_performance(taglist, gold_evid2tag, sys_evid2tag)
        tag2prf = scoredict['tag2prf']

        #confmatrix = None
        confmatrix = util.get_confusion_matrix(sysTagList, goldTagList)

        evid2tag = (gold_evid2tag, sys_evid2tag )
        testPerformance = [ tag2prf, confmatrix, evid2tag, scoredict ]

        return testPerformance




    def cross_validation_experiments(self):
        '''
        Here, cross validation is to predict the test set.
             but the classifier will only be trained one time.
        '''
        classifier_model = SKLearnClassifier()

        self.train_classifier_with_noisy_data(classifier_model)

        foldId2testPerformance = {}
        for foldId in range(0, 3):
            foldId += 1

            self.hneedGoldData.load_train_test_by_foldId(foldId)

            testPerformance = self.predict_on_test_set(classifier_model)

            foldId2testPerformance[foldId] = testPerformance

            #break

        #arg.print_arguments()
        score = util.output_test_cross_val_performance( foldId2testPerformance )
        return score




    def main_train_test_hneed_classifier(self):

        score = self.cross_validation_experiments()
        return score



    def eval_on_test_set(self, clf):
        foldId2testPerformance = {}
        for foldId in range(0, 3):
            foldId += 1

            self.hneedGoldData.load_train_test_by_foldId(foldId)

            testPerformance = self.predict_on_test_set(clf)

            foldId2testPerformance[foldId] = testPerformance

            #break
        #arg.print_arguments()
        score = util.output_test_cross_val_performance( foldId2testPerformance, 0 )
        return score


    def eval_on_dev_set(self, clf):
        devlist = self.hneedGoldData.load_dev_data()
        dev_feats = []
        for event in devlist:
            feat2val = self.eventFeatExtractor.extract_event_features(event)
            dev_feats.append(feat2val)
        dev_predY = clf.predict(dev_feats)

        eval_result = self.evaluate_predictions(devlist, dev_predY)
        #[ tag2prf, confmatrix, evid2tag, scoredict ] = eval_result
        dev_score = eval_result[-1]

        return dev_score




    def main_evaluate_hneed_classifier(self, evalTest=False):
        classifier_model = SKLearnClassifier()
        self.train_classifier_with_noisy_data(classifier_model)

        dev_score = self.eval_on_dev_set(classifier_model)

        test_score = None
        if evalTest:
            test_score = self.eval_on_test_set( classifier_model )

        return dev_score, test_score







