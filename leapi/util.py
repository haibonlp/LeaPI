# coding:utf-8

from gensim.models import KeyedVectors
from time import time


def load_gensim_word2vec_model(infpath):
  isBinary = True if infpath.endswith('.bin') else False
  print( '-- start loading gensim w2v model from file: ', infpath)
  t0 = time()
  w2v = KeyedVectors.load_word2vec_format(infpath, binary=isBinary, encoding='utf-8')

  #print 'word num : ', len(w2v.wv.vocab)
  print( '   loading complete! time :', (time()-t0) )
  return w2v



def compute_performance(taglist, gold_evid2tag, sys_evid2tag, debug=0):

  #num_correct = {'pos':0, 'neu':0, 'neg':0}
  #num_gold = {'pos':0, 'neu':0, 'neg':0}
  #num_sys = {'pos':0, 'neu':0, 'neg':0}

  num_correct = {}
  num_gold = {}
  num_sys = {}
  for tag in taglist:
    num_correct[tag] = 0
    num_gold[tag] = 0
    num_sys[tag] = 0



  total_num = 0
  eval_num = 0
  for evid in gold_evid2tag:
    total_num += 1
    if evid not in sys_evid2tag:
      continue
    eval_num += 1
    gt = gold_evid2tag[evid]
    st = sys_evid2tag[evid]
    num_gold[gt] += 1
    num_sys[st] += 1
    if gt == st:
      num_correct[gt] += 1
    else:
      if debug>0:
        print(' event : ', evid, ' \n \t\t gold : ', gt, ' sys: ', st )

  if debug > 0:
    print( 'correct  : ',  num_correct)
    print( 'sys score: ', num_sys)
    print( 'gold ann : ', num_gold)

    print( ' total_num : ', total_num, '  eval_num : ', eval_num, ' Percent : %.2f'% (float(eval_num)/total_num) )

  ap =0; ar = 0
  totalGold = 0; totalSys=0; totalCorrect = 0;
  avgF = 0
  tag2prf = {}
  for t in num_correct:
    cn = num_correct[t]
    gn = num_gold[t]
    sn = num_sys[t]
    totalGold += gn
    totalSys += sn
    totalCorrect += cn


    p = 0.0
    if sn > 0:
      p = float(cn)/float(sn)
    r = 0.0
    if gn > 0:
      r = float(cn)/gn
    f = 0.0
    if p+r > 0:
      f = 2*p*r/(p+r)
    ap += p
    ar += r
    avgF += f
    if debug >0:
      print( ' tag : %15s'% t , ' P : %.3f'%p, '  R : %.3f'%r, ' F1 : %.3f'%f)

    tag2prf[t] = [p, r, f]


  ap = float(ap)/len(num_correct)
  ar = float(ar)/len(num_correct)
  af = 2*ap*ar/(ap+ar)

  avgF = avgF/len(num_correct)

  mip = float(totalCorrect)/totalSys
  mir = float(totalCorrect)/totalGold
  mif = 2*mip*mir/(mip+mir)

  if debug > 0:
    print( ' #### macro avgscore : P: %.3f'%ap, '  R: %.3f'%ar, '  F1 : %.3f'%af )
    print( ' #### micro avgscore : P: %.3f'%mip, '  R: %.3f'%mir, '  F1 : %.3f'%mif, ' totalCorrect:', totalCorrect)

    print( ' #### AVG  : P: %.3f'%ap, '  R: %.3f'%ar, '  F1 : %.3f'%avgF)

  scoredict = { 'tag2prf' : tag2prf,
          'macroF1' : af,
          'microF1' : mif
          }

  #return tag2prf, f1dict
  return scoredict




def get_confusion_matrix(alist, blist, printInfo=False):
  confdict = {}
  tagkeys = set()
  for i in range(len(alist)):
    atag = alist[i]
    btag = blist[i]
    if atag not in confdict:
      confdict[atag] = {}
    if btag not in confdict[atag]:
      confdict[atag][btag] = 0
    confdict[atag][btag] += 1
    tagkeys.add(atag)
    tagkeys.add(btag)

  tagkeys = [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental', 'None' ]

  if printInfo:
    print( '======== confusion matrix ======')
    tagkeys = list(tagkeys)
    print( ' '*12, end='' )
    for atag in tagkeys:
      print( '%10s'%atag, end='')
    print('')

    for atag in tagkeys:
      print('%14s'%atag, end='')
      if atag not in confdict:
        for btag in tagkeys:
          print(' %8d '%0, end='')
      else:
        for btag in tagkeys:
          if btag not in confdict[atag]:
            print( ' %8d '%0, end='')
          else:
            print( ' %8d '%confdict[atag][btag], end='')
      print('')
    print('\n')

  a2b2freq =  confdict
  return a2b2freq



def output_test_cross_val_performance( foldId2testPerformance, debug=0 ):
  tag2avgPRF = {}
  for foldId in foldId2testPerformance:
    tag2prf = foldId2testPerformance[foldId][0]
    for tag in tag2prf:
      p, r, f = tag2prf[tag]
      if tag not in tag2avgPRF:
        tag2avgPRF[tag] = [0, 0, 0]
      tag2avgPRF[tag][0] += p
      tag2avgPRF[tag][1] += r
      tag2avgPRF[tag][2] += f

  fold_num = len(foldId2testPerformance)
  avgP, avgR, avgF = 0, 0, 0
  for tag in tag2avgPRF:
    tag2avgPRF[tag][0] = tag2avgPRF[tag][0]/float(fold_num)
    tag2avgPRF[tag][1] = tag2avgPRF[tag][1]/float(fold_num)
    tag2avgPRF[tag][2] = tag2avgPRF[tag][2]/float(fold_num)

    avgP += tag2avgPRF[tag][0]
    avgR += tag2avgPRF[tag][1]
    avgF += tag2avgPRF[tag][2]

  class_num = len(tag2avgPRF)
  if debug > 0:
    print( '----- class_num : ', class_num )
    print( ' avgF : ', avgF )
  avgP /= float(class_num)
  avgR /= float(class_num)
  avgF /= float(class_num)


  #### print scores
  if debug > 0:
    print( '\n\n #### CrossValidation  avg performance for '+ str(fold_num) + '  folds' )
  for tag in tag2avgPRF:
    p, r, f = tag2avgPRF[tag]
    if debug > 0:
      print( '%15s'%tag, '  P: %.1f'%(100*p), '  R: %.1f'%(100*r), '  F1: %.1f'%(100*f) )

  if debug > 0:
    print( '\n ------  All AVG ------' )
    print( '  avgP: %.1f'%(avgP*100), '  avgR: %.1f'%(100*avgR), '  avgF1: %.1f'%(100*avgF) )
    print( '' )



  if debug > 0:
    print( '\n ##### Latex Format ---')
  tagkeys = [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental', 'None' ]
  for tag in tagkeys:
    if debug > 0:
      print( '%8s'%tag[:8], '&', end='')
  if debug > 0:
    print(' AvgPRF ' )
  for tag in tagkeys:
    p, r, f = tag2avgPRF[tag]
    if debug > 0:
      print('%.0f'%(100*p), '%.0f'%(100*r), '%.0f'%(100*f), '&', end='')
  if debug > 0:
    print( '%.1f'%(100*avgP), '%.1f'%(100*avgR), '%.1f'%(100*avgF) )
    print( '' )


  #out_confusion_matrix(foldId2testPerformance)
  testScoredict = out_micro_avg_performance( foldId2testPerformance, tagkeys, 0 )

  foldAvgF = avgF
  testAvgMicroF1 = testScoredict['microF1']
  testAvgMacroF1 = testScoredict['macroF1']

  foldScoredict = {'avgF': foldAvgF, 'tag2prf' : tag2avgPRF   }
  testScoredict = {'avgF': testScoredict['macroF1'], 'tag2prf': testScoredict['tag2prf'] }

  #return foldScoredict
  return testScoredict



def out_micro_avg_performance(foldId2testPerformance, taglist, debug=0):
  #print( '------------------'*10 )
  #taglist = set()
  gold_evid2tag = {}
  sys_evid2tag = {}

  gold_taglist = []
  sys_taglist = []
  for foldId in foldId2testPerformance:
    gEvid2tag, sEvid2tag = foldId2testPerformance[foldId][2]
    for evid in gEvid2tag:
      gtag = gEvid2tag[evid]
      stag = sEvid2tag[evid]
      #taglist.add(gtag)
      gold_evid2tag[evid] = gtag
      sys_evid2tag[evid] = stag

      gold_taglist.append(gtag)
      sys_taglist.append(stag)

  #taglist = list(taglist)
#
  if debug > 0:
    print( '\n'*2 )
    print( '##'*10, '  Micro Average ##########' )
  scoredict = compute_performance(taglist, gold_evid2tag, sys_evid2tag)

  #print("\n ======== Micro Performance by Sklearn ========")
  #print(classification_report(gold_taglist, sys_taglist, digits=3 ) )

  return scoredict



def out_confusion_matrix(foldId2testPerformance):
  sys2gold2freq = {}

  for foldId in foldId2testPerformance:
    a2b2f = foldId2testPerformance[foldId][1]
    for systag in a2b2f:
      if systag not in sys2gold2freq:
        sys2gold2freq[systag] = {}
      for goldtag in a2b2f[systag]:
        f = a2b2f[systag][goldtag]
        if goldtag not in sys2gold2freq[systag]:
          sys2gold2freq[systag][goldtag] = 0
        sys2gold2freq[systag][goldtag] += f

  print( '========  Before Avg confusion matrix across folds ======')
  print_conf_matrix( sys2gold2freq )


  fold_num =len(foldId2testPerformance)
  for systag in sys2gold2freq:
    for goldtag in sys2gold2freq[systag]:
      sys2gold2freq[systag][goldtag] = round( sys2gold2freq[systag][goldtag] / float(fold_num) )

  #confdict =  sys2gold2freq
  print( '========  Avg confusion matrix across folds ======' )
  print_conf_matrix( sys2gold2freq )


def print_conf_matrix( confdict ):

  tagkeys = [ 'Physiological', 'Health', 'Leisure', 'Social', 'Finance', 'Cognition', 'Mental', 'None' ]
  tagkeys = list(tagkeys)
  print( ' '*12, end='')
  for atag in tagkeys:
    print( '%10s'%atag, end='')
  print( '')

  for atag in tagkeys:
    print( '%14s'%atag,end='')
    if atag not in confdict:
      for btag in tagkeys:
        print(' %8d '%0,end='')
    else:
      for btag in tagkeys:
        if btag not in confdict[atag]:
          print(' %8d '%0,end='')
        else:
          print(' %8d '%confdict[atag][btag],end='')
    print('')
  print('\n')



