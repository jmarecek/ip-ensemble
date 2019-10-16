
# coding: utf-8

# In[ ]:

import os.path
import numpy as np
import pandas as pd

from anafora import evaluate2 as evaluate

# refFolder is where the Platinum output xml files are stored
refFolder = './Training-test-folders/Tempeval16_Platinum_clinic'

# sysFolder is where the ensemble output folders will be stored
sysFolder = './Ensemble_clinic_results/'   # use this for ensembles

# classifierList is the list of classifiers 
classifierList = ['uthealth-p2s1',
                  'uthealth-p2s2',
                  'LIMSI_COT-RUN1',
                  'LIMSI_COT-RUN2',
                  'CDE-IIITH-crf',
                  'CDE-IIITH-deepnl',
                  'GUIR-Phase2-Run1',
                  'KULeuven-LIIR-run1',
                  'KULeuven-LIIR-run2',
                  'VUACLTL-run1-phase2',
                  'VUACLTL-run2-phase2',
                  'UtahBMI-RUN1-corrected',
                  'UtahBMI-RUN2-corrected'
                  ]


# In[ ]:

def getTargetScore(testnum, testset, testtype, score) :
    return df_weights[(df_weights.testnum==testnum) 
                       & (df_weights.testset==testset) 
                       & (df_weights['type']==testtype)][score].max()


# In[ ]:

# getTestnum is the number of the rest set - 00..09
getTestnum = lambda x: x[len(x)-4:len(x)-2]
# get Testset returns the letter 'A' or 'B' (training or test sets)
getTestset = lambda x: x[len(x)-1:len(x)-0]
# weightFile contains the results for all classifiers - used as weights
weightFile = 'results-training-and-test.csv'

# Put all scores for each classifier and each training/test set in dataframe df_weights
df_weights = pd.read_csv(weightFile)
df_weights.loc[:,'testnum']  = (df_weights['reffile'].map(getTestnum))
df_weights.loc[:,'testset']  = (df_weights['reffile'].map(getTestset))


# In[ ]:


results_file = 'ensemble-results.csv'
# ensemble lists the classifiers separated by '-' that make up the ensemble composition
# 1='uthealth-p2s1', 2='uthealth-p2s2', etc.
# next line must be changed for each composition
ensemble = '1-2-3-4-5-6-7-8-9-10-11-12-13'

# pThreshold is Precision value below which classifier cannot nominate TLINKS to ensemble, but can vote
pThreshold = 0.65

# headings for result file
columns = ('reffile', 
           'sysfile', 
           'num',
           'set',
           'pthr',
           'opt',
           'type', 
           'ref', 
           'pred', 
           'corr1', 
           'corr2', 
           'P', 
           'R', 
           'F1', 
           'arcs', 
           'P2', 
           'target',
           'delta')

df = pd.DataFrame(columns=columns)
row = 0
# for each of the test/training sets - there are 10 pairs
for n in range(10) :
    for testSet in ['A', 'B'] :
        ref = refFolder + '_' + str(n).zfill(2) + '_' + testSet
        target = getTargetScore(str(n).zfill(2), testSet, 'TLINK:Type:CONTAINS', 'F1')
        for optimise in range(2) :
            opt = 'NO' if not optimise else 'YES'
            sysname = str(n).zfill(2) + '_' + testSet + '/' + ensemble + '-' + opt + '-' + str(pThreshold)
            sys = sysFolder + sysname
            if (os.path.isdir(sys)) :
                inc = 'TLINK:Type:CONTAINS'          # type of test being run
                result = evaluate.main(['-r', ref, 
                        '-p', sys,
                        '-i', inc,
                        '--temporal-closure'])
                df.loc[row] = [ref[3:],              # reference file
                               ensemble,             # system file = ensemble name
                               str(n).zfill(2),      # sample number (00..09)
                               testSet,              # testset (A or B)
                               str(pThreshold),      # precision threshold
                               opt,                  # optimise (YES or NO)
                               inc,                  # test type (CONTAINS)
                               result[inc][0],       # number of reference links
                               result[inc][1],       # number of predicted links
                               result[inc][2][0],    # number of correctly labelled links
                               result[inc][2][1],    # number of correctly labelled closed links
                               result[inc][3],       # precision score
                               result[inc][4],       # recall score
                               result[inc][5],       # F1 score
                               result[inc][2][2],    # number of correctly detected closed links
                               result[inc][6],       # detection precision
                               target,               # target F1 score
                               result[inc][5] - target # difference between target and result
                              ]
                row += 1
                print result[inc][3]

header = not os.path.isfile(results_file)
with open(results_file, 'a') as f:
    df.to_csv(f, header=header, index=False)


# In[ ]:




# In[ ]:



