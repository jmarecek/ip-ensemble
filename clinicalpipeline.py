"""
Process classifiers in test folders and combine into ensemble 

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE file in the root 
directory of this source tree or at 
http://www.apache.org/licenses/LICENSE-2.0.
Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice 
indicating that they have been altered from the originals.

If you use this code, please cite our paper:
@article{Kerr2020,
  author    = {Catherine Kerr and Terri Hoare and Paula Carroll and Jakub Marecek},
  title     = {Integer-Programming Ensemble of Temporal-Relations Classifiers},
  journal   = {Data Mining and Knowledge Discovery},
  volume    = {to appear},
  year      = {2020},
  url       = {http://arxiv.org/abs/1412.1866},
  archivePrefix = {arXiv},
  eprint    = {1412.1866},
}
"""

from __future__ import division
#from __future__ import print
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import glob
import os.path
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import clinicaloptimizer
import time


# In[13]:

def getWeight(cl, testnum, testset, testtype, score) :
    return df_weights[(df_weights.sysfile==cl)
                       & (df_weights.testnum==testnum) 
                       & (df_weights.testset==testset) 
                       & (df_weights['type']==testtype)][score].item()

#getWeight(ensembleList[1], '00', 'A', 'TLINK:Type:CONTAINS', 'F1')


# In[14]:

def buildResultFile(result, filename, filenum):
    #print("Building file " + str(filenum) + "  " + filename)
    try:    
        refFile = refFolder + filename + '/' + filename + goldfn
        soup = BeautifulSoup(open(refFile), "xml")             # open the Platinum file
        root = soup.annotations

        for rel in soup.find_all("relation"):                  # remove all existing relations
            soup.relation.decompose() 

        linkNum = 1                                           # starting number for lid attribute
        for ix, row in result.iterrows():
            source = ix[0]
            target = ix[1]
            relation = row['relation']
            if (relation in tempClasses) :
                source = ix[1]
                target = ix[0]
                relation = invRelClasses[relClasses.index(relation)]
            #print(source, target, relation)
            if (target != "t0") :
            # Now build the new relations
                relTag = soup.new_tag("relation")
                idTag  = soup.new_tag("id")
                idTag.string = str(linkNum) + "@r@" + filename + "@gold"
                relTag.append(idTag)
                typeTag  = soup.new_tag("type")
                typeTag.string = "TLINK"
                relTag.append(typeTag)
                parentsTag  = soup.new_tag("parentsType")
                parentsTag.string = "TemporalRelations"
                relTag.append(parentsTag)
                propTag  = soup.new_tag("properties")
                srcTag  = soup.new_tag("Source")
                srcTag.string = source
                propTag.append(srcTag)
                tTag  = soup.new_tag("Type")
                tTag.string = relation
                propTag.append(tTag)
                tgtTag  = soup.new_tag("Target")
                tgtTag.string = target
                propTag.append(tgtTag)
                relTag.append(propTag)
                root.append(relTag)
                linkNum += 1    
        #print soup
        #outputText = soup.prettify()
        dirname = sysFolder + ensembleName + filename
        if not os.path.isdir(dirname):                 # create RESULTS directory if it doesn't exist
            os.makedirs(dirname)

        f = open(dirname + "/" + filename + postfn, 'w')
        f.write(str(soup))                                   # write out the new file
        f.close()

    except Exception as X:
        print "Error writing results to file"
        raise
    return


# In[15]:

def convertRelationToTuple(relation) :
    try :
        rp = relClasses.index(relation)
        relTuple = emptyTuple[:rp] + (1,) + emptyTuple[rp+1:] 
    except :
        relTuple = emptyTuple
    return relTuple


# In[16]:

def convertTupleToRelation(reltuple) :
    try :
        relation = relClasses[reltuple.index(1)]
    except :
        relation = 'None'
    return relation


# In[17]:

def getProbabilities(row) :
    try :
        result = emptyTuple
        num = 0
        for cl in ensembleList :
            #print(cl, testNum, trainingSet, 'TLINK:Type:CONTAINS', 'F1')
            clWeight = clWeights[ensembleList.index(cl)]
            #print(cl, clWeight)
            if (sum(row[cl]) != 0) :
                num += 1 
                for i in range(0, len(relClasses)) :
                    newval = result[i] + row[cl][i] * clWeight
                    result = result[:i] + (newval,) + result[i+1:] 
    except :
        result = emptyTuple
        raise
    return result


# In[18]:

def maxProbability(relTuple):
    error = ""
    try :
        maxValue = float('-inf')
        maxIndex = 0
        for i in range(len(relClasses)) :
            if float(relTuple[i]) > maxValue :
                maxValue = relTuple[i]
                maxIndex = i
        return relClasses[maxIndex]
    except:
        return error
        raise
            


# In[22]:

def createResults() :
    fileList = []
    for dirName in os.listdir(refFolder):
        fileList.append(dirName)
    try :
        for i in range(len(fileList)) :
        #for i in range(1, 2) :
	    print(fileList[i])
            df_ensembles = pd.read_csv(ensembleFolder + fileList[i] + ".csv")

            colNames = ['source', 'target'] + ensembleList

            df_ensembles = df_ensembles[colNames]
            df_ensembles.dropna(subset=precisionList, how='all', axis=0, inplace=True)
            
            test = 1/len(df_ensembles)
            
            # Convert relation classes to tuples
            df_ensembles[ensembleList] = df_ensembles[ensembleList].applymap(convertRelationToTuple)
            
            df_ensembles['percent'] = df_ensembles.apply(lambda row : getProbabilities(row), axis=1)
            #print(df_ensembles)

            df_opt = df_ensembles[['source','target','percent']][df_ensembles['target']!='t0']

            df_opt.set_index(['source', 'target'], inplace=True)

            if not optimise :
                #print("No optimise")
                df_opt['relation'] = df_opt.apply(lambda row : maxProbability(row['percent']), axis=1)
                df_result = df_opt
            else :
                #print("Optimise")
                df_result = clinicaloptimizer.main(df_opt)
                df_result['relation'] = df_result['relation'].map(convertTupleToRelation)

            buildResultFile(df_result, fileList[i], i)
    except ArithmeticError :
        print("pThreshold is too high - no data in %s qualifies " % refFolder)
        

    print("Finished processing %s files " % refFolder)


# # Global variables
# 

# In[24]:

relClasses  =   ['AFTER',
                 'BEFORE',
#                 'BEFORE/OVERLAP',
#                 'AFTER/OVERLAP',
                 'CONTAINS',
                 'CONTAINS_INV',
                 'OVERLAP',
                 'BEGINS-ON',
                 'ENDS-ON',
                 'NONE'
                ]

invRelClasses = ['BEFORE',
                 'AFTER',
#                 'AFTER/OVERLAP',
#                 'BEFORE/OVERLAP',
                 'CONTAINS_INV',
                 'CONTAINS',
                 'OVERLAP',
                 'ENDS-ON',
                 'BEGINS-ON',
                 'NONE'
                ]

tempClasses  =   ['AFTER',
#                 'AFTER/OVERLAP',
                 'CONTAINS_INV'
                ]

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
#                  'Platinum'
                  ]

ensembleList =   [
#                  'uthealth-p2s1',
                  'uthealth-p2s2',
#                  'LIMSI_COT-RUN1',
                  'LIMSI_COT-RUN2',
                  'CDE-IIITH-crf',
#                  'CDE-IIITH-deepnl',
                  'GUIR-Phase2-Run1',
#                  'KULeuven-LIIR-run1',
                  'KULeuven-LIIR-run2',
#                  'VUACLTL-run1-phase2',
                  'VUACLTL-run2-phase2',
#                  'UtahBMI-RUN1-corrected',
                  'UtahBMI-RUN2-corrected'
#                  'Platinum'
                  ]

eName = ""
for cl in ensembleList :
    eName += str(classifierList.index(cl)+1) + "-"

emptyTuple = (0.0,)*len(relClasses)

getTestnum = lambda x: x[len(x)-4:len(x)-2]
getTestset = lambda x: x[len(x)-1:len(x)-0]
weightFile = 'anaforatools/results-training-and-test.csv'
df_weights = pd.read_csv(weightFile)
df_weights.loc[:,'testnum']  = (df_weights['reffile'].map(getTestnum))
df_weights.loc[:,'testset']  = (df_weights['reffile'].map(getTestset))

rootRefFolder  = 'Training-test-folders/Tempeval16_Platinum_clinic'
postfn  = '.Temporal-Relation.system.completed.xml'
goldfn  = '.Temporal-Relation.gold.completed.xml'
ensembleFolder = 'Tempeval16_ensembles/'
resultFolder   = 'Ensemble_clinic_results/'


# # Process ensembles

# In[25]:

tests   = ['A', 'B']
pThreshold = 0.65
optimise = 0
#for n in range(10) :
for n in range(1) :
    testNum = str(n).zfill(2)
    for testSet in tests :
        trainingSet = tests[1 - tests.index(testSet)]
        for optimise in range(1, 2) :
            refFolder  = rootRefFolder + "_" + testNum + "_" + testSet + '/'
            sysFolder  = resultFolder + testNum + "_" + testSet + '/'
            ensembleName = eName + 'NO' if optimise == 0 else eName + 'YES'
            ensembleName += '-' + str(pThreshold) + '/'
            clWeights = []
            precisionList = []
            for cl in ensembleList :
                clWeights.append(getWeight(cl, testNum, trainingSet, 'TLINK:Type:CONTAINS', 'F1'))
                if (getWeight(cl, testNum, trainingSet, 'TLINK:Type:CONTAINS', 'P') > pThreshold) :
                    precisionList.append(cl)
            #print(refFolder, sysFolder, ensembleName)
            start_time = time.time()
            createResults()
            print("%s \t %d \t %s seconds " % (cl, optimise, (time.time() - start_time)))



# In[ ]:



