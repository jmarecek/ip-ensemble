"""
Class ClinicalEnsemble
Parse file across all classfiers and combine into single dataframe
For example
Source     Target     classifier1    classifier2    ...    classifiern
e1         e2         BEFORE                               CONTAINS
e1         e3                        CONTAINS              OVERLAP
etc.

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

import numpy as np
import pandas as pd
from clinicalclassifier import ClinicalClassifier 

class ClinicalEnsemble :
    
    def __init__(self, fileName, prefix):
        self.fileName = fileName      
        self.df = pd.DataFrame(columns=('source', 'target'))
        self.eMap = self.getEmap(fileName, prefix)
        
        return
    
    # Get EVENT map - this is a file created from the gold standard data using the span start and end as key
    # and the event ID as value
    def getEmap(self, fileName, prefix):
    
        mapFileName = prefix + fileName + ".csv"
        eMap = pd.read_csv(mapFileName)
        eMap.set_index(['start', 'end'], inplace=True)
        #eMap.sort()
        
        return eMap
    
    # Add a classifier to the ensemble dataframe
    def addClassifier(self, classifier) :

        cl = classifier.classID
        df_classifier = classifier.relations
	df_classifier.drop_duplicates(['source', 'target'], keep='first', inplace=True)
        if (self.df.empty) :
            print("Adding first classifier : " + cl)
            self.df = df_classifier
        else :
            print("Adding new classifier : " + cl)
            
            # If there are any reverse relations in current classifier, put them in new dataframe df_inner
            df_inner  = pd.merge(self.df, 
                        df_classifier, 
                        left_on=['source', 'target'], right_on=['target','source'],
                        how='inner',
                        suffixes=('', '_y'))[df_classifier.columns]
            
            # Invert the relations for the new dataframe
            df_inner[cl] = df_inner[cl].map(lambda x : classifier.invert(x))

            # Join the new classifier with the inverted relations
            df_classifier = pd.merge(df_classifier, 
                        df_inner, 
                        left_on=['source', 'target'], right_on=['target','source'],
                        how='left',
                        suffixes=('', '_y'))

            # Remove reverse relations from classifier
            df_classifier= df_classifier[(pd.isnull(df_classifier['source_y']))][['source', 'target', cl]]

            # Add inverted relations to classifier
            df_classifier = pd.concat([df_classifier, df_inner])

            # Add new classifier to ensemble
            self.df = pd.merge(self.df, 
                        df_classifier, 
                        left_on=['source', 'target'], right_on=['source','target'],
                        how='outer',
                        left_index=False,
                        right_index=False,
                        suffixes=('', '_y'))

            # Remove superfluous index column
            if ('index' in (self.df).columns) :
                (self.df).drop('index', axis=1, inplace=True)
                
            #print((self.df).head())

        return
    
