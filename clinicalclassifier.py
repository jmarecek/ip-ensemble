"""
Parse file across all classfiers and combine into single dataframe

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
#from clinicalclassifier import ClinicalClassifier as classifier


# # Parse single file and create dataframe
# 

# In[3]:

class ClinicalClassifier :
    
    relClasses  =   ['AFTER',
                     'BEFORE',
                     'BEFORE/OVERLAP',
                     'AFTER/OVERLAP',
                     'CONTAINS',
                     'CONTAINS_INV',
                     'OVERLAP',
                     'BEGINS-ON',
                     'ENDS-ON'
                    ]

    invRelClasses = ['BEFORE',
                     'AFTER',
                     'AFTER/OVERLAP',
                     'BEFORE/OVERLAP',
                     'CONTAINS_INV',
                     'CONTAINS',
                     'OVERLAP',
                     'ENDS-ON',
                     'BEGINS-ON'
                    ]


    def __init__(self, classID, fn, eMap):
        self.classID = classID      # takes name of folder
        self.weight = 1.0
        # (dfe, dfr) = self.parseFile(fn, eMap)
        self.relations = self.parseFile(fn, eMap)
        return
    
    def invert(self, relation) :
        try :
            return self.invRelClasses[self.relClasses.index(relation)]
        except :
	    print(self.classID, relation, self.relClasses, self.invRelClasses)
            return "INVALID"

    def swap(self, x, y) :
        return (y, x)

    def getSpan(self, df, eid) :
        try :
    	    start = int(df[df.eid == eid]['start'].iloc[0])
	    end   = int(df[df.eid == eid]['end'].iloc[0])
	    return (start, end)
	except :
	    raise

    def mapEvent(self, start, end, eid, eMap) :
        eidMap = ""
	try :
	    findKey = (start, end)
	    eidMap = eMap.at[findKey,'eid']
	    etype = eMap.at[findKey,'type']
	    if not isinstance(eidMap, basestring) :
		ix = np.where(eidMap == eid)
		if (not ix[0]) :
		    ix = np.where(etype == 'EVENT')
		eidMap = eidMap[ix[0][0]]
	    return eidMap
        except Exception as inst :
            print(self.classID, "error mapping event ", eid)
	    raise
	return eidMap

    def parseFile(self, fn, eMap) :
        df_entities  = pd.DataFrame(columns=('start', 'end', 'eid', 'doctimerel'))
        df_relations = pd.DataFrame(columns=('source', 'target', 'ttype'))
        row = 0
        erow = 0
        #print("Opening file : ", fn)
        f = open(fn)
        soup = BeautifulSoup(f, "xml")
        # get list of entities and spans
        for entity in soup.find_all('entity'):
            eid = entity.find("id").string 
            span = entity.find("span").string.split(",")
            properties = entity.find("properties")
            doctimetag = properties.find("DocTimeRel")
            if doctimetag is not None :
                doctimerel = doctimetag.string
            else:
                doctimerel = ""
            df_entities.loc[erow] = [span[0], span[len(span)-1], eid, doctimerel]
            erow += 1

	    try :
                if (doctimerel != "") :
                    (start, end) = (int(span[0]), int(span[len(span)-1]))
		    eid = self.mapEvent(start, end, eid, eMap)
                    df_relations.loc[row] = [eid, "t0", doctimerel]
                    row += 1
                    #print eid, " t0 ", doctimerel
            except :
                print(self.classID, "doctime key not found : ", eid)
		continue		# change to continue
        
        # get list of relations
        #row = 0
        for rel in soup.find_all('relation'):
            relID = rel.find("id").string
            if (rel.find("type").string == "TLINK") :
                properties = rel.find("properties")

                try :
                    # find Source in entities and get span
                    source = properties.find("Source").string
		    (start, end) = self.getSpan(df_entities, source)
		    source = self.mapEvent(start, end, source, eMap)

                    # find Target in entities and get span
                    target = properties.find("Target").string
		    (start, end) = self.getSpan(df_entities, target)
		    target = self.mapEvent(start, end, target, eMap)

                    tType  = properties.find("Type").string
		    if (source > target) :
		        (source, target) = (target, source)
			tType = self.invert(tType)
                    df_relations.loc[row] = [source, target, tType]
                    row += 1
                except Exception as X:
                    #print X
                    print "Error processing file ", fn, " relID: ", relID
                    continue	# change to continue
        
        df_relations.rename(columns={'ttype': self.classID}, inplace=True)

        return df_relations

