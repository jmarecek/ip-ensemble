"""
Construct the ROC plot.

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

import glob
import os.path
import sys
from itertools import combinations
import traceback
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

import ensemble
import tempeval
import newmetric


def buildOutputString(dirName, opt, formula, weight, f1, precision, recall, f1x, precisionx, recallx, matched, benchmark):
    outputText = dirName+","
    outputText += opt+","
    outputText += formula+","
    outputText += weight+","
    outputText += str(f1)+","
    outputText += str(precision)+","
    outputText += str(recall)+","
    outputText += str(f1x)+","
    outputText += str(precisionx)+","
    outputText += str(recallx)+","
    outputText += str(matched)+","
    outputText += str(benchmark)+","
    outputText += "\n"
    return outputText

"""
ensembles = {"2-9-5":(2,9,5),
             "2-9-5-4":(2,9,5,4),
             "2-9-5-4-A":(2,9,5,4,"A"),
             "2-9-5-4-A-B":(2,9,5,4,"A","B"),
             "2-9-5-4-A-B-1":(2,9,5,4,"A","B",1),
             "2-9-5-4-A-B-1-3":,
            }
"""

        
groundset = (1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B")
#groundset = (2,9,5,4)
#groundset = (2,9,5)
ensembles = {}
for i in range(2, len(groundset)):
  for e in set(list(combinations(groundset, i))):
      name = '-'.join([ str(x) for x in e ])
      ensembles[name] = e
print "There are", len(ensembles), "ensembles:", ensembles.keys()

#ensembles = {"1-2-3-4":(1,2,3,4),
#             "1-2-3-4-5":(1,2,3,4,5),
#             "1-2-3-4-5-9":(1,2,3,4,5,9),
#ensembles = {"1-2-3-4":(3,2,4,1),
#             "1-2-3-4-5":(3,5,2,4,1),
#             "1-2-3-4-5-9":(3,5,2,9,4,1),
#ensembles = {"1-2-3-4-5-9":(3,5,2,9,4,1)
#            }
#ensembles =   {"2-A-5-4-9-B-1-3-6":(2,"A",5,4,9,"B",1,3,6),
#ensembles = {"1-cleartk-1":(1,),
#             "2-cleartk-2":(2,),
#             "3-cleartk-3":(3,),
#             "4-cleartk-4":(4,),
#             "5-navytime-1":(5,),
#             "6-UT-1":(6,),
#             "7-UT-2":(7,),
#             "8-UT-3":(8,),
#             "9-UT-4":(9,),
#             "A-UT-5":("A",),
#             "B-navytime-2":("B",)
#             }

#ensembles = {"UT reltype only":(1, 2, 3)} 
dirName  = "Training-"          # start of folder names - will be concatenated with alphanumeric code for classifier
weights  = (8,)              # all weights
formulas = (2,)                 # divide by sum of options finding arc
options  = (0,)                # 0,1 - with and without optimizer

resultsFile = open("ROC2.csv","a")
outputText = buildOutputString("Ensemble","Optimise","Formula","Weight","F1","Precision","Recall","F1X","PrecisionX","RecallX","Matched","Benchmark")
resultsFile.write(outputText)

results = {}

try:
 for eName, eComp in ensembles.iteritems():
  for convexifying in [x/2.0 for x  in range(3)]:
    eList = []
    for x in eComp:
        eDirName = dirName+str(x)
        eList.append(eDirName)
        # print "Attempting to load", eDirName
    for opt in options:
        for formula in formulas:
            for weight in weights:
                w = str(weight)
                f = str(formula)
                o = str(opt)
                print "Processing ensemble",eList,"with option",o,", formula",f,", weight",w, ", and convexifying coefficient", convexifying
                result = ensemble.main([eList,"weight="+w,"formula="+f,"opt="+o,"convexifying="+str(convexifying)])
                #print "Ensemble is", result
                if result:
                    (f1, pr, re) = tempeval.main(["x","Platinum","RESULTS",0])
                    results[(eName, convexifying)] = (pr, re)
                    s = ','.join([ eName, str(convexifying), str(pr), str(re) ])
                    resultsFile.write(s + "\n")
                    with open("ROC2.pkl", 'wb') as resultsPickle: pickle.dump(results, resultsPickle)
                else:
                    print eName, o, f, w, " didn't work!"
except (KeyboardInterrupt, SystemExit):
  raise
except:
  print traceback.format_exc()

resultsFile.close()
