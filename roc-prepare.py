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
import ensemble
import tempeval
import newmetric

import traceback

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

ensembles = {"2-9-5":(2,9,5),
             "2-9-5-4":(2,9,5,4),
             "2-9-5-4-A":(2,9,5,4,"A"),
             "2-9-5-4-A-B":(2,9,5,4,"A","B"),
             "2-9-5-4-A-B-1":(2,9,5,4,"A","B",1),
             "2-9-5-4-A-B-1-3":(2,9,5,4,"A","B",1,3),
            }
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

resultsFile = open("AllResults.csv","a")
outputText = buildOutputString("Ensemble","Optimise","Formula","Weight","F1","Precision","Recall","F1X","PrecisionX","RecallX","Matched","Benchmark")
resultsFile.write(outputText)

listofprre = []

try:
  for convexifying in [x/10.0 for x  in range(11)]:
    eList = []
    # eComp = (2,9,5)
    # eName = "2-9-5"
    eName = "2-9-5-4-A-B-1-3"
    eComp = (2,9,5,4,"A","B",1,3)
    for x in eComp:
        eDirName = dirName+str(x)
        eList.append(eDirName)
        print "Attempting to load", eDirName
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
                    listofprre.append((pr, re))
                    #(f1x, prx, rex, matched) = newmetric.main(["ref=Platinum","sys=RESULTS"])
                    #outputText = buildOutputString(eName, o, f, w, f1, pr, re, f1x, prx, rex, matched, "Platinum")
                    #resultsFile.write(outputText)
#                    (f1, pr, re) = tempeval.main(["x","Platinum-optimised-5","RESULTS",0])
#                    (f1x, prx, rex, matched) = newmetric.main(["ref=Platinum-optimised-5","sys=RESULTS"])
#                    outputText = buildOutputString(eName, o, f, w, f1, pr, re, f1x, prx, rex, matched, "Optimised-5")
#                    resultsFile.write(outputText)
                else:
                    print eName, o, f, w, " didn't work!"
except (KeyboardInterrupt, SystemExit):
  raise
except:
  print traceback.format_exc()

print listofprre
resultsFile.close()
