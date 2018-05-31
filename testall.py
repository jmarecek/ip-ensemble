import glob
import os.path
import sys
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
weights  = (1,2,3,4,5,6,7)      # all weights
formulas = (2,)                 # divide by sum of classifiers finding arc
options  = (0,1)                # 0,1 - with and without optimizer

resultsFile = open("AllResults.csv","a")
outputText = buildOutputString("Ensemble","Optimise","Formula","Weight","F1","Precision","Recall","F1X","PrecisionX","RecallX","Matched","Benchmark")
resultsFile.write(outputText)

for eName in ensembles:
    eList = []
    eComp = ensembles[eName]
    for x in eComp:
        eList.append(dirName+str(x))
    for opt in options:
        for formula in formulas:
            for weight in weights:
                w = str(weight)
                f = str(formula)
                o = str(opt)
                print "Processing ensemble",eList,"with option",o,", formula",f,", and weight",w
                result = ensemble.main([eList,"weight="+w,"formula="+f,"opt="+o])
                if result:
                    (f1, pr, re) = tempeval.main(["x","Platinum-training","RESULTS",0])
                    (f1x, prx, rex, matched) = newmetric.main(["ref=Platinum-training","sys=RESULTS"])
                    outputText = buildOutputString(eName, o, f, w, f1, pr, re, f1x, prx, rex, matched, "Platinum")
                    resultsFile.write(outputText)
#                    (f1, pr, re) = tempeval.main(["x","Platinum-optimised-5","RESULTS",0])
#                    (f1x, prx, rex, matched) = newmetric.main(["ref=Platinum-optimised-5","sys=RESULTS"])
#                    outputText = buildOutputString(eName, o, f, w, f1, pr, re, f1x, prx, rex, matched, "Optimised-5")
#                    resultsFile.write(outputText)
                else:
                    print eName, o, f, w, " didn't work!"

resultsFile.close()


    


