# Calculates RecallX, PrecisionX and F1X scores for classifiers/ensemble
# Input parameters
# "ref="dirname         name of directory with benchmark data - default Platinum
# "sys="dirname         name of directory with classifier/ensemble data - default RESULTS
# "file="file*          partial or full name of files to be compared - default *.tml
# "outfile="filename    name of file with detailed results - default results.csv
# "summary="filename    name of file with summary results - default summary.csv

from classifier import Classifier
import glob
import os.path
import sys

relTypes = ["BEFORE",
             "AFTER",
             "INCLUDES",
             "IS_INCLUDED",
             "DURING",
             "DURING_INV",
             "SIMULTANEOUS",
             "IAFTER",
             "IBEFORE",
             "IDENTITY",
             "BEGINS",
             "ENDS",
             "BEGUN_BY",
             "ENDED_BY",
             "NONE"]	  # list of relationship types

invRelTypes = ["AFTER",
             "BEFORE",
             "IS_INCLUDED",
             "INCLUDES",
             "DURING_INV",
             "DURING",
             "IDENTITY",
             "IBEFORE",
             "IAFTER",
             "SIMULTANEOUS",
             "ENDS",
             "BEGINS",
             "ENDED_BY",
             "BEGUN",
             "NONE"]	        # inverse of relationship types

def buildOutputString(classifierName, filename, geid1, geid2, setA_eiid1, setA_eiid2, eiid1, eiid2, crelType, relType):
    outputText = classifierName+","
    outputText += filename+","
    outputText += str(geid1)+","
    outputText += str(geid2)+","
    outputText += str(setA_eiid1)+","
    outputText += str(setA_eiid2)+","
    outputText += str(eiid1)+","
    outputText += str(eiid2)+","
    outputText += str(crelType)+","
    outputText += str(relType)+"\n"
    return outputText

def buildOutputString2(filename, geid1, geid2, clReltypes, relType, eiid1, eiid2):
    outputText  = filename+","
    outputText += str(geid1)+","
    outputText += str(geid2)+","
    outputText += str(clReltypes)
    outputText += str(relType)+","
    outputText += str(eiid1)+","
    outputText += str(eiid2)+"\n"
    return outputText

def invArc(arc):
    left = arc[0]
    right = arc[1]
    return (right, left)

def invRelType(relType):
    return invRelTypes[relTypes.index(relType)]

def extract(filename, arcs):
#    global outputResults
#    global outputSummary
    global sysClassifiers
    global refClassifier
    goldLinks = 0
    clScores = {}
    
    for arc in arcs:
        geid1 = arc[0]              # global event instance id or time id
        geid2 = arc[1]
        try:
            relType = relTypes[arcs[arc][refDir].index(1)]
        except:
            relType = "NONE"
        if relType != "NONE":
            goldLinks += 1

        if geid1[0] == "e":
            eiid1 = refClassifier.mapEventInstanceID(refClassifier.classID, geid1)
        else:
            eiid1 = refClassifier.mapTimeID(refClassifier.classID, geid1)
        if geid2[0] == "e":
            eiid2 = refClassifier.mapEventInstanceID(refClassifier.classID, geid2)
        else:
            eiid2 = refClassifier.mapTimeID(refClassifier.classID, geid2)

        clReltypes = ""
        clResults = ""
        for cl in sysClassifiers:
            if cl.classID in clScores:
                (links, matchedLinks, matchedReltypes) = clScores[cl.classID]
            else:
                (links, matchedLinks, matchedReltypes) = (0, 0, 0)
            try:
                crelType = relTypes[arcs[arc][cl.classID].index(1)]
            except:
                crelType = "NONE"
            clReltypes += crelType + ","
            if crelType != "NONE":
                links += 1
                if relType != "NONE":
                    matchedLinks += 1
                    if crelType == relType:
                        matchedReltypes += 1
            clScores[cl.classID] = (links, matchedLinks, matchedReltypes)
            clResults += str(links)+","+str(matchedLinks)+","+str(matchedReltypes)+","
            
    return (links, matchedLinks, matchedReltypes, goldLinks)

refDir = "Platinum"
sysDir = "RESULTS"
refFile = "*.tml"
outFile = "results.csv"
outFileSummary = "summary.csv"
sysClassifiers = []
refClassifier = ""


def main(args):
    global refDir
    global sysDir
    global refFile
    global sysClassifiers
    global refClassifier

    for arg in args:
        if arg[:4] == "ref=":
            refDir = arg[4:]
        if arg[:4] == "sys=":
            sysDir = arg[4:]
        if arg[:5] == "file=":
            refFile = arg[5:]       
        if arg[:8] == "outfile=":
            outFile = arg[8:]       
        if arg[:8] == "summary=":
            outFileSummary = arg[8:]       

    if not os.path.isdir(refDir):
        print refDir, " is not a valid directory"
        return False
        
    # list of classifiers - sub-directories starting with CLASSIFIER_
    classifiers = []                                        # initialise list of classifiers
    classifierList = glob.glob(sysDir)                      # Change input parameter if you want to use different directories
    clList = []
    for cl in classifierList:
        if os.path.isdir(cl):
            clList.append(cl)
    if len(clList) == 0:
        print "No directories starting with ", sysDir
        return False
    #print clList
        
    # Files for each classifier must be placed in folders starting with "CLASSIFIER_" (default) or folder name specified in parameter
    filelist = glob.glob(refDir+"/"+refFile)   # List files in reference directory
    if len(filelist) == 0:
        print "No files in directory ", refDir
        exit()

    i = 0

    sysClassifiers = []
    (totalLinks, totalMatchedLinks, totalMatchedReltypes, totalGoldLinks) = (0, 0, 0, 0)
    for filename in filelist:
        globalClassifier = Classifier("FINAL", True)         # TLINKS and probability of relTypes across ALL classifiers goes here
        refClassifier = Classifier(refDir,False)
        refClassifier.parseTml(filename)
        i = 0
        sysClassifiers = []
        for classifier in clList:
            sysClassifiers.append(Classifier(classifier, False))
            cFilename = filename.replace(refDir,classifier)
            if os.path.isfile(cFilename):
#                print "Calling parseTml ", cFilename, classifier
                sysClassifiers[i].parseTml(cFilename)
            else:
                print "File ", cFilename, "not found"
            i += 1
        (links, matchedLinks, matchedReltypes, goldLinks) = extract(filename, globalClassifier.getGlobalArcs())
        totalLinks += links
        totalMatchedLinks += matchedLinks
        totalMatchedReltypes += matchedReltypes
        totalGoldLinks += goldLinks
    recall = totalMatchedLinks/float(totalGoldLinks)
    precision = totalMatchedReltypes/float(totalMatchedLinks)
    f1 = 2*recall*precision/float(recall+precision)
    print f1, precision, recall, totalMatchedReltypes
    return (f1, precision, recall, totalMatchedReltypes)
        
if __name__ == "__main__":
    main(sys.argv)

    

