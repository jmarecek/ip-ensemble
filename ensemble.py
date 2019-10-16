"""
Tests the ensemble

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

from classifier import Classifier
import glob
import os.path
import sys
import optimizer

def main(args):
#    classDir = "CLASSIFIER_"
    classFile = "*.tml"
    weight  = "1"
    convexifying = 0.5
    weights = ["1","2","3","4","5","6","7","8"]# '1' = same weight across classifiers
                                               # '2' = F1 score of each classifier
                                               # '3' = Precision score of each classifier
                                               # '4' = Recall score of each classifier
                                               # '5' = new F1 score of each classifier
                                               # '6' = new Precision score of each classifier
                                               # '7' = new Recall score of each classifier
                                               # '8' = a convex combination of Precision and Recall, with convexifying coefficient
    weightFormula = "1"
    formulas = ["1","2","3","4","5","6","7","8","9"]      # '1' = proportion of classifiers that detected arc
                                              # '2' = proportion of ALL classifiers
                                              # '3' = weights normalised to 1 and link excluded if below 0.5 threshold
                                              # '4' = sum of logs of probabilities
                                              # '5' = product of probabilities
                                              # '6' = loss function
    opt = '1'
    for arg in args:
        if arg[:4] == "dir=":
            classDir = arg[4:]
        if arg[:5] == "file=":
            classFile = arg[5:]
        if arg[:7] == "weight=":
            weight = arg[7:]
            if not weight in weights:
                print "Weight must be in range 1 to 8"
                return False        
        if arg[:8] == "formula=":
            weightFormula = arg[8:]
            if not weightFormula in formulas:
                print "Weight formula must be in range 1 to 9"
                return False        
        if arg[:4] == "opt=":
            opt = arg[4:]
        if arg[:13] == "convexifying=":
            convexifying = float(arg[13:])
            #print "Received a convexifying coefficient of", convexifying

    # list of classifiers - sub-directories starting with CLASSIFIER_
#    classifiers = []                                        # initialise list of classifiers
    classifierList = args[0]                # Change input parameter if you want to use different directories
    clList = []
    for cl in classifierList:
        if os.path.isdir(cl):
            clList.append(cl)
    if len(clList) == 0:
        print "No directories in list ", classifierList
        return False

    # Files for each classifier must be placed in folders starting with "CLASSIFIER_" (default) or folder name specified in parameter
    filelist = glob.glob(clList[0]+"/"+classFile)           # Lists the files in the first directory
    if len(filelist) == 0:
        print "No files in directory ", clList[0]
        return False

    try:
        for filename in filelist:                               # for each file in the first directory
            finalClassifier = Classifier("FINAL", True)         # TLINKS and probability of relTypes across ALL classifiers goes here
            finalClassifier.setWeightFormula(weight, weightFormula, convexifying) # determine which weighting formula to use
            for classifier in clList:                           # for each classifier
                clName = Classifier(classifier, False)          # Classifier object instantiated for each classifier
                cFilename = filename.replace(clList[0],classifier)  # file path is replaced by classifier path name
                if os.path.isfile(cFilename):                   # parse the file for events, timex3, makeinstance and tlinks
#                    print "Calling parseTml ", cFilename, classifier
                    clName.parseTml(cFilename)
                else:
                    print "File ", cFilename, "not found"
         
            v = finalClassifier.arcProbability()                # Dictionary of probabilities across multiple classifiers
    #        print v
            if opt == '1':
                result = optimizer.main(v)                     # optimise to maximise probability of reltypes
            else:
                result = finalClassifier.maxProbability(v)      # get relType with maximum probability

            finalClassifier.mapResults(result)                  # create new tml in folder RESULTS
    except Exception as X:
        print X
        return False
    return True

if __name__ == "__main__":
    main(sys.argv)
