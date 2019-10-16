"""
Parses tml files in Tempeval-13 challenge
Maps events and timex tags to global identifiers
Calculates probabilities of classes using weighting formulas

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

from xml.etree.ElementTree import parse
from bs4 import BeautifulSoup
import os.path
import math
import numpy as np
import pandas as pd
  
class Classifier:
  
  geid = 0                # global event identifier
  gtid = 0                # global timex identifier
  globalEvents = {}       # dictionary of global events indexed by sentence id and text - sid_text
  globalArcs = {}         # dictionary of global arcs
  finalArcs = {}          # dictionary of acrs passed to optimiser
  eventInstances = {}     # mapping for event instance ids
  timeMapping = {}        # mapping for time expressions
  emptyTuple = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  noneTuple  = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)    # change last value to 1 if NONE is valid - for future use only
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
  
  invRelTypes = ["AFTER",       # inverse of BEFORE
              "BEFORE",         # inverse of AFTER
              "IS_INCLUDED",    # inverse of INCLUDES
              "INCLUDES",       # inverse of IS_INCLUDED
              "DURING_INV",     # inverse of DURING
              "DURING",         # inverse of DURING_INV
              "SIMULTANEOUS",   # inverse of SIMULTANEOUS
              "IBEFORE",        # inverse of IAFTER
              "IAFTER",         # inverse of IBEFORE
              "IDENTITY",       # inverse of IDENTITY
              "BEGUN_BY",       # inverse of BEGINS
              "ENDED_BY",       # inverse of ENDS
              "BEGINS",         # inverse of BEGUN_BY
              "ENDS",           # inverse of ENDED_BY
              "NONE"]	        # inverse of relationship types

  probReltype = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # probabilities of each reltype occurring in Platinum will be read from file
  bestClassifier = ()             # used for building new tml file - relevant for TaskABC
  mostEvents = 0                  # holds number of events for best classifier
  classifierList = []             # names of classifiers
  classifierWeight = {}           # weights of each classifier, key is classID
  weight = '1'                    # '1' = same weight across classifiers
                                  # '2' = F1 score of each classifier
                                  # '3' = Precision score of each classifier
                                  # '4' = Recall score of each classifier
                                  # '5' = new F1 score of each classifier
                                  # '6' = new Precision score of each classifier
                                  # '7' = new Recall score of each classifier
                                  # '8' = a convex combination of precision and recall
  convexifying = 0.5              # a convexifying coefficient for the combination of precision and recall (weight = '8')
  weightFormula = '1'             # '1' = proportion of classifiers that detected arc
                                  # '2' = proportion of ALL classifiers
                                  # '3' = weights normalised to 1 and link excluded if below 0.5 threshold
                                  # '4' = sum of logs of probabilities
                                  # '5' = product of probabilities
                                  # '6' = inverted loss function
                                  # '7' = loss function

  def __init__(self, classID, new):
      self.classID = classID      # takes name of folder
      self.arcs = {}              # dictionary of arcs for classifier
      self.eInstances = []	  # event instance IDs
      self.eInstRefs = []	  # event IDs corresponding to each event instance ID
      self.etMaps = {}            # mapping from classifier ids to global ids
      if new:                     # 'new' means starting to parse new file
        self.nextFile()           # reset static variables
      else:
        self.addClassifier(classID) # add classifier to list of classifiers and default weight to 1
        self.getScores(classID, Classifier.weight) # get scores for classifier
      return

  def addClassifier(classID):
    Classifier.classifierList.append(classID)
    Classifier.setWeight(classID, 1.0)
    return
    
  addClassifier = staticmethod(addClassifier)

  def setWeightFormula(weight, weightFormula, convexifying = 0.1):
    Classifier.weight = weight
    Classifier.weightFormula = weightFormula
    Classifier.convexifying = convexifying
    # print "Setting weight to", weight, "with convexifying coefficient", convexifying
    return
    
  setWeightFormula = staticmethod(setWeightFormula)

  def getScores(classID, weight):
      # print "Running getScores ... "
      inFile = "ClassifierScores.csv"    # open file containing scores for classifier
      try:
        if weight > '1':
          scores = pd.read_csv(inFile)
          if classID[0:4] == "Test" :
            folder = "Training"+classID[4:6]
          elif classID[0:8] == "Training" :
            folder = "Test"+classID[8:10]
          else :
            folder = ""
          if folder != "" :
            if weight == '2':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['F1_A'])
            elif weight == '3':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['Precision_A'])
            elif weight == '4':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['Recall_A'])
            elif weight == '5':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['F1_B'])
            elif weight == '6':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['Precision_B'])
            elif weight == '7':
              Classifier.setWeight(classID, scores[scores.Folder==folder]['Recall_B'])
            elif weight == '8':
              convex = Classifier.convexifying * scores[scores.Folder==folder]['Precision_B']
              #print Classifier.convexifying * scores[scores.Folder==folder]['Precision_B']
              #print (1.0 - Classifier.convexifying) * scores[scores.Folder==folder]['Recall_B']
              convex += (1.0 - Classifier.convexifying) * scores[scores.Folder==folder]['Recall_B']
              #print "Actual convexified weight is", convex # where the first is 0-based index of the classifier
              Classifier.setWeight(classID, convex)
            else:
              print "Unknown weight"
            if (Classifier.weightFormula == '4' or Classifier.weightFormula == '5') and Classifier.badScore(Classifier.getWeight(classID)):
  #            print "Weight must be a probability between 0.5 and 1.0"
              raise TypeError("Weight must be a probability between 0.5 and 1.0")
          else :
            raise TypeError("Classifier folder must start with Test- or Training-")
        else:
          Classifier.setWeight(classID, 1.0)
      except Exception as X:
        print X
        print "Error getting scores for classifier ", classID
        raise
      return

  getScores = staticmethod(getScores)

  def badScore(weight):
    return(weight <= 0.5 or weight >= 1.0)                                                                                     

  badScore = staticmethod(badScore)

  def nextFile():                       # reset static variables
      Classifier.geid = 0               #
      Classifier.gtid = 0               #
      Classifier.globalEvents = {}      #
      Classifier.globalArcs = {}        #
      Classifier.finalArcs = {}         # 
      Classifier.eventInstances = {}    #
      Classifier.timeMapping = {}       #
      Classifier.bestClassifier = ()    #
      Classifier.mostEvents = 0         #
      Classifier.classifierList = []    #
      Classifier.classifierWeight = {}  #
      Classifier.weight = '1'           #
      Classifier.weightFormula = '1'    #
      reltypeFile = "reltypes.dat"      #
      i = 0
      for reltypeProbability in open(reltypeFile):
        Classifier.probReltype[i] = float(reltypeProbability.strip())
        i += 1
        if i == 14:
          break
      return

  nextFile = staticmethod(nextFile)

  def getBestClassifier():
      return Classifier.bestClassifier

  getBestClassifier = staticmethod(getBestClassifier)
  
  def setBestClassifier(classID, filename, numEvents):
      # For TaskABC, best classifier is one that identified most events
      # Not relevant for TaskC only because all classifiers start with same events
      Classifier.bestClassifier = (classID, filename, numEvents)
      Classifier.mostEvents = numEvents
      return

  setBestClassifier = staticmethod(setBestClassifier)
  
  def incGeid(gKey):                # increment global event id
      Classifier.geid += 1
      Classifier.globalEvents[gKey] = Classifier.geid
      return

  incGeid = staticmethod(incGeid)

  def incGtid(gKey):                # increment global tid
      Classifier.globalEvents[gKey] = Classifier.gtid
      Classifier.gtid += 1
      return

  incGtid = staticmethod(incGtid)

  def getGlobalEvents(gKey):        # check if gKey is in global set of events/times
      return Classifier.globalEvents.get(gKey)

  getGlobalEvents = staticmethod(getGlobalEvents)
  
  def getAllGlobalEvents():        # check if gKey is in global set of events/times
      return Classifier.globalEvents

  getAllGlobalEvents = staticmethod(getAllGlobalEvents)

  def getGlobalArcs():              # get union of arcs across all classifiers
      return Classifier.globalArcs

  getGlobalArcs = staticmethod(getGlobalArcs)

  def globalClassifier(globalID, globalRID):
      return (globalID, globalRID) in Classifier.globalArcs   # if TLINK already found by another classifier

  globalClassifier = staticmethod(globalClassifier)

  def invArc(arc):
    left = arc[0]
    right = arc[1]
    return (right, left)

  invArc = staticmethod(invArc)

  def invRelTuple(relTuple):
    rp = relTuple.index(1)
    relType = Classifier.relTypes[rp]
    invrp = Classifier.invRelTypes.index(relType)
    relTuple = Classifier.emptyTuple[:invrp] + (1,) + Classifier.emptyTuple[invrp+1:] # put 1 in relevant position for relType
    return (relTuple)

  invRelTuple = staticmethod(invRelTuple)

  def setGlobalClassifier(id, rid, relTuple, classID):    # add this classifier's TLINK data to global classifier
      arc = (id, rid)
      clValues = {}
      if rid < id:
        arc = Classifier.invArc(arc)                      # invert the arc
        relTuple = Classifier.invRelTuple(relTuple)       # invert the reltype tuple
      if arc in Classifier.globalArcs:                    # if arc already in global classifier
        clValues = Classifier.globalArcs[arc]             
        for cl in Classifier.classifierList:
          if not cl in clValues:                          # make sure there is an entry for each of the classifiers
            clValues[cl] = Classifier.noneTuple
      else:                                               # otherwise
        for cl in Classifier.classifierList:              # add empty tuple for each of the classifiers
          clValues[cl] = Classifier.noneTuple
      clValues[classID] = relTuple                        # add relTuple for THIS classifier to arc
      Classifier.globalArcs[arc] = clValues               # set the relType in the new classifier
      return

  setGlobalClassifier = staticmethod(setGlobalClassifier)

  def addClassifierToArcs(classID):                       # adds empty tuples to each arc recored for this classifier
    for arc in Classifier.globalArcs:
      clValues = Classifier.globalArcs[arc]
      clValues[classID] = Classifier.noneTuple

  addClassifierToArcs = staticmethod(addClassifierToArcs)

  def setFinalClassifier(arc, relTuple):
      Classifier.finalArcs[arc] = relTuple                # set the relType in the new classifier
      return

  setFinalClassifier = staticmethod(setFinalClassifier)

  def getEventInstances(id):                              # get all event instances for global eid across all classifiers
      return Classifier.eventInstances.get(id)            # format is {eid : {classifier : (eiid1, .. , eiidn)}}
      
  getEventInstances = staticmethod(getEventInstances)      

  def setEventInstances(id, clValues):                    # set all event instances for global eid across all classifiers
      Classifier.eventInstances[id] = clValues
      
  setEventInstances = staticmethod(setEventInstances)      

  def getTimeMapping(id):                                 # get global time id
      return Classifier.timeMapping.get(id) 
      
  getTimeMapping = staticmethod(getTimeMapping)      

  def setTimeMapping(id, clValues):                       # set global time id
      Classifier.timeMapping[id] = clValues
      
  setTimeMapping = staticmethod(setTimeMapping)      

  def findGlobalKey(self, gKey, id):            # check whether the event/timex3 was already found by a previous classifier
    if self.getGlobalEvents(gKey) == None:      # if event/timex3 not already found
      if id[0] == "e":
        self.incGeid(gKey)                      # increment global event id number
      else:
        self.incGtid(gKey)
    globalID = self.getGlobalEvents(gKey)       # find global ID corresponding to event/timex3
    self.etMaps[id] = id[0]+str(globalID)       # map the classifier's event/time id to the global event/time id
    return
  
  def parseEvent(self, root):                   #
    sStart = 0                                  # position to start search for text in sentence
    stext = root.get_text()                     # full text of sentence
    for event in root.find_all("EVENT"):        # for each event in the sentence
      eid = event.get('eid')		        # get event_id
      eclass = event.get('class')		# get event_class
      etext = event.text.strip()		# get event text
      epos = stext.find(etext, sStart)          # find start position of text in sentence
      sStart = epos+1                           # next search will start from current position + 1
      gKey = str(epos)+"_"+etext                # unique key is sentence number + position in sentence + event text
      self.findGlobalKey(gKey, eid)             # set global id for classifier's event id
    return
      
  def parseTime(self, root):                    #
    sStart = 0                                  #
    stext = root.get_text()                     #
    for timex in root.find_all("TIMEX3"):       #
      tid = timex.get('tid')			# get time_id
      tType = timex.get('type')			# get time type
      tValue = timex.get('value')		# get time value
      ttext = timex.text.strip()                # get text of time
      epos = stext.find(ttext, sStart)          #
      sStart = epos+1                           #
      gKey = str(epos)+"_"+ttext                #
      self.findGlobalKey(gKey, tid)             # global id is sentence number + position in sentence + time text
      gtid = self.etMaps[tid]                   #
      if self.getTimeMapping(gtid) == None:     # append to global time dictionary
        clValues = {}                           # used for reverse mapping from global TLINKs to classifier TLINKs
        clValues[self.classID] = [tid]          #
        self.setTimeMapping(gtid, clValues)     #
      else:                         
        clValues = self.getTimeMapping(gtid)    #
        tids = clValues.get(self.classID)       #
        if tids == None:                        #
          tids = [tid]                          #
        else:
          tids.append(tid)                      #
        clValues[self.classID] = tids           #
        self.setTimeMapping(gtid, clValues)     #
    return

  def parseMakeInstance(self, root):
    for makeinst in root.find_all('MAKEINSTANCE'):
      eiid = makeinst.get('eiid')		# get eiid
      eid = makeinst.get('eventID')		# get eventID
      geid = self.etMaps[eid]
      if self.getEventInstances(geid) == None:  # append to global event instance dictionary
        clValues = {}                           # used for reverse mapping from global TLINKs to classifier TLINKs
        clValues[self.classID] = [eiid]         # set {classifier : [eiid]}
        self.setEventInstances(geid, clValues)  # set {geid : {classifier : [eiid]}}
      else:
        clValues = self.getEventInstances(geid)
        eiids = clValues.get(self.classID)
        if eiids == None:
          eiids = [eiid]
        else:
          eiids.append(eiid)                    # [eiid1, eiid2]
        clValues[self.classID] = eiids          # set {classifier : [eiid1, eiid2]}
        self.setEventInstances(geid, clValues)  # set {geid : {classifier : [eiid1, eiid2]}}
      if not eiid in self.eInstances:           # set mapping from event instance id to event id
        self.eInstances.append(eiid)
        self.eInstRefs.append(eid)		# reference back to event ID
#    print Classifier.eventInstances
    return

  def parseTlink(self,tlink):
      lid 	= tlink.get('lid')			# get link_id
      relType 	= tlink.get('relType')			# get relationship type
      timeID 	= tlink.get('timeID')			# get timeID
      relTime 	= tlink.get('relatedToTime')		# get related to time
      eiid	= tlink.get('eventInstanceID')		# get event instance id
      relEInst 	= tlink.get('relatedToEventInstance')	# get related to event instance
      rp	= self.relTypes.index(relType)		# get position of relType in relTypes
      if timeID != None:                                # TLINK is Time-x
        ID = timeID
        globalID = self.etMaps[ID]                        # get corresponding global ID
      else:                                             # TLINK is Event-x
        ID = self.eInstRefs[self.eInstances.index(eiid)] # get eid that eiid maps to
        geid = self.etMaps[ID]
        eInstNum = Classifier.eventInstances[geid][self.classID].index(eiid)
        if eInstNum == 0:                               # if first event instance, set global id to event id
          globalID = geid
        else:                                           # else set global id to event id + number
          globalID = geid+"_"+str(eInstNum)
      if relTime != None:                               # TLINK is x-Time
        rID = relTime
      else:                                             # TLINK is x-Event
        rID = self.eInstRefs[self.eInstances.index(relEInst)]       # get eid that eiid corresponds to
      globalRID = self.etMaps[rID]                      # get corresponding global ID
      relTuple = self.emptyTuple[:rp] + (1,) + self.emptyTuple[rp+1:] # put 1 in relevant position for relType
      self.arcs[(globalID, globalRID)] = relTuple
      self.setGlobalClassifier(globalID, globalRID, relTuple, self.classID)
      return

  def parseTml(self, filename):
    try:
      tree = BeautifulSoup(open(filename), "xml")	    # parse each file 
      dct = tree.find('DCT')                                # document creation time
      self.parseTime(dct)

      fullText = tree.find('TEXT')                          # parse each EVENT in the text
      self.parseEvent(fullText)

      self.parseTime(fullText)                              # parse each TIMEX3 tag in the text
      numEvents = len(self.etMaps)                          # number of events found in classifier
      if numEvents > Classifier.mostEvents:                 # best classifier is one with most events detected - used as baseline for new tml file
        self.setBestClassifier(self.classID, filename, numEvents)
        
      self.parseMakeInstance(tree)                          # parse each MAKEINSTANCE tag in the tml file

      self.addClassifierToArcs(self.classID)                # add this classifier to every existing arc
      
      for tlink in tree.find_all('TLINK'):                  # parse each TLINK tag in the tml file
        self.parseTlink(tlink)
      
    except Exception as X:
      print "Error parsing file ", filename
      raise
    return

  # Now assign probabilities to each of the relTypes in the final classifier
  def arcProbability(self):
    try:
      # Next code applies product of probabilities, or sum of logs of probabilities
      if Classifier.weightFormula == '4' or Classifier.weightFormula == '5':
        for arc in Classifier.globalArcs:                 # for each TLINK in the final classifier
          relTuple = ()
          for i in range(15):
            if Classifier.weightFormula == '4':           # use sum of logs
              prob = 0
            else:                                         # use product of probabilities
              prob = 1
            assignedByClassifier = False                  # relType was not assigned by any classifier - possible future use
            for cl in Classifier.classifierList:
                if Classifier.globalArcs[arc][cl][i] == 1:
                    assignedByClassifier = True           # if at least one classifier assigns the relType, set to True
                    if Classifier.weightFormula == '4':   # if '4' use sum of logs formula
                      prob += math.log1p(Classifier.getWeight(cl))
                    else:                                 # if '5' use product formula
                      prob *= Classifier.getWeight(cl)
                else:
                  if Classifier.weightFormula == '4':
                    prob += math.log1p(1-Classifier.getWeight(cl))
                  else:
                    prob *= 1-Classifier.getWeight(cl)
#            if assignedByClassifier == False:              # if no classifier assigned the reltype, weigh according to popularity of reltype
#              prob *= Classifier.probReltype[i]            # might implement this later
            relTuple += (prob,)            
          self.setFinalClassifier(arc, relTuple)
        return Classifier.finalArcs

      # If formula is '3' or '6' or '7', need to normalise sum of weights to 1
      if Classifier.weightFormula == '3' or Classifier.weightFormula >= '6':    
        totalWeight = float(0)
        for cl in Classifier.classifierList:
          totalWeight += Classifier.getWeight(cl)               # get total weight of all classifiers
        for cl in Classifier.classifierList:
          Classifier.setWeight(cl, Classifier.getWeight(cl)/totalWeight) # divide each classifier weight by total weight to normalise sum to 1

      # Next code applies LOSS functions
      if Classifier.weightFormula >= '6':
        for arc in Classifier.globalArcs:                 # for each TLINK in the final classifier
          relTuple = ()
          for i in range(15):
            prob = 0
            for cl in Classifier.classifierList:
                if Classifier.globalArcs[arc][cl][i] != 1:  # if prediction doesn't match
                   t = 1
                else:
                   t = 0
                if Classifier.weightFormula == '6':
                   prob += t*Classifier.getWeight(cl)       # add classifier's weight to total probability
                elif Classifier.weightFormula == '7':
                   prob += t*Classifier.getWeight(cl)     # hinge loss   
                elif Classifier.weightFormula == '8':
                   prob += t*Classifier.getWeight(cl)*Classifier.getWeight(cl)   # square loss   
                elif Classifier.weightFormula == '9':
                   prob += t*math.log1p(Classifier.getWeight(cl)) + t*math.log1p(1-Classifier.getWeight(cl))   # log loss   
            if Classifier.weightFormula == '6':
                prob = 1 - prob                           # invert the loss
                prob *= Classifier.probReltype[i]         # multiply total by prior probability of reltype
            else:
                prob *= (Classifier.probReltype[i]-1)     # multiply by probability of NOT being reltype and negate for Maximise objective function
            relTuple += (prob,)            
          self.setFinalClassifier(arc, relTuple)
        return Classifier.finalArcs

          
      for arc in Classifier.globalArcs:                   # for each TLINK in the final classifier
        if Classifier.weightFormula != '1':
          numClassifiers = float(len(Classifier.classifierList))  # total number of classifiers
        else:
          numClassifiers = float(0)                       # increment for each classifier that detects the arc
        relTuple = self.emptyTuple
        for cl in Classifier.classifierList:
          if cl in Classifier.globalArcs[arc]:            # if classifier has identified the arc
            if Classifier.weightFormula == '1' and sum(Classifier.globalArcs[arc][cl]) >= 1: 
              numClassifiers += 1                         # if weight is proportion of classifiers that identified arc, increment number
            relTuple = self.addRelTuple(relTuple, Classifier.globalArcs[arc][cl],cl) # add this classifier's probability to the existing tuple of probabilities
          else:
            relTuple = self.addRelTuple(relTuple, self.noneTuple, cl) # redundant unless we introduce NONE as valid reltype
        if Classifier.weightFormula == '1':               # if weight is number of classifiers that identified arc, need to divide probability by this number
          newTuple = ()
          for prob in relTuple:                           # for each relType
            newTuple += (prob/numClassifiers,)            # set to proportion of classifiers that assigned that relType
#          if numClassifiers > 1:                          # temporary
          self.setFinalClassifier(arc, newTuple)
        elif Classifier.weightFormula == '3':
          if sum(relTuple) >= 0.5:                        # arc is only included if total probability greater than threshold (default 0.5)
            self.setFinalClassifier(arc, relTuple)
        else:
            self.setFinalClassifier(arc, relTuple)
    except Exception as X:
      print "Error assigning probabilities to relTypes "
      raise
    return Classifier.finalArcs

  def addRelTuple(self, relTuple, addTuple, classID):   # add this classifier's probabilities to the existing reltype tuple
    i = 0
    newTuple = ()
    for prob in relTuple:
      newTuple += (relTuple[i] + Classifier.getWeight(classID)*int(addTuple[i]),) # apply classifier weight to new tuple
      i += 1
    return newTuple

  def getWeight(classID):                               # get weight for classifier
    return float(Classifier.classifierWeight[classID])

  getWeight = staticmethod(getWeight)

  def setWeight(classID, weight):                       # set weight for classifier
    Classifier.classifierWeight[classID] = weight
    return

  setWeight = staticmethod(setWeight)

  def outputClassifier(self):
    if not os.path.isdir("RESULTS"):                    # create RESULTS directory if it doesn't exist
      os.makedirs("RESULTS")
    (cl, inFile, num) = self.getBestClassifier()        # base the final tml filae on the best classifier - the one with the most events identified
#    print "Best classifier: ", inFile, "   Number of events: ", num
    pos = inFile.rfind('/')
    outFile = "RESULTS/"+inFile[pos+1:]                 # set outFile name to same as inFile name in directory RESULTS
#    print outFile
    return (cl, inFile, outFile)

  def mapResults(self, result):
    try:
      (cl, inFile, outFile) = self.outputClassifier()
      tree = BeautifulSoup(open(inFile), "xml")		  # open the final tml file
      timeML = tree.TimeML
      for tlink in tree.find_all("TLINK"):                  # remove all existing TLINKs
        timeML.TLINK.decompose() 

      linkNum = 1                                           # starting number for lid attribute
#      print result.keys()
      for arc in result.keys():                             # for each arc generate a new TLINK
        relType = Classifier.relTypes[result[arc].index(1)] # relType is one where value in tuple is 1
#        print arc, relType
        if relType != "NONE":
          left = arc[0]
          right = arc[1]
          (eiid, timeID, relTime, relEiid) = ("","","","")
          if left[0] == "e":                                  # if its an event, find corresponding eiid
            eiid = self.mapEventInstanceID(cl, left)
            if eiid == None:
              continue
          else:
            timeID = self.mapTimeID(cl, left)                 # if its a time, find corresponding timeID
            if timeID == None:
              continue
          if right[0] == "e":                                 # if related to event, find related eiid
            relEiid = self.mapEventInstanceID(cl, right)
            if relEiid == None:
              continue
          else:
            relTime = self.mapTimeID(cl, right)               # if related to time, find related timeID
            if relTime == None:
              continue
          
          # Now build the new TLINK
          newTlink = tree.new_tag("TLINK", lid="l"+str(linkNum), relType=relType)
          if eiid != "":
            newTlink["eventInstanceID"] = eiid
          if timeID != "":
            newTlink["timeID"] = timeID
          if relEiid != "":
            newTlink["relatedToEventInstance"] = relEiid
          if relTime != "":
            newTlink["relatedToTime"] = relTime
          timeML.append(newTlink)                           # add the TLINK to the file
          linkNum += 1                                      # increment the TLINK number
  #    outputText = tree.prettify()                         # get new TimeML file ready for output
      outputText = str(tree)
      f = open(outFile, 'w')
      f.write(outputText)                                   # write out the new file
      f.close()
    except Exception as X:
      print "Error mapping optimised TLINKs to new tml file"
      raise
    return

  def mapEventInstanceID(self, cl, id):                   # map global id back to eiid
    parts  = id.split('_')                                # if more than one eiid for event, global arc was set to geid_n
    clList = self.getEventInstances(parts[0])             # where geid is global event id and n is incremented by 1 for each eiid found
    if clList != None:
      eiList = clList.get(cl)                             # gets eiid list for this event and this classifier
      if eiList != None:
        if len(parts) == 1:                               # if not subscript then event had only one event instance id
          return eiList[0]                                # or this is the first event instance id for the event
        if len(eiList) >= int(parts[1]) + 1:              # else
          return eiList[int(parts[1])]                    # this is an additional event instance id (subscript _n)
    return None

  def mapTimeID(self, cl, id):                            # map global time id back to classifier time id
    clList = self.getTimeMapping(id)
    if clList != None:
      tidList = clList.get(cl)
      if tidList != None:
        return tidList[0]
    return None

  def maxProbability(self, arcs):
    for arc in arcs:
      relTuple = arcs[arc]
#      print "Before: ", relTuple
      maxValue = 0
      maxIndex = 0
      for i in range(15):
        if float(relTuple[i]) > maxValue:
          maxValue = relTuple[i]
          maxIndex = i
      relTuple = self.emptyTuple[:maxIndex] + (1,) + self.emptyTuple[maxIndex+1:] # put 1 in relevant position for relType
#      print "After : ", relTuple
      arcs[arc] = relTuple
    return arcs
      
      
    

  
