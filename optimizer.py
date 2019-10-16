"""
Create IP model and solves is using the Cbc solver

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

#import sys
#sys.path.append('/usr/local/lib/python2.7/dist-packages')
#import coopr.environ
#from coopr.opt import SolverFactory
#from coopr.pyomo import *
#from coopr.pyomo import Var
from pyomo.environ import *
from pyomo.opt import SolverFactory
#from pyomo import *
import glob

##############################################################################################################################################
## 
## Relation Algebra on TLINK Temporal Intervals. Defines R1 R2 R3 such that iR1j ^ jR2k => iR3k
## (below to be read from .DAT into Abstract model - see pyomoArcs.dat)
##  p=BEFORE; pi=AFTER; o=INCLUDED; oi=IS_INCLUDED; d=DURING; di=DURING_INV; Ii=SIMULTANEOUS; mi=IAFTER; m=IBEFORE; I=IDENTITY; s=BEGINS; 
##  f=ENDS; si=BEGUN_BY; fi=ENDED_BY; n=NONE

reltypes = ['p', 'pi', 'o', 'oi', 'd', 'di', 'Ii', 'mi', 'm', 'I', 's', 'f', 'si', 'fi', 'n']

maptuple = {'p': 0, 'pi': 1, 'o': 2, 'oi': 3, 'd': 4, 'di': 5, 'Ii': 6, 'mi': 7, 'm': 8, 'I': 9, 's': 10, 'f': 11, 'si': 12, 'fi': 13, 'n': 14}

compositerelations = { 
       
('p','p')  : ('p'), 
('p','pi') : ('.'), 
('p','o')  : ('p'),
('p','oi') : ('p','d','o','m','s'),
('p','d')  : ('p','d','o','m','s'),
('p','di') : ('p'),
('p','Ii') : ('p'),
('p','mi') : ('p','d','o','m','s'),
('p','m')  : ('p'),
('p','I')  : ('p'), 
('p','s')  : ('p'), 
('p','f')  : ('p','d','o','m','s'),
('p','si') : ('p'),
('p','fi') : ('p'), 
('p','n')  : ('.'), 
 
('pi','p')  : ('.'), 
('pi','pi') : ('pi',), 
('pi','o')  : ('pi','d','oi','mi','f'),
('pi','oi') : ('pi',),
('pi','d')  : ('pi','d','oi','mi','f'),
('pi','di') : ('pi',),
('pi','Ii') : ('pi',),
('pi','mi') : ('pi',),
('pi','m')  : ('pi','d','oi','mi','f'),
('pi','I')  : ('pi',), 
('pi','s')  : ('pi','d','oi','mi','f'), 
('pi','f')  : ('pi',),
('pi','si') : ('pi',),
('pi','fi') : ('pi',),
('pi','n')  : ('.'),

('o','p')  : ('p'), 
('o','pi') : ('pi','di','oi','mi','si'), 
('o','o')  : ('p','o','m'),
('o','oi') : ('.',),
('o','d')  : ('d','o','s'),
('o','di') : ('p','di','o','m','fi'),
('o','Ii') : ('o'),
('o','mi') : ('di','oi','si'),
('o','m')  : ('p'),
('o','I')  : ('o'), 
('o','s')  : ('o'), 
('o','f')  : ('d','o','s'),
('o','si') : ('di','o','fi'),
('o','fi') : ('p','o','m'),
('o','n')  : ('.'),

('oi','p')  : ('p','di','o','m','fi'), 
('oi','pi') : ('pi',), 
('oi','o')  : ('.',),
('oi','oi') : ('pi','oi','mi'),
('oi','d')  : ('d','oi','f'), 
('oi','di') : ('pi','di','oi','mi','si'),
('oi','Ii') : ('oi',),
('oi','mi') : ('pi',),
('oi','m')  : ('di','o','fi'),
('oi','I')  : ('oi',), 
('oi','s')  : ('di','o','fi'),
('oi','f')  : ('oi',),
('oi','si') : ('pi','oi','mi'),
('oi','fi') : ('di','oi','si'),
('oi','n')  : ('.'), 

('d','p')  : ('p',), 
('d','pi') : ('pi',), 
('d','o')  : ('p','d','o','m','s'),
('d','oi') : ('pi','d','oi','mi','f'),
('d','d')  : ('d'), 
('d','di') : ('.'),
('d','Ii') : ('d'), 
('d','mi') : ('pi',),
('d','m')  : ('p'),
('d','I')  : ('d'), 
('d','s')  : ('d'),
('d','f')  : ('d'),
('d','si') : ('pi','d','oi','mi','f'),
('d','fi') : ('p','d','o','m','s'),
('d','n')  : ('.'), 

('di','p')  : ('p','di','o','m','fi'),
('di','pi') : ('pi','di','oi','mi','si'), 
('di','o')  : ('di','o','fi'),
('di','oi') : ('di','oi','si'),
('di','d')  : ('.',),
('di','di') : ('di',),
('di','Ii') : ('di',), 
('di','mi') : ('di','oi','si'),
('di','m')  : ('di','o','fi'),
('di','I')  : ('di',), 
('di','s')  : ('di','o','fi'),
('di','f')  : ('di','oi','si'),
('di','si') : ('di',),
('di','fi') : ('di',),
('di','n')  : ('.'), 

('Ii','p')  : ('p'),
('Ii','pi') : ('pi',), 
('Ii','o')  : ('o'),
('Ii','oi') : ('oi',),
('Ii','d')  : ('d'),
('Ii','di') : ('di',),
('Ii','Ii') : ('Ii',), 
('Ii','mi') : ('mi',),
('Ii','m')  : ('m'),
('Ii','I')  : ('Ii',), 
('Ii','s')  : ('s'),
('Ii','f')  : ('f'),
('Ii','si') : ('si',),
('Ii','fi') : ('fi',),
('Ii','n')  : ('.'), 

('mi','p')  : ('p','di','o','m','fi'),
('mi','pi') : ('pi',), 
('mi','o')  : ('d','oi','f'),
('mi','oi') : ('pi',),
('mi','d')  : ('d','oi','f'),
('mi','di') : ('pi',),
('mi','Ii') : ('mi',), 
('mi','mi') : ('pi',),
('mi','m')  : ('I','Ii','s','si'),
('mi','I')  : ('mi',), 
('mi','s')  : ('d','oi','f'),
('mi','f')  : ('mi',),
('mi','si') : ('pi',),
('mi','fi') : ('mi',),
('mi','n')  : ('.'), 
 
('m','p')  : ('p'),
('m','pi') : ('pi','di','oi','mi','si'), 
('m','o')  : ('p'),
('m','oi') : ('d','o','s'),
('m','d')  : ('d','o','s'),
('m','di') : ('p'),
('m','Ii') : ('m'), 
('m','mi') : ('I','Ii','f','fi'),
('m','m')  : ('p'),
('m','I')  : ('m'), 
('m','s')  : ('m'),
('m','f')  : ('d','o','s'),
('m','si') : ('m'),
('m','fi') : ('p'),
('m','n')  : ('.'), 

('I','p')  : ('p'),
('I','pi') : ('pi',), 
('I','o')  : ('o'),
('I','oi') : ('oi',),
('I','d')  : ('d'),
('I','di') : ('di',),
('I','Ii') : ('I'), 
('I','mi') : ('mi',),
('I','m')  : ('m'),
('I','I')  : ('I'), 
('I','s')  : ('s'),
('I','f')  : ('f'),
('I','si') : ('si',),
('I','fi') : ('fi',),
('I','n')  : ('.'), 

('s','p')  : ('p'),
('s','pi') : ('pi',), 
('s','o')  : ('p','o','m'),
('s','oi') : ('d','oi','f'),
('s','d')  : ('d'),
('s','di') : ('p','di','o','m','fi'),
('s','Ii') : ('s'), 
('s','mi') : ('mi',),
('s','m')  : ('p'),
('s','I')  : ('s'), 
('s','s')  : ('s'),
('s','f')  : ('.'),
('s','si') : ('I','Ii','s','si'),
('s','fi') : ('p','o','m'),
('s','n')  : ('.'), 

('f','p')  : ('p'),
('f','pi') : ('pi',), 
('f','o')  : ('d','o','s'),
('f','oi') : ('pi','oi','mi'),
('f','d')  : ('d'),
('f','di') : ('pi','di','oi','mi','si'),
('f','Ii') : ('f'), 
('f','mi') : ('pi',),
('f','m')  : ('m'),
('f','I')  : ('d'), 
('f','s')  : ('.'),
('f','f')  : ('d'),
('f','si') : ('pi','oi','mi'),
('f','fi') : ('I','Ii','f','fi'),
('f','n')  : ('.'), 

('si','p')  : ('p','di','o','m','fi'),
('si','pi') : ('pi',), 
('si','o')  : ('di','o','fi'),
('si','oi') : ('oi',),
('si','d')  : ('d','oi','f'),
('si','di') : ('di',),
('si','Ii') : ('si',), 
('si','mi') : ('mi',),
('si','m')  : ('di','o','fi'),
('si','I')  : ('si',), 
('si','s')  : ('I','Ii','s','si'),
('si','f')  : ('oi',),
('si','si') : ('si',),
('si','fi') : ('di',),
('si','n')  : ('.'), 

('fi','p')  : ('p'),
('fi','pi') : ('pi','di','oi','mi','si'), 
('fi','o')  : ('o'),
('fi','oi') : ('di','oi','si'),
('fi','d')  : ('d','o','s'),
('fi','di') : ('di',),
('fi','Ii') : ('fi',), 
('fi','mi') : ('di','oi','si'),
('fi','m')  : ('m'),
('fi','I')  : ('fi',), 
('fi','s')  : ('o'),
('fi','f')  : ('I','Ii','f','fi'),
('fi','si') : ('di',),
('fi','fi') : ('fi',),
('fi','n')  : ('.'),

('n','p')  : ('.'), 
('n','pi') : ('.'), 
('n','o')  : ('.'),
('n','oi') : ('.'),
('n','d')  : ('.'),
('n','di') : ('.'),
('n','Ii') : ('.'),
('n','mi') : ('.'),
('n','m')  : ('.'),
('n','I')  : ('.'), 
('n','s')  : ('.'), 
('n','f')  : ('.'),
('n','si') : ('.'),
('n','fi') : ('.'), 
('n','n')  : ('.')}

###########################################################################################################
#
## Graph functions

# Nodes_init here for future use 
def Nodes_init(v):
    retval = []
    for arc in v:
        if not arc[0] in retval:
            retval.append(arc[0])
        if not arc[1] in retval:
            retval.append(arc[1])
    return retval

# Arcs_init adds arcs to model         
def Arcs_init(v):
    numVars = 0
    retval = []
    for arc in v:
        retval.append(arc)
        numVars += 1
#    print "Number of decision variables: ", numVars*15
    return retval

# Add connected tri-graphs to model
def Connected_Arcs_init(v):
    retval = []
    for arc1 in v:
        i = arc1[0]
        ij = arc1[1]
        for arc2 in v:
            if arc1 != arc2: 
               jk = arc2[0]
               k = arc2[1]
               if ij == jk:
                  ik = (i,k) 
                  if ik in v:
                    tuple = ()   
                    tuple += (i,) + (ij,) + (k,) 
                    retval +=(tuple,)
    return retval                 

# Loads 2-d arc/reltype weights into model
def Rels_init(model, left, right, i):
    global v
    index = (left, right)
    return v[index][i]

# Objective function
def Obj_rule(model):
  return summation(model.Rels, model.x)

# Constraint - only one reltype can be assigned per arc
def OnlyOneReltype_rule(model, left, right):
    global numConstraints
    index = (left, right)
    numConstraints += 1
    return sum(model.x[index,i] for i in model.I) == 1

# Constraint - transitive closure rules
def Transitivity_rule(model, i , j, k, arc1IntType, arc2IntType):
    global numConstraints
    arc1 = (i,j)
    arc2 = (j,k)
    arc3 = (i,k)
    arc1RT = model.mapTuple[arc1IntType]
    arc2RT = model.mapTuple[arc2IntType]
    arc3RT = model.compositeRelations[arc1IntType, arc2IntType]
    if arc3RT[0] == '.':
        return Constraint.Feasible
    numConstraints += 1
    return model.x[arc1,arc1RT] + model.x[arc2,arc2RT] - sum(model.x[arc3,model.mapTuple[arc3RT[j]]] for j in range(len(arc3RT))) <=1            


#############################################################################################################    
#
# Optimise Final Classifier using pyomo model
#
v = {}              # contains dictionary passed from pre-processor
numConstraints = 0
def main(inDict):
    global v
    global numConstraints
    numConstraints = 0
    v = inDict
    opt = SolverFactory('cplex')
    model = ConcreteModel()
    model.relTypes = Set(initialize=reltypes, ordered=True);
    model.mapTuple = Param(model.relTypes, initialize=maptuple)
    model.compositeRelations = Param (model.relTypes, model.relTypes, initialize=compositerelations) 

    model.I = RangeSet(0, 14)
    model.Arcs = Set(initialize=Arcs_init(v))
    model.Connected_Arcs = Set(initialize=Connected_Arcs_init(v), dimen=3)
    model.Rels = Param(model.Arcs, model.I, initialize=Rels_init)
    model.x = Var(model.Arcs, model.I, domain=Binary)
    model.xProb = Param(model.Arcs, model.I)
    model.Obj = Objective(rule=Obj_rule, sense=maximize)
    model.OnlyOneReltype = Constraint(model.Arcs, rule=OnlyOneReltype_rule) 
    model.Transitivity = Constraint(model.Connected_Arcs, model.relTypes, model.relTypes, rule=Transitivity_rule)
    results = opt.solve(model)
    rDict = {}
    for (arcFrom, arcTo, index) in model.x.keys():
        arc = (arcFrom, arcTo)
        if not rDict.has_key( arc ): rDict[arc] = []
        rDict[arc].append(index)
    return rDict

if __name__ == "__main__":
    main(sys.argv[1])
