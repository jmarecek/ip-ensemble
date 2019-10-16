"""
Creates IP model and solves it using Cbc solver

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

from pyomo.environ import *
from pyomo.opt import SolverFactory
import glob

##############################################################################################################################################
## 
## Relation Algebra on TLINK Temporal Intervals. Defines R1 R2 R3 such that iR1j ^ jR2k => iR3k
## (below to be read from .DAT into Abstract model - see pyomoArcs.dat)
##  p=BEFORE; pi=AFTER; o=OVERLAP; C=CONTAINS; ci=CONTAINS_INV; b=BEGINS_ON; bi=ENDS_ON; n=NONE;

reltypes = ['p', 'pi', 'c', 'ci', 'o', 'b', 'bi', 'n']

maptuple = {'p': 0, 'pi': 1, 'c': 2, 'ci': 3, 'o': 4, 'b': 5, 'bi': 6, 'n': 7}

compositerelations = { 
       
('p','p')   : ('p'), 
('p','pi')  : ('.'), 
('p','c')   : ('p'),
('p','ci')  : ('p','ci','o','bi'),
('p','o')   : ('p','ci','o','bi'),
('p','b')   : ('p','ci','o','bi'),
('p','bi')  : ('p'), 
('p','n')   : ('.'), 
 
('pi','p')  : ('.'), 
('pi','pi') : ('pi'), 
('pi','c')  : ('pi'),
('pi','ci') : ('pi','ci','o','b'),
('pi','o')  : ('pi','ci','o','b'),
('pi','b')  : ('pi','ci','o','b'),
('pi','bi') : ('pi'),
('pi','n')  : ('.'),

('c','p')   : ('p','c','o','bi'), 
('c','pi')  : ('pi','c','o','b'), 
('c','c')   : ('c'),
('c','ci')  : ('c','ci','o'),
('c','o')   : ('co','co'),
('c','b')   : ('co','co'),
('c','bi')  : ('co','co'),
('c','n')   : ('.'),

('ci','p')  : ('p'), 
('ci','pi') : ('pi'), 
('ci','c')  : ('.'),
('ci','ci') : ('ci'),
('ci','o')  : ('p','pi','o','b','bi'), 
('ci','b')  : ('pi'),
('ci','bi') : ('p'),
('ci','n')  : ('.'), 

('o','p')   : ('p','c','o','bi'), 
('o','pi')  : ('pi','c','o','b'), 
('o','c')   : ('c','o'),
('o','ci')  : ('ci','o'),
('o','o')   : ('ci','o','b','bi'), 
('o','b')   : ('pi','c','o'),
('o','bi')  : ('p','c','o'), 
('o','n')   : ('.'), 

('b','p')   : ('p','c','o','bi'),
('b','pi')  : ('pi'), 
('b','c')   : ('pi'),
('b','ci')  : ('ci','o'),
('b','o')   : ('pi','ci','o'),
('b','b')   : ('pi'),
('b','bi')  : ('c','o'), 
('b','n')   : ('.'), 


('bi','p')  : ('p'),
('bi','pi') : ('pi','c','o','b'), 
('bi','c')  : ('p'),
('bi','ci') : ('ci','o'),
('bi','o')  : ('p','ci','o'),
('bi','b')  : ('c','o'),
('bi','bi') : ('p'), 
('bi','n')  : ('.'),

('n','p')   : ('.'), 
('n','pi')  : ('.'), 
('n','c')   : ('.'),
('n','ci')  : ('.'),
('n','o')   : ('.'),
('n','b')   : ('.'),
('n','bi')  : ('.'),
('n','n')   : ('.')}

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
    opt = SolverFactory('cbc')
    model = ConcreteModel()
    model.relTypes = Set(initialize=reltypes, ordered=True);
    model.mapTuple = Param(model.relTypes, initialize=maptuple)
    model.compositeRelations = Param (model.relTypes, model.relTypes, initialize=compositerelations) 

    model.I = RangeSet(0, 7)
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
