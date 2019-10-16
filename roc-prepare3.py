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


resultsFile = open("./anaforatools/ensemble-results.csv","r")
results = {}

try:
 for row in resultsFile.readlines()[1:]:
    # reffile,sysfile,num,set,pthr,opt,type,ref,pred,corr1,corr2,P,R,F1,arcs,P2,target,delta,threshold
    # Training-test-folders/Tempeval16_Platinum_clinic_00_A,1-3-7-8-12,0,A,0.6,NO,TLINK:Type:CONTAINS,2991,4261,2250,1892,0.5280450598,0.632564359
    f = row.split(",")
    if len(f) < 14: continue
    (pr, re, f1) = map(float, f[11:14])
    results[f[0]] = (pr, re)

except (KeyboardInterrupt, SystemExit):
  raise
except:
  print traceback.format_exc()

with open("ROC3.pkl", 'wb') as resultsPickle: pickle.dump(results, resultsPickle)
resultsFile.close()
