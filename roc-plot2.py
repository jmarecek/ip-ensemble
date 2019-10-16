"""
Construct the other ROC plot.

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

import traceback
import numpy as np
import pickle
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

def get_fscore(pr):
    (p, r) = pr
    if p+r == 0: 
        return 0 
    return 2.0*p*r/(p+r) 

corr = {
"1": "C1",
"2": "C2",
"3": "C3",
"4": "C4",
"5": "N1",
"6": "U1",
"7": "U2",
"8": "U3",
"9": "U4",
"A": "U5",
"B": "N2"
}

def mapnames(s):
    return "-".join(map(lambda x: corr[x], s.split("-")))

inFile = "ROC2.pkl"
outFile = 'newsfeeds.pdf' 

#inFile = "ROC3.pkl"
#outFile = 'clinical.pdf' 
annotations = False

with open(inFile, 'rb') as resultsPickle: results = pickle.load(resultsPickle)
pairs = set(results.values())
print "Distinct precision-recall pairs:", pairs

print "Non-dominated precision-recall pairs:" 
a = np.asarray(list(pairs))
which = np.ones(len(pairs), dtype = bool)
for i in range(len(pairs)): 
     c = a[i,:]
     if np.all(np.any(a<=c, axis=1)): print c
     else: which[i] = 0

tolerance = 0.1
close = np.ones(len(pairs), dtype = bool)     
print "Precision-recall pairs within", tolerance, "tolerance:" 
for i in range(len(pairs)): 
     c = a[i,:]     
     if np.all(np.any(a<=c+tolerance, axis=1)): print c
     #else: close[i] = 0

plt.style.use('grayscale')
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
scatter, ax = plt.subplots()
prop = fm.FontProperties(fname='./Calibri.ttf')
plt.grid(True, alpha=0.2)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)
#ax.set_xlim(ax.get_xlim()[::-1])
#ax.set_ylim(ax.get_ylim()[::-1])

#scatter.suptitle('ROC Curve', fontproperties=prop, size=10)
# Convert to the false positive rate
#a[0,:] = 1 - a[0,:]
ax.set_xlabel('Precision', fontproperties=prop, size=10)
ax.set_ylabel('Recall', fontproperties=prop, size=10)   # relative to plt.rcParams['font.size']
ax.plot(a[which,0], a[which,1], "o")

labels = ['F1 = %.2f' % get_fscore(results[s]) for s in results.keys()]
#labels = ['%s (F1 = %.2f)' % (mapnames(s[0]), get_fscore(results[s])) for s in results.keys()]
placedY = []
if annotations:
  for cnt, (label, x, y) in enumerate(zip(labels, a[:, 0], a[:, 1])):
    if not which[cnt]: continue 
    tooClose = False
    for placed in placedY: 
        print y, placedY
        if math.fabs(placed - y) < 0.015: tooClose = True
    if tooClose: continue
    offsetSign = 1
    if x < 0.22: continue
    if x > 0.28: offsetSign = -1
    placedY.append(y)
    plt.annotate(
        label, annotation_clip = True,
        xy=(x, y), xytext=(offsetSign * 100, 0), 
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(alpha = 0.2, arrowstyle = '->', connectionstyle='arc3,rad=0'))

pp = PdfPages(outFile)
plt.savefig(pp, format='pdf')

ax.plot(a[close,0], a[close,1], "o", alpha=0.2)
plt.savefig(pp, format='pdf')

pp.close()