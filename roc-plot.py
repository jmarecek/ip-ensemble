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

precomputed = [
(0.2857,	0.4501),
(0.2941,	0.4594),
(0.2917,	0.4710),
(0.2952,	0.4756),
(0.2988,	0.4687),
(0.2962, 0.4826)]
precomputedLabelsLong = [
"C2-U4-N1 (F1 = 0.3495)", 
"C2+4-U4-N1 (F1 = 0.3586)",
"C2+4-U4-U5-N1-N2 (F1 = 0.3602)", 
"C1-C2-C3-C4-U4-U5-N1-N2 (F1 = 0.3643)",    
"C2-C4-U4-U5-N1 (F1 = 0.3649)",
"C1-C2-C4-U4-U5-N1-N2 (F1 = 0.3671)"]
precomputedLabels = [
"F1 = 0.3495", 
"F1 = 0.3586",
"F1 = 0.3602", 
"F1 = 0.3643",    
"F1 = 0.3649",
"F1 = 0.3671"]


individualLables = ["C1", "C2", "C3", "C4", "N1", "N2", "U4", "U5"]
individual = [
 (0.3302, 0.2993), 
 (0.3398, 0.3318), 
 (0.3144, 0.3341), 
 (0.3320, 0.3341), 
 (0.3145, 0.2459), 
 (0.2899, 0.1972), 
 (0.3074, 0.2251), 
 (0.3224, 0.3364)
 ]

def mapnames(s):
    return "-".join(map(lambda x: corr[x], s.split("-")))


inFile = "ROC2.pkl"
outFile = 'newsfeeds.pdf' 

#inFile = "ROC3.pkl"
#outFile = 'clinical.pdf' 
annotations = False
annotationsPrecomputed = False
annotationsIndividual = True

with open(inFile, 'rb') as resultsPickle: results = pickle.load(resultsPickle)
print "In total, there were", len(results.keys()), "runs"

pairs = set(results.values() + precomputed)
#print "Distinct precision-recall pairs:", pairs
F1Scores = [get_fscore(pr) for pr in pairs]
#print "F1 scores", F1Scores
print "Maximum F1 score therein", max(F1Scores)

labels = ['F1 = %.2f' % get_fscore(results[s]) for s in results.keys()]
labelsLong = ['%s (F1 = %.2f)' % (mapnames(s[0]), get_fscore(results[s])) for s in results.keys()]

f1limit = 0.388
with open("newsfeeds.tex", 'w') as resultsTeX: 
  ss = sorted(results.keys(), key=lambda s: get_fscore(results[s]))
  s = ''.join(['%s & %.1f & %.4f & %.4f & %.4f \\\\ \n' % (mapnames(s[0]), s[1], get_fscore(results[s]), results[s][0], results[s][1]) for s in ss if get_fscore(results[s]) >= f1limit])
  resultsTeX.write(s)
  resultsTeX.close()

print "Non-dominated precision-recall pairs:" 
a = np.asarray(list(pairs))
which = np.ones(len(pairs), dtype = bool)
for i in range(len(pairs)): 
     c = a[i,:]
     if np.all(np.any(a<=c, axis=1)): print labelsLong[i]
     else: which[i] = 0

best = np.ones(len(pairs), dtype = bool)     
for i in range(len(pairs)): 
     c = a[i,:]     
     if get_fscore(c) > f1limit: None
     else: best[i] = 0

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

ax.plot(a[best,0], a[best,1], "bo")

p = np.asarray(precomputed)
ax.plot(p[:,0], p[:,1], "go")
plt.savefig(pp, format='pdf')

placedY = []
if annotationsPrecomputed:
  for cnt, (label, x, y) in enumerate(zip(precomputedLabels, p[:, 0], p[:, 1])):
    tooClose = False
    for placed in placedY: 
        print y, placedY
        if math.fabs(placed - y) < 0.015: tooClose = True
    offsetSign = -1
    if x > 0.294: offsetSign = -0.2    
    #if tooClose: offsetSign = -0.2    
    #if x < 0.22: continue
    placedY.append(y)
    print "Label", label
    plt.annotate(
        label, annotation_clip = True,
        xy=(x, y), xytext=(offsetSign * 100, 0), 
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(alpha = 0.2, arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.savefig(pp, format='pdf')

# individualLables
i = np.asarray(individual)
ax.plot(i[:,0], i[:,1], "ro")
plt.savefig(pp, format='pdf')

placedY = []
if annotationsIndividual:
  for cnt, (label, x, y) in enumerate(zip(individualLables, i[:, 0], i[:, 1])):
    tooClose = False
    for placed in placedY: 
        print y, placedY
        if math.fabs(placed - y) < 0.015: tooClose = True
    offsetSign = -1
    offsetAbove = 0.0
    if x > 0.33: offsetAbove = +10.0    
    #if tooClose: offsetSign = -0.2    
    #if x < 0.22: continue
    placedY.append(y)
    print "Label", label
    plt.annotate(
        label, annotation_clip = True,
        xy=(x, y), xytext=(offsetSign * 20, offsetAbove), 
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(alpha = 0.2, arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.savefig(pp, format='pdf')

pp.close()

for prep in precomputed:
  print get_fscore(prep)