# Integer-Programming Ensemble of Temporal-Relations Classifiers

The extraction and understanding of temporal events and their relations are major challenges in natural language processing. Processing text on a sentence-by-sentence or expression-by-expression basis often fails, in part due to the challenge of capturing the global consistency of the text. We present an ensemble method, which reconciles the outputs of multiple classifiers of temporal expressions across the text using integer programming. Computational experiments show that the ensemble improves upon the best individual results from two recent challenges, SemEval-2013 TempEval-3 (Temporal Annotation) and SemEval-2016 Task 12 (Clinical TempEval).

_This project originated in University College Dublin Michael Smurfit Graduate Business School in 2014, as an MSc in Business Analytics, suggested by Dr. Jakub Mareček (IBM Research), and undertaken by Terri Hoare and Catherine Kerr under supervision of Dr. Paula Carroll (UCD School of Business) and Dr. Jakub Mareček (IBM Research). Already the first version provided best-known results on SemEval-2013. Subsequently, the best-known performance in terms of F1 scores has been demonstrated on the SemEval-2016 in 2016 and automation of the construction of the ensemble and ROC curves has been added in 2018. The code was written by Catherine Kerr in collaboration with Dr. Jakub Mareček._

If you use the code, please also cite our paper: 

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

The data we trained on are available from the original authors. 

Any questions can be directed to Jakub Marecek, `jakub [at] marecek.cz`.


## Set-up

### Requirements

- Install Python 2. We recommend anaconda from: https://www.anaconda.com/download/ to simplify the setup of the development environment
- Create an anaconda environment using the supplemented environment file

      conda env create -f environment.yml
    
    Activate the environment using

      conda activate ipensemble
        
    If you use an IDE, e.g. PyCharm, choose this environment as project interpreter. Otherwise just run the scripts (see below) with that environment active from the command line.


## Getting Started

If the set-up has been successfully completed, you can start running some scripts. 


### The Experiments

To recreate results from the paper, run

    python pipeline.py
    python clinicalpipeline.py
    python testall.py
    

### The Core 

To learn about the functionality of the code, see: 

    optimizer.py
    clinicaloptimizer.py

To read about the pre-processing of the data, see:

    classifier.py

Check the header of each file for more detail. 


## Thank you!

Thank you for your interest in this project! Don't hesitate to contact us if you have any questions, comments, or concerns. 
