Python Code
----------

* Trajectory.py - create stimulus trajectories for the experiment

* Experiment.py - run the experiment

* Settings.py - settings used in experiment and analysis

* Constants.py - constants used in experiment and analysis

* ETData.py - contains ETData class used for pre-processing of the eyetracking data

* ETSettings.py - pre-processing settings used by the ETData class

* Preprocess.py - read the ET data, create ETData class and pre-processes the data

* ReplayData.py - displays stimulus along with gaze data and shows some results of pre-processing done by the ETData class, Coder class was used for coding and its use is explained in the [coding protocoll](http://github.com/simkovic/Chase/blob/master/evaluation/coding/tracking.prot)

* Behavior.py - performs analyses of behavioral data

* Coord.py - performs all coordinate based analyses

* Pixel.py - performs all pixel-based analyses 

* FiguresMoviesTables.py - produces the figures, movies and tables included with the manuscript

In each file there is a script at the end of the file that documents how the code was used in the research project. This script can be used to follow the analyses. In principle it can be used to reproduce the analyses, however note that it will take some months with current (2015) desktop computers to execute.


Dependencies
-----------
* [Python 2.7](http://python.org/download/releases/2.7/)

* [Psychopy](http://www.psychopy.org/) to run the Experiment

* [PyStan](http://pystan.readthedocs.org/en/latest/) for statistical inference

* [Eyelink routines](http://github.com/simkovic/eyetracking/releases/tag/pursuittrackschase) to run eyetracking experiment

* [matustools](http://github.com/simkovic/matustools/releases/tag/pursuittrackschase) to reproduce the figures

* [Libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

* [iNVT](http://ilab.usc.edu/toolkit/)

Looking to Reuse Code?
=======

The code in this repository is meant as an documentation for the published research. The code has not been developed since the publication and may no longer work with the latest versions of its dependencies (such as Psychopy or PyStan). If you are interested in an maintained and (hopefully) up-to-date version of my code please have a look at the following repositories:

* [goal-directed motion toolkit](http://github.com/simkovic/GoaldirectedMotionPsychoexperiments) to create Stimuli and run Experiments

* [eyetracker](http://github.com/simkovic/eyetracking) to run Eyetracking experiments with Psychopy

* [matusplotlib](http://github.com/simkovic/matusplotlib/) to plot figures
