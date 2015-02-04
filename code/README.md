Python Code
----------

* Trajectory.py - create stimulus trajectories for the experiment

* Experiment.py - run the experiment

* Settings.py - settings used in experiment and analysis

* Constants.py - constants used in experiment and analysis

* ETData.py - contains ETData class used for pre-processing of the eyetracking data

* ETSettingsAdult.py - pre-processing settings used by the ETData class

* Preprocess.py - read the ET data, create ETData class and pre-processes the data

* ReplayData.py - displays stimulus along with gaze data and shows some results of pre-processing done by the ETData class, Coder class was used for coding

* Behavior.py - performs analyses of behavioral data

* Coord.py - performs all coordinate based analyses

* Pixel.py - performs all pixel-based analyses 

* FiguresMoviesTables.py - produces the figures, movies and tables included with the manuscript

There is a script at the end of the files that actually do something. This script can be used to follow the analyses. In principle it can be used to reproduce the analyses, however note that it will take some months with current (2015) desktop computers to finish.


Dependencies
-----------
* [Python 2.7](http://python.org/download/releases/2.7/)

* [Psychopy](http://www.psychopy.org/) to run the Experiment

* [PyStan](http://pystan.readthedocs.org/en/latest/) for statistical inference

* [Eyelink routines](http://github.com/simkovic/eyetracking/blob/master/Eyelink.py) to run eyetracking experiment

* [matustools](https://github.com/simkovic/matustools) to reproduce the figures
