import numpy as np
#tobii baby settings
FILTERCUTOFF = 50 #hz, cutoff of the gaussian filter
#blink handling
BLKMINDUR=0.05
EYEDEV=4
BLINK2SAC =2# deg, mininal blink before-after distance to create saccade
INTERPMD=0.1 # max duration for which blink interpolation is performed

LSACVTH=np.inf
LSACATH=np.inf
LSACMINDUR=np.inf

SACVTH=15 # velocity threshold deg/sec
SACATH=1500 # acceleration threshold deg/sec^2
SACMINDUR=0.02

FIXVTH=18
FIXATH=1500
FIXMINDUR=0.08 #second
NFIXMINDUR=0.05
# these agent extraction setting are similar to those of BabyExperiment
FIXFOCUSRADIUS=3
FIXSACTARGETDUR=0.187

PLAG = 0 # lag between agent movement and pursuit movement in sec