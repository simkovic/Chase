import numpy as np
# settings, eyelink data
FILTERCUTOFF = 50 #hz, cutoff of the gaussian filter
# blink handling
BLKMINDUR=0.05
EYEDEV=4 
BLINK2SAC =2# deg, mininal blink before-after distance to create saccade
INTERPMD=0.1 # max duration for which blink interpolation is performed
# events
LSACVTH=80
LSACATH=np.inf
LSACMINDUR=0.02

SACVTH=21
SACATH=4000
SACMINDUR=0.02

FIXVTH=6
FIXATH=800
FIXMINDUR=0.08 #second
NFIXMINDUR=0.05
FIXFOCUSRADIUS=4
FIXSACTARGETDUR=30

OLPURVTHU= SACVTH
OLPURVTHL= 4
OLPURATH= FIXATH
OLPURMD=0.08
OLFOCUSRADIUS=4 # focus radius for agents

PLAG = 0.15 # lag between agent movement and pursuit movement in sec
CLPURVTHU=SACVTH
CLPURVTHL=9
CLPURATH=FIXATH
CLPURMD=0.1
CLFOCUSRADIUS=4
CLSACTARGETDUR=0.1 # sec
CLSACTARGETRADIUS=1 # deg
MAXPHI=25#10 #