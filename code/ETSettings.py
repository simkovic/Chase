## The MIT License (MIT)
##
## Copyright (c) <2015> <Matus Simkovic>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

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
