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
BLKMINDUR=0.05 # minimum duration in seconds of missing value block to qualify as blink
EYEDEV=4 # minimum discrepancy between position of the two eyes that qualifies as blink
BLINK2SAC =2# deg, maximal blink before-after distance to turn blink into saccade
INTERPMD=0.1 # max duration in seconds for which blink interpolation is performed
# threshold values for basic events
# velocity in deg/s, acceleration in deg/s^2, minimum duration in seconds
SACVTH=21 
SACATH=4000
SACMINDUR=0.02

FIXVTH=6
FIXATH=800
FIXMINDUR=0.08
NFIXMINDUR=0.05  # minimum allowed duration (seconds) of gap between two consecutive fixations
FIXFOCUSRADIUS=4 # agents within this distance in degrees are flagged as being focussed
FIXSACTARGETDUR=30 # period after saccade in ms during which agents in focus are identified
# slow smooth eye movement
OLPURVTHU= SACVTH #upper threshold
OLPURVTHL= 4 # lower threshold
OLPURATH= FIXATH
OLPURMD=0.08
OLFOCUSRADIUS=4 #agents within this distance in degrees are flagged as being focussed
# fast smooth eye movement
PLAG = 0.15 # lag between agent movement and pursuit movement in sec
CLPURVTHU=SACVTH
CLPURVTHL=9
CLPURATH=FIXATH
CLPURMD=0.1
# three criteria for identification of FSM
CLFOCUSRADIUS=4 #agent's with average distance (over the entire FSM) in degrees
# has to be smaller than this threshold
CLSACTARGETDUR=0.1 # critical period after the saccade in seconds during
#which the criteria below are applied 
CLSACTARGETRADIUS=1 # agent's with average distance (deg) during critical period
# needs to be smaller than this value
MAXPHI=25# the absolute angle difference between the direction of motion of
# the agent and the direction of motion of gaze has to be smaller than this
# value in degrees
