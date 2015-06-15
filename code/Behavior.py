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
import pylab as plt
import os, pystan
from scipy import stats
from matustools.matustats import lognorm
#from matustools.matusplotlib import *
from multiprocessing import cpu_count
NCPU=cpu_count()
np.random.seed(42)

ipath= os.getcwd().rstrip('code')+'behavioralOutput'+os.path.sep
opath= os.path.sep.join([os.getcwd().rstrip('code')+'evaluation','behData',''])

def loadData(vpn, verbose=False):
    D=[]
    for vp in vpn:
        dat=np.loadtxt(ipath+'vp%03d.res'%vp)
        if verbose: print vp, dat.shape
        D.append(dat[dat[:,1]>0,:])
    D=np.concatenate(D,0)
    return D

################################################
#
#
#
################################################

BLMAX=4 # maximum number of block per subject
T=40 # number of trials
vpn=range(20,70) # subject ids
D=loadData(vpn)

acc=-np.ones((len(vpn),BLMAX*T),dtype=int) # detection accuracy
rts=np.zeros((len(vpn),BLMAX*T))*np.nan # detection speed
rejs=-np.ones((len(vpn),BLMAX*T),dtype=int) # 1 - trial with min dist 3 deg
for i in range(len(vpn)):
    sel= D[:,0]==vpn[i]
    acc[i,:sel.sum()]= D[sel,-1]
    rts[i,:sel.sum()]= D[sel,6]
    rejs[i,:sel.sum()]=np.int32(D[sel,3]<35)
# code ommisions with -1    
acc[rts==30]=-1
rts[rts==30]=-1
# data format for stan
D=[]
for vp in range(rejs.shape[0]):
    sel=acc[vp,:]>-1
    omit=[(rejs[vp,~sel]==0).sum(),(rejs[vp,~sel]==1).sum()]
    D.append({'T':sel.sum(),'omit':omit,'rts': rts[vp,sel],
           'acc':acc[vp,sel],'rejs':rejs[vp,sel],'ind':np.arange(BLMAX*T)[sel]/10000.0})

print rts.shape
successs


BLMAX=25 # maximum number of block per subject
T=40 # number of trials
vpn=[1,2,3,4] # subject ids
D=loadData(vpn)

acc=-np.ones((len(vpn),BLMAX*T),dtype=int) # detection accuracy
rts=np.zeros((len(vpn),BLMAX*T))*np.nan # detection speed
rejs=-np.ones((len(vpn),BLMAX*T),dtype=int) # 1 - trial with min dist 3 deg
for i in range(len(vpn)):
    sel= D[:,0]==vpn[i]
    acc[i,:sel.sum()]= D[sel,-1]
    rts[i,:sel.sum()]= D[sel,6]
    rejs[i,:sel.sum()]=np.int32(D[sel,3]<35)
# code ommisions with -1    
acc[rts==30]=-1
rts[rts==30]=-1
# data format for stan
D=[]
for vp in range(rejs.shape[0]):
    sel=acc[vp,:]>-1
    omit=[(rejs[vp,~sel]==0).sum(),(rejs[vp,~sel]==1).sum()]
    D.append({'T':sel.sum(),'omit':omit,'rts': rts[vp,sel],
           'acc':acc[vp,sel],'rejs':rejs[vp,sel],'ind':np.arange(BLMAX*T)[sel]/10000.0})

# model definition

model = """
data {
    int<lower=0> T; // number of trials
    vector[T] rejs; // trial type; 1 - minimun dist constraint, 0 - no constraint
    int<lower=0> omit[2]; // number of omissions for each trial type
    int acc[T]; // accuracy; 1 - correct, 0 - incorrect
    vector[T] rts; // reaction time in seconds
    vector[T] ind; // trial order
}
parameters {
    real<lower=0,upper=10> rtmc; // mean rt for constraint trials
    real<lower=0,upper=10> rtmd; // mean rt no constraint
    real<lower=0,upper=10> rtsc;
    real<lower=0,upper=10> rtsd;
    real<lower=-10,upper=0> ttrend;
    real<lower=0,upper=1>  csrate;
    real<lower=0,upper=1>  dsrate;
}
model {
    rts~ lognormal(rejs*rtmc+(1-rejs)*rtmd, 
                   rejs*rtsc+(1-rejs)*rtsd);
    acc~ bernoulli(rejs*csrate+(1-rejs)*dsrate);
}
"""

modelCensor=model[:-2]+'''    increment_log_prob(omit[2]*log1m(lognormal_cdf(30.0,rtmc,rtsc)));
    increment_log_prob(omit[1]*log1m(lognormal_cdf(30.0,rtmd,rtsd)));
}'''

modelTrend=modelCensor.replace('*rtmd','*rtmd+ttrend*ind')

#compile models
SM=[]
for m in [modelCensor]:
    SM.append(pystan.StanModel(model_code=m))
# run stan
F=[]
for sm in SM:
    F.append([])
    for vp in range(rejs.shape[0]):
        fit=sm.sampling(data=D[vp],iter=1000,chains=8,
                seed=np.random.randint(2**16),warmup=200,thin=2,n_jobs=NCPU)
        print fit
        F[-1].append(fit)

# plot fit and save estimates
for f in F:
    rtmc=[]
    plt.figure(figsize=(16,12))
    for vp in range(4):
        d=rts[vp,rejs[vp,:]==1]
        d=d[d>0]
        plt.subplot(2,2,vp+1)
        x=np.linspace(1,30,31)
        plt.hist(d,bins=x,facecolor='w')
        plt.ylim([0,140])
        plt.xlim([0,30])
        w=f[vp].extract()
        varc=np.square(w['rtsc'])
        vard=np.square(w['rtsd'])
        rtmc.append([np.exp(w['rtmc']+varc/2.),
                     np.exp(w['rtmd']+vard/2.),
                     #np.sqrt((np.exp(varc)-1)*np.exp(2*w['rtmc']+varc)),
                     #np.sqrt((np.exp(vard)-1)*np.exp(2*w['rtmd']+vard)),
                     w['csrate'],w['dsrate'],
                     w['rtmc'],w['rtsc']])
        plt.plot(x-0.5,d.size*lognorm(mu=w['rtmc'].mean(),
                    sigma=w['rtsc'].mean()).pdf(x-0.5))
    rtmc=np.array(rtmc,ndmin=2)
np.save(os.getcwd().rstrip('code')+'evaluation/rtmc',rtmc)
np.save(os.getcwd().rstrip('code')+'evaluation/rejs',rejs)
np.save(os.getcwd().rstrip('code')+'evaluation/rts',rts)
np.save(os.getcwd().rstrip('code')+'evaluation/acc',acc)
