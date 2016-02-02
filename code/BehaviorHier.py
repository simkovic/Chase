import numpy as np
import pylab as plt
import os, pystan
from scipy import stats
#from matustools.matustats import lognorm
#from matustools.matusplotlib import *
from multiprocessing import cpu_count
NCPU=cpu_count()
np.random.seed(4)

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

BLMAX=4 # maximum number of block per subject
T=40 # number of trials
vpn=range(20,70) # subject ids
D=loadData(vpn)

acc=-2*np.ones((len(vpn),BLMAX*T),dtype=int) # detection accuracy
rts=-2*np.ones((len(vpn),BLMAX*T)) # detection speed
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
D={'N':rts.shape[0],'T':rts.shape[1],'rts': rts,'acc':acc,'rejs':rejs}
    
# stan model

modelHier="""
data {
    int<lower=0> N; // number of subjects
    int<lower=0> T; // number of trials
    real rejs[N,T]; // trial type; 1 - minimun dist constraint, 0 - no constraint
    int acc[N,T]; // accuracy; 1 - correct, 0 - incorrect
    real rts[N,T]; // reaction time in seconds
}
parameters {
    vector<lower=0,upper=10>[2] rtac[N]; // mean rt for constraint trials
    real<lower=0,upper=10> rtsd[N];
    vector[2] hm;
    real nc[2];
    real<lower=0,upper=10> hsd[2];
    real<lower=0, upper=1> r;
}
transformed parameters {
    matrix[2,2] hS;
    hS[1,1]<-square(hsd[1]);
    hS[2,2]<-square(hsd[2]);
    hS[2,1]<-hsd[1]*r*hsd[2];
    hS[2,1]<-hS[1,2];   
}
model {
    for (n in 1:N){
    //rtac[n]~multi_normal(hm,hS);
    for (t in 1:T){
    if(rts[n,t]>-1){
        rts[n,t]~lognormal(rtac[n][1]+rejs[n,t]*nc[1],rtsd[n]);
        acc[n,t]~bernoulli_logit(rtac[n][2]+rejs[n,t]*nc[2]);
    } else if (rts[n,t]==-1){
        increment_log_prob(log1m(lognormal_cdf(30.0,
            rtac[n][1]+rejs[n,t]*nc[1],rtsd[n])));
    }}}
}"""

sm=pystan.StanModel(model_code=modelHier,model_name='rtaccHier')
#fit=sm.sampling(data=D,iter=200,chains=2,seed=np.random.randint(2**16),warmup=100,thin=1,n_jobs=6)





