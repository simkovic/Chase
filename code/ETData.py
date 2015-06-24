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
import matplotlib as mpl
from psychopy import core,event,visual
from scipy import signal
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.stats import nanmean
from copy import copy
import time,os

from Settings import *
from ETSettings import *
from Constants import *

PATH = os.getcwd().rstrip('code')+'evaluation'+os.path.sep

########################################
# helper functions
def t2f(t,tser):
    out=np.diff(tser<t).nonzero()[0]
    if out.size==0: return tser.size
    else: return out[0]
def mergeEvents(events):
    maxx=events[0][-1][1]
    for i in range(1, len(events)): maxx=max(maxx,events[i][-1][1])
    out=[]
    for i in range(maxx):
        for ev in events:
            for k in range(len(ev)):
                if ev[k][0]==i:out.append(ev[k])
    return out

def norm(x,axis=0):
    x=np.power(x,2)
    x=x.sum(axis)
    return np.sqrt(x)

def interpRange(xold,yold,xnew):
    ''' interpolate values for elements in xnew
        based on the mapping between xold and yold
    '''
    ynew=np.float32(xnew.copy())
    f=interp1d(xold,yold,bounds_error=False)
    
    for i in range(xnew.size):
        try: ynew[i] =f(xnew[i])
        except: 
            print 'interp',i,xnew[i],xold[0].shape,xold.shape
            np.save('xnew.npy',xnew)
            np.save('xold.npy',xold)
            raise
    return ynew

########################################################################
# conversion functions, avoid invoking monitors and can convert x,y axis separately
def myPix2cm(pix,cent,width):
    return (pix-cent)/float(cent)/2.0*width
def myCm2pix(cm,cent,width):
    return cm/float(width) *cent*2.0 +cent
def myCm2deg(cm,dist):
    return cm/(dist*0.017455) 
def myDeg2cm(deg,dist):
    return deg*dist*0.017455
def myPix2deg(pix,dist,cent,width):
    cm= myPix2cm(pix,cent,width)
    #return np.arctan((pix/37.795275591)/float(dist))*180/np.pi
    return myCm2deg(cm,dist)
def myDeg2pix(deg,dist,cent,width):
    cm = myDeg2cm(deg,dist)
    return myCm2pix(cm,cent,width)
####################################################################
# functions that determine whether agent is focused during an event 
def selectAgentCL(dist,dev,hz):
    #a=(dist.max(0)<3).nonzero()[0]
    #b=norm(np.median(dev,axis=0),axis=1)<MAXPHI
    #b=np.mod(dev/np.pi*180.0+180,360)-180
    #print (np.abs(dev)<MAXPHI).mean(axis=0)
    b=(np.abs(dev)<MAXPHI).mean(axis=0)>0.5
    a=np.logical_and(nanmean(dist,0)<CLFOCUSRADIUS,b)
    a= np.logical_or(nanmean(dist[:int(hz*CLSACTARGETDUR)],0)
        < CLSACTARGETRADIUS,a).nonzero()[0]
    return a
def selectAgentOL(dist):
    a=(np.nanmax(dist,axis=0)<OLFOCUSRADIUS).nonzero()[0]
    return a

def selectAgentFIX(dist,hz):
    a=(nanmean(dist[:int(hz*FIXSACTARGETDUR)])<FIXFOCUSRADIUS).nonzero()[0]
    return a

def selectAgentTRACKING(fs,fe,evs):
    acounter=np.zeros(50)
    tot=0
    for ev in evs:
        if ev[-1]==CLPUR and fs<=ev[0] and fe>=ev[1]:
            acounter[ev[2:-1]]+=1
            tot+=1
    if tot==0:
        for ev in evs:
            if ev[-1]==OLPUR and fs<=ev[0] and fe>=ev[1]:
                acounter[ev[2:-1]]+=1
                tot+=1
    out=(acounter>=min(0.5*float(tot),3)).nonzero()[0].tolist()
    if len(out)==0: print 'NO AGENTS SELECTED !!!'
    return out,[[fs,fe]]*len(out)
###########################################################3
# basic preprocessing and filtering
def filterGaussian(x,hz):
    #return self.filterCausal(x)
    # cutoff in hz
    sigma=hz/float(FILTERCUTOFF)
    #print np.ceil(12*sigma), sigma# sigma, self.hz, cutoff
    fg=signal.gaussian(np.ceil(5*sigma),sigma,sym=False)
    fg=fg/fg.sum()
    #print fg.shape,x.shape
    return np.convolve(x,fg,mode='same')

def computeVelocity(tser,hz,filt=False):
    """returns velocity in deg per second"""
    vel=(np.sqrt(np.diff(tser[:,7])**2
        +np.diff(tser[:,8])**2)*hz)
    vel=np.concatenate([[vel[0]],vel])
    if filt: vel=filterGaussian(vel,hz)
    return vel
def computeAcceleration(tser,hz ,filt=False):
    vel = computeVelocity(tser,hz,filt)
    acc=np.diff(vel*hz)
    acc=np.concatenate([acc,[acc[-1]]])
    if filt: acc=filterGaussian(acc,hz)
    return acc
################################################################
# helper functions for identification of basic events
def computeState(isFix,md):
    ''' generic function that determines event start and end
        isFix - 1d array, time series with one element for each
            gaze data point, 1 indicates the event is on, 0 - off
        md - minimum event duration
        returns
            list with tuples with start and end for each
                event (values in frames)
            timeseries analogue to isFix but the values
                correspond to the list
    '''
    fixations=[]
    if isFix.sum()==0: return np.int32(isFix),[]
    fixon = np.bitwise_and(isFix,
        np.bitwise_not(np.roll(isFix,1))).nonzero()[0].tolist()
    fixoff=np.bitwise_and(np.roll(isFix,1),
        np.bitwise_not(isFix)).nonzero()[0].tolist()
    if len(fixon)==0 and len(fixoff)==0: fixon=[0]; fixoff=[isFix.size-1]
    if fixon[-1]>fixoff[-1]:fixoff.append(isFix.shape[0]-1)
    if fixon[0]>fixoff[0]:fixon.insert(0,0)
    if len(fixon)!=len(fixoff): print 'invalid fixonoff';raise TypeError
    for f in range(len(fixon)):
        fs=fixon[f];fe=(fixoff[f]+1);dur=fe-fs
        if  dur<md[0] or dur>md[1]:
            isFix[fs:fe]=False
        else: fixations.append([fs,fe-1])
    #fixations=np.array(fixations)
    return isFix,fixations

def tseries2eventlist(tser):
    ''' translates from the time series to the event list
        representation '''
    tser=np.int32(tser)
    if tser.sum()==0: return []
    d=np.bitwise_and(tser,np.bitwise_not(np.roll(tser,1)))
    on = (d[1:].nonzero()[0]+1).tolist()
    d=np.bitwise_and(np.roll(tser,1),np.bitwise_not(tser))
    off=d[1:].nonzero()[0].tolist()
    if len(off)==0:off.append(tser.shape[0]-1)
    if len(on)==0: on.insert(0,0)
    if on[-1]>off[-1]: off.append(tser.shape[0]-1)
    if on[0]>off[0]: on.insert(0,0)
    if len(on)!=len(off): print 'invalid fixonoff';raise TypeError
    out=np.array([on,off]).T
    return out.tolist()
################################################################
# functions for identification of basic events   
def computeFixations(tser,vel,acc,hz):
    ''' identify fixations
        tser - 1d array, gaze data, not used by the computation
        vel - 1d array, gaze velocity
        acc - 1d array, gaze acceleration
        hz - gaze data recording rate
    ''' 
    isFix=np.logical_and(vel<FIXVTH,np.abs(acc)<FIXATH)
    if isFix.sum()==0: return np.int32(isFix),[]
    fev=tseries2eventlist(isFix)
    fe=0
    for fix in fev:
        if  fix[0]-fe <NFIXMINDUR*hz:
            isFix[fe:(fix[1]+1)]=True
        fe=fix[1] 
    return computeState(isFix,[FIXMINDUR*hz,np.inf])

##def computePursuit(tser,vel,acc,hz):
##    isFix=np.logical_and(vel<FIXVTH,np.abs(acc)<FIXATH)
##    isFix,b=computeState(isFix,[FIXMINDUR*hz,np.inf])
##    isPur=np.logical_and(np.logical_and(vel<PURVTHU,np.abs(acc)<PURATHU),
##        np.logical_or(vel>PURVTHL,np.abs(acc)>PURATHL))
##    isPur=np.logical_and(isPur,~isFix)
##    return computeState(isPur,[PURMINDUR*hz,np.inf])
def computeOLpursuit(tser,vel,acc,hz):
    ''' identify slow smooth eye movements
        tser - 1d array, gaze data, not used by the computation
        vel - 1d array, gaze velocity
        acc - 1d array, gaze acceleration
        hz - gaze data recording rate
    ''' 
    temp=np.logical_and(vel>OLPURVTHL,
        np.logical_and(vel<OLPURVTHU,np.abs(acc)<OLPURATH))
    return computeState(temp,[OLPURMD*hz,np.inf])
def computeCLpursuit(tser,vel,acc,hz):
    ''' identify fast smooth eye movements
        tser - 1d array, gaze data, not used by the computation
        vel - 1d array, gaze velocity
        acc - 1d array, gaze acceleration
        hz - gaze data recording rate
    ''' 
    temp=np.logical_and(vel>CLPURVTHL,
        np.logical_and(vel<CLPURVTHU,np.abs(acc)<CLPURATH))
    return computeState(temp,[CLPURMD*hz,np.inf])

def computeSaccades(tser,vel,acc,hz):
    ''' identify saccades
        tser - gaze data
        vel - 1d array, gaze velocity
        acc - 1d array, gaze acceleration
        hz - gaze data recording rate
    ''' 
    isFix=np.logical_or(np.isnan(vel),
        np.logical_or(vel>SACVTH,np.abs(acc)>SACATH))
    discard, b = computeState(np.logical_or(vel>SACVTH,np.abs(acc)>SACATH),[SACMINDUR*hz,np.inf])
    #compute info statistics: length, duration
    for sac in b:
        length=((tser[sac[0],7]-tser[sac[1],7])**2 +
                (tser[sac[0],8]-tser[sac[1],8])**2)**0.5
        dur=(tser[sac[1],0]-tser[sac[0],0])
        sac.extend([length,dur])
    a,discard= computeState(isFix,[SACMINDUR*hz,np.inf])
    return a,b
########################################################
# blink handling
def interpolateBlinks(t,d,hz):
    ''' Interpolate short missing intervals
        d - 1d array, time series with gaze data, np.nan indicates blink
        hz - gaze data recording rate
    '''
    isblink= np.isnan(d)
    if isblink.sum()<2 or isblink.sum()>(isblink.size-2): return d
    blinkon = np.bitwise_and(isblink,np.bitwise_not(
        np.roll(isblink,1))).nonzero()[0].tolist()
    blinkoff=np.bitwise_and(np.roll(isblink,1),
        np.bitwise_not(isblink)).nonzero()[0].tolist()
    if len(blinkon)==0 and len(blinkoff)==0: return d
    #print 'bla',len(blinkon), len(blinkoff)
    if blinkon[-1]>blinkoff[-1]: blinkoff.append(t.size-1)
    if blinkon[0]>blinkoff[0]: blinkon.insert(0,0)
    if len(blinkon)!=len(blinkoff):
        print 'Blink Interpolation Failed'
        raise TypeError
    f=interp1d(t[~isblink],d[~isblink],bounds_error=False)
    for b in range(len(blinkon)):
        bs=blinkon[b]-1
        be=(blinkoff[b])
        if (be-bs)<INTERPMD*hz:
            d[bs:be]=f(t[bs:be])
            #for c in [7,8]: tser[bs:be,c]=np.nan
    return d

def computeBlinks(tser,hz):
    ''' identify blinks
        tser - gaze data
        hz - gaze data recording rate
    ''' 
    isblink=np.isnan(tser[:,7])
    return computeState(isblink,[BLKMINDUR*hz*0,np.inf])
###############################################################                   
class ETData():
    def __init__(self,dat,calib,t0,info,recTime="0:0:0",fs=None,
                 focus=BINOCULAR,msgs=[]):
        '''dat - eyetracking data with columns: 0-time,1-left eye x,
                2 - left eye y, 3 - left eye pupil size, 4 - right eye x,
                5 - right eye y, 6 - right eye pupil size
            calib - calibration info
            t0 - tuple with information on trial start and trial end
            info - [subject id, block id, trial id, True if both eyes]
            recTime - string with recording time in H:M:S format
            fs - ids of the trajectory frames corresponding to the gaze data
            focus - dominant eye, if BINOCULAR eyes will be averaged
            msgs - additional messages for debuging of gaze-contingent ET
        '''
        self.calib=calib
        self.gaze=dat
        self.vp=int(info[0])
        self.block=int(info[1])
        self.trial=int(info[2])
        self.hz=float(info[3])
        self.eye=info[4]
        self.t0=[t0[1]-t0[0],t0[2]-t0[0]]
        self.fs=fs
        self.msgs=msgs
        if self.fs==None or self.fs.size==0: self.fs= self.computeFs()
        #fsa=self.computeFs()
        #m=min(self.fs.shape[0],fsa.shape[0])
        #print np.round(np.max(np.abs(self.fs[:m,:2]-fsa[:m,:])),1), self.t0[1]-self.t0[0]
        self.focus=focus
        if self.t0[0]>=0: self.ts=min((dat[:,0]>(self.t0[0])).nonzero()[0])
        else: self.ts=-1
        #print 'hhh',self.t0, self.ts
        
        if  self.t0[1]>0 and len((dat[:,0]>(self.t0[1])).nonzero()[0])>0: 
            self.te=min((dat[:,0]>(self.t0[1])).nonzero()[0])
        elif self.t0[1]>0: self.te=dat.shape[0]-1
        self.recTime=datetime.strptime(recTime,"%H:%M:%S")
        
    def computeFs(self):
        '''if the frame ids for the gaze points was not recorded
            this function can be used to interpolate frame ids
            assumes equal frame spacing
        '''
        path = getcwd()
        path = path.rstrip('code')
        try: f=open(path+'input/vp%03d/SettingsExp.pkl'%self.vp)
        except: f=open(path+'input/vp%03d/Settings.pkl'%self.vp)
        while f.readline().count('refreshRate')==0: pass
        f.readline();
        monhz=float(f.readline().lstrip('F').rstrip('\r\n'))
        f.close()
        dur=self.t0[1]-self.t0[0]
        inc=1000/monhz
        N=min(int(round((dur)/inc)), int(Q.trialDur*monhz)+1)
        return np.array([range(N),np.linspace(inc/2.0,dur-inc,N)]).T
#######################################################
# helper functions for drift correction and extraction of basic events   
    @staticmethod
    def helpf(fev,inds):
        #if inds[1]==579: print fev[-1],inds
        out=[]
        for ff in fev:
            if ((ff[0]>=inds[0] and ff[0]<=inds[1])
                or (ff[1]>=inds[0] and ff[1]<=inds[1])):
                temp=[max(ff[0]-inds[0],0),min(ff[1]-inds[0],inds[1]-inds[0]-1)]+ff[2:]
                out.append(temp)
        return out
    @staticmethod
    def resample(g,hz):
        """
            g - rowise data, with first collumn time dimension
                all remaining columns will be rescaled to frequency hz
        """
        if hz== None: return g
        if np.isscalar(hz):
            tm=np.linspace(g[0,0],g[-1,0],int(round((g[-1,0]-g[0,0])*hz/1000.0)))
        else: tm=hz
        out=[tm]
        #print g.shape,tm.shape
        for kk in range(1,g.shape[1]):
            out.append(interpRange(g[:,0], g[:,kk],tm))
        return np.array(out).T
    def loadTrajectories(self):
        ''' load trajectories '''
        path = getcwd()
        path = path.rstrip('code')
        order = np.load(path+'/input/vp%03d/ordervp%03db%d.npy'%(self.vp,self.vp,self.block))[self.trial]
        if self.vp>1 and self.vp<10: nfo=(1,1,self.block,order) 
        else: nfo=(self.vp,self.vp,self.block,order) 
        s=path+'/input/vp%03d/vp%03db%dtrial%03d.npy'%nfo
        traj=np.load(s)
        self.oldtraj=traj
        
    def computeAgentDistances(self):
        ''' self.dist computes the distance from gaze to agent
            self.dev is the angle difference between the direction
            of the gaze motion and the direction of the agent motion'''
        self.loadTrajectories()
        #self.traj=traj
        g=self.getGaze()
        traj=self.oldtraj[:self.fs.shape[0],:,:]
        self.dist=np.zeros((g.shape[0],traj.shape[1]))
        self.traj=np.zeros((g.shape[0],traj.shape[1],traj.shape[2]))
        self.dev=np.zeros((g.shape[0]-1,traj.shape[1]))
        for a in range(traj.shape[1]):
            self.traj[:,a,0]=interpRange(self.fs[:,1], traj[:,a,0],g[:,0])
            self.traj[:,a,1]=interpRange(self.fs[:,1], traj[:,a,1],g[:,0])
            dx=np.roll(self.traj[:,a,0],int(0*self.hz),axis=0)-g[:,7]
            dy=np.roll(self.traj[:,a,1],int(0*self.hz),axis=0)-g[:,8]
            self.dist[:,a]= np.sqrt(np.power(dx,2)+np.power(dy,2))
            dg=np.diff(g[:,[7,8]],axis=0);
            #print int(PLAG*self.hz)
            dt=np.roll(np.diff(self.traj[:,a,:],axis=0),int(PLAG*self.hz),axis=0)
            self.dev[:,a]= (np.arctan2(dg[:,0],dg[:,1])
                -np.arctan2(dt[:,0],dt[:,1]))
        self.dev=np.mod(self.dev/np.pi*180.0+180,360)-180

############################################################
# drift correction
    def driftCorrection(self,jump=0):
        """ performs drift correction for each eye and axis
            separately based on the fixation location immediately
            preceding the trial onset
            jump - determines the frame at which the drift correction
                is done relative to trial onset
        """
        s= '\tt= %d, '%(self.trial)
        kk=(np.diff(self.gaze[:,0])>8).nonzero()[0]
        if len(kk)>0: print s, 'LAG >8, ',kk, self.gaze[kk+1,0]-self.gaze[kk,0], self.gaze[kk,3], self.gaze[kk,6]
        dif=np.isnan(self.gaze[:,1]).sum()-np.isnan(self.gaze[:,4]).sum()
        if self.focus==LEFTEYE and dif>0:
            print s,' right eye has more data '%dif
        if self.focus==RIGHTEYE and dif<0:
            print s,' left eye has more data '% dif

        if len(self.calib)>0:
            if self.calib[-1][0][2]<1.5 and self.calib[-1][1][2]<1.5:
                print s,'calibration good ',len(self.calib)
            else: print s, 'calibration BAD',self.calib[-1]
        
        if self.ts==-1: print  s, 'online dcorr failed'
        if isinstance(jump,(int,long)) and jump==-1: 
            print 'skipping drift correction'
            self.dcfix=[0,0]
        elif isinstance(jump,(int,long)):
            # find the latest fixation
            #print 'manual drift correction'
            isFix=self.getFixations(phase=1)
            isFix=np.concatenate([self.getFixations(phase=0),isFix[:50]])
            h= isFix.size-50-jump
            while (not np.all(isFix[h:(h+50)]) and h>=0 ): h-=1
            #print i, d.isFix[0].size-h
            self.dcfix=[self.gaze[h+10,0]-self.t0[0],self.gaze[h+40,0]-self.t0[0]]
            if h>=0:
                for j in [1,4,2,5]:
                    dif=self.gaze[(h+10):(h+40),j].mean()
                    self.gaze[:,j]-=dif
            else: print s,'DRIFT CORRECTION FAILED', np.sum(isFix[-50:])
        else: 
                self.gaze[:,[1,2]]-=np.array(jump,ndmin=2)
                self.gaze[:,[4,5]]-=np.array(jump,ndmin=2)
        # recompute events
        self.gaze=self.gaze[:,:7]
        self.extractBasicEvents()
        self.extractPursuitEvents()
##        # reclassify some blinks as saccades
##        g=self.getGaze()
##        bsac=[]
##        for i in range(len(self.bev)): 
##            s=self.bev[i][0]; e=self.bev[i][1]
##            if norm(g[e,1:]-g[s-1,1:])>BLINK2SAC:
##                bsac.append([e-int(BLKMINDUR*self.hz), e])
##            if e-s>1000: print 't= %d, LONG SEQ OF MISSING DATA'% self.trial
        # merge events into single stream
        self.events=[]
        ind=[0,0,0,0,0]
        #data[t].extractAgentDistances()
        d=[self.fev,self.opev,self.cpev,self.sev,self.bev]#,bsac]
        for f in range(int(30*self.hz)):
            for k in range(len(d)):
                if ind[k]< len(d[k]) and d[k][ind[k]][0]==f:
                    ev=copy(d[k][ind[k]]);
                    ev.append(k);ind[k]+=1;
                    self.events.append(ev)

#####################################################
# identification of basic events
    def extractBasicEvents(self):
        """ extract and filter gaze location,
            compute velocity, acceleration
            and find fixations and saccades
        """
        dat=self.gaze
        # add two columns with binocular gaze point
        if self.focus==BINOCULAR:
            print self.gaze.shape
            gazep=np.array([dat[:,[1,4]].mean(1),dat[:,[2,5]].mean(1)]).T
            temp=dat[np.isnan(dat[:,1]),:]
            if temp.size>0: gazep[np.isnan(dat[:,1]),:]=temp[:,[4,5]]
            temp=dat[np.isnan(dat[:,4]),:]
            if temp.size>0: gazep[np.isnan(dat[:,4]),:]=temp[:,[1,2]]
        elif self.focus==LEFTEYE:
            gazep=dat[:,[1,2]]                        
            #gazep[np.isnan(gazep[:,0]),:]=dat[k][np.isnan(gazep[:,0]),[4,5]]
        elif self.focus==RIGHTEYE:
            gazep=dat[:,[4,5]]
            #gazep[np.isnan(gazep[:,0]),:]=dat[k][np.isnan(gazep[:,0]),[1,2]]       
        self.gaze=np.concatenate([dat,gazep],1)
        # discard parts with large discrepancy between two eyes
        dist=np.sqrt(np.power(np.diff(self.gaze[:,[1,4]],1),2)+
            np.power(np.diff(self.gaze[:,[2,5]],1),2))
        if (dist>EYEDEV).sum()>0:
            for hhh in [1,2,4,5,7,8]:
                #print (dist>3).shape
                self.gaze[(dist>EYEDEV).squeeze(),hhh]=np.nan*np.ones((dist>EYEDEV).sum())
        #if self.trial==29: print self.gaze.shape,np.isnan(self.gaze[:,7]).nonzero()
        for i in [7,8]:
            self.gaze[:,i]=filterGaussian(self.gaze[:,i],self.hz)
            self.gaze[:,i] =interpolateBlinks(self.gaze[:,0],self.gaze[:,i],self.hz) 
        self.vel=computeVelocity(self.gaze,self.hz)
        self.acc=computeAcceleration(self.gaze,self.hz)
        self.isFix,fev=computeFixations(self.gaze,self.vel,self.acc,self.hz)
        self.isSac,sev=computeSaccades(self.gaze,self.vel,self.acc,self.hz)
        self.isBlink,bev=computeBlinks(self.gaze,self.hz)
        self.bev=ETData.helpf(bev,[self.ts,self.te])
        self.fev=ETData.helpf(fev,[self.ts,self.te])
        self.sev=ETData.helpf(sev,[self.ts,self.te])
        # missing trajectories for the following subjects
        if self.vp>=100 and self.vp<140:
            self.traj=np.zeros((0,0,2))
        else:
            self.computeAgentDistances()
            for i in range(len(self.fev)):
                s=self.fev[i][0];e=self.fev[i][1]-1
                a=selectAgentFIX(self.dist[s:e,:],self.hz)
                self.fev[i].extend(a)
        self.opev=[];self.cpev=[]
    def extractPursuitEvents(self):
        ''' extracts smooth eye movements'''
        opur,opev=computeOLpursuit(self.gaze,self.vel,self.acc,self.hz)
        cpur,cpev=computeCLpursuit(self.gaze,self.vel,self.acc,self.hz)
        self.opev=ETData.helpf(opev,[self.ts,self.te])
        self.cpev=ETData.helpf(cpev,[self.ts,self.te])
        self.opur=opur[self.ts:self.te];
        self.cpur=cpur[self.ts:self.te]
        i=0
        while i<len(self.cpev):
            s=self.cpev[i][0];e=self.cpev[i][1]-1
            a=selectAgentCL(self.dist[s:e,:],self.dev[s:e,:],self.hz)
            if len(a)>0: self.cpev[i].extend(a); i+=1
            else:self.cpev.pop(i);self.cpur[s:(e+1)]=False

        b= np.logical_and(np.logical_and(self.opur,~self.cpur),~self.isFix[self.ts:self.te])
        self.opur,self.opev= computeState(b,[OLPURMD*self.hz,np.inf])
        for i in range(len(self.opev)):
            s=self.opev[i][0];e=self.opev[i][1]-1
            a=selectAgentOL(self.dist[s:e,:])
            self.opev[i].extend(a)
#####################################################
# identification of complex events
    def extractSearch(self):
        ''' identifies exploration saccades based on catch-up saccades
            in self.track
            puts them into self.search which is a list similart to self.track'''
        from copy import copy
        for tr in self.track: tr.extend([[],[]])
        self.search=[];tracked=False
        for k in range(0,len(self.events)-1):# go through all basic events
            if (self.events[k+1][-1] in [FIX,OLPUR,CLPUR] and
                self.events[k][-1] in [SAC]):
                # found saccade now check whether its catch-up sac
                tracked=False
                for tr in self.track: 
                    if (tr[0]<=self.events[k][0] and tr[1]>=self.events[k][1] or 
                        tr[0]<=self.events[k+1][0] and tr[1]>=self.events[k+1][1]):
                        tracked=True
                        temp=copy(self.events[k][:-1])
                        temp.append(self.events[k+1][-1])
                        if self.events[k+1][-1] in [OLPUR,CLPUR]:
                            temp.append(self.events[k+1][:-1])
                        tr[4].append(temp)
                if not tracked:
                    temp=copy(self.events[k][:-1])
                    temp.extend([self.events[k+1][-1]])
                    self.search.append(temp)

                
    def extractComplexEvents(self):
        """ identifies complex events: pursuit and exploration"""
        # build smooth eye movements episodes (SEME)
        hev=[]
        A=[]; lastA=[]
        info=[-1,-1,False,[]]; lastEtype=-1
        trackon=False
        acounter=np.zeros(self.traj.shape[1])
        for ev in self.events:
            if not ev[-1] in [FIX,OLPUR,CLPUR]: 
                if len(info[-1]): info.append([]);
                continue
            #if not ev[-1] in [FIX,OLPUR,CLPUR] or (ev[1]-ev[0])/self.hz<0.07:    continue
            lastA=copy(A) ;
            A= ev[2:-1]
            if ( ev[-1] in [FIX,OLPUR,CLPUR] and not trackon and len(set(lastA) & set(A)) # standard case
                or ev[-1]==CLPUR and not len(set(lastA) & set(A)) and (ev[1]-ev[0])/self.hz>0.03 # no search but long pursuit
                ): # start SEME
                if len(info):
                    if len(info[-1])==0: info.pop(-1)
                    hev.append(info);
                info=[ev[0],ev[1],True,[]]
                trackon=True
                #acounter=np.zeros(self.traj.shape[1])
            elif (len(set(lastA) & set(A)) or (len(A) and np.any(acounter[A]>len(info[3])/2))) and trackon: # continue tracking
                info[1]=ev[1]
            else: # terminate SEME
                if len(info):
                    if len(info[-1])==0: info.pop(-1)
                    hev.append(info);
                info=[ev[0],ev[1],False,[]]
                trackon=False
            info[-1].append(ev);
            #if ev[-1]==CLPUR: acounter[A]+=1 
            lastEtype=ev[-1]
        if len(info)>0: hev.append(info)
        self.hev=hev
        # identify exploration and pursuit, look 
        #for first block isolated by saccades wheter it contians SEME
        # also look at the agents being tracked
        track=[];
        for h in hev:
            s=-1;e=-1
            for i in range(3,len(h)):
                for ev in h[i]: 
                    if ev[-1]==CLPUR: 
                        if s==-1: s=h[i][0][0];
                        e=h[i][-1][1]
            if h[2] and s!=-1: track.append([s,h[1]])
            #else: track.append([h[0],h[1],h,False])
        # remove events that are too short, this can only happen
        # at the start or the end, when an event got split by the trial window
        rem=[]
        for i in range(len(track)):
            if track[i][1]-track[i][0]<8: 
                print 'WARNING!: extractComplexEvents: s>=e, removing event'
                rem.append(i)
        for ri in rem[::-1]: track.pop(ri)
        self.track=track
        # identify tracked agents
        self.computeTrackedAgents()
        # and merge consecutive pursuit events with similar agent sets
        i=1
        while i < len(self.track):
            if len(set(self.track[i][2]) & set(self.track[i-1][2])):
                # count saccades between the two tracking events
                nrsac=0
                for sev in self.sev:
                    if self.track[i-1][1]<=sev[0] and self.track[i][0]>=sev[1]:
                        nrsac+=1
                if nrsac>2: i+=1
                else: # merge 
                    self.track[i-1]= [self.track[i-1][0], self.track[i][1]]
                    #self.track[i-1].append(set(self.track[i][2]) | set(self.track[i-1][2]))
                    ags,fs=selectAgentTRACKING(self.track[i-1][0],self.track[i-1][1],self.events)
                    self.track[i-1].extend([ags,fs])
                    self.track.pop(i)
            else: i+=1
        self.extractSearch()
    def computeTrackedAgents(self):
        ''' identify agents that are tracked during pursuit'''
        for tr in self.track:
            if len(tr)==2:
                ags,fs=selectAgentTRACKING(tr[0],tr[1],self.events)
                tr.extend([ags,fs])
#########################################################
# load/save basic and complex events
    def importComplexEvents(self,coderid=1):
        ''' imports complex events from the coding file
            coderid - coder id determines which file to use
                0 - automatic
                1 - human coder 1
                2 - human coder 2
                4 - merge of 1 and 2 used in the analysis
        '''
        from ReplayData import Coder
        try:
            dat=Coder.loadSelection(self.vp,self.block,self.trial,coder=coderid)
            self.track=[]
            for tr in dat:
                fs=[]
                for k in tr[-2]: fs.append([k[2],k[5]])
                self.track.append([tr[2],tr[5],tr[-3],fs])
            self.extractSearch()
        except IOError:
            try: self.track # dont overwrite existing track
            except: self.track=None;self.search=None # avoid unknown attribute error
            print 'import of complex events failed'
    def exportEvents(self):
        ''' exports events to text file'''
        out=[]
        for ev in self.events:
            if ev[-1]!=SAC:
                out.append([ev[0]*2,ev[1]*2,ev[-1]])
        np.savetxt('vp%02db%02dt%02d.evt'%(self.vp,self.block,self.trial),np.array(out).T,fmt='%d')
    def exportTracking(self):
        ''' saves pusruit events to a text file'''
        out=[]
        ei=0
        for tr in self.track:
            for k in tr[2][5]:
                out.append([int(k[0]*2),int(k[1]*2),k[-1],ei,
                            int(tr[-1])])   
            ei+=1
        np.savetxt('vp%02db%02dt%02d.trc'%(self.vp,self.block,self.trial),np.array(out).T,fmt='%d')
########################################################
# plotting routinates for debugging
    def plotEvents(self):
        ''' plots basic events '''
        #plt.figure()
        ax=plt.gca()
        clrs=['r','g','b','k']
        for f in self.events:
            r=mpl.patches.Rectangle((f[0],self.trial-0.1),
                    f[1]-f[0],0.8,color=clrs[f[-1]])
            ax.add_patch(r)
        plt.xlim([0,10000])
        plt.ylim([-1,41])
        plt.show()  
    def plotTracking(self):
        ''' plots pursuit '''
        #plt.figure()
        ax=plt.gca()
        for f in self.track:
            if f[-1]:r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.4,color='k')
            else: r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.8,color='r')
            ax.add_patch(r)
        plt.xlim([0,15000])
        plt.ylim([-1,41])
        plt.show()
    def plotAgents(self):
        ''' plot which agents were tracked'''
        plt.figure()
        ax=plt.gca()
        for f in self.events:
            for i in range(2,len(f)-1):
                a=f[i]
                #print a
                r=mpl.patches.Rectangle((f[0],a),f[1]-f[0],0.8,color='k')
                ax.add_patch(r)
        plt.xlim([0,self.gaze[1].shape[0]])
        plt.ylim([-1,15])
        plt.show()
    def plotMissingData(self):
        '''plots missing data '''
        g=self.getGaze()
        #plt.cla()
        ax=plt.gca()
        clrs=['g','r'];xy=[0.8,0.4];idd=[7]
        for e in range(len(idd)):
            m=np.isnan(g[:,idd[e]])
            on = np.bitwise_and(m,np.bitwise_not(np.roll(m,1))).nonzero()[0].tolist()
            off=np.bitwise_and(np.roll(m,1),np.bitwise_not(m)).nonzero()[0].tolist()
            if len(on)==0 and len(off)==0: continue
            if on[-1]>off[-1]: off.append(m.shape[0]-1)
            if on[0]> off[0]: on.insert(0,0)
            if len(on)!=len(off): print 'invalid onoff';raise TypeError
            for f in range(len(on)):
                fs=g[on[f],0];fe=g[off[f],0];dur=fe-fs
                if dur>10:
                    r=mpl.patches.Rectangle((fs,self.trial-0.1),fe-fs,xy[e],color=clrs[e])
                    ax.add_patch(r)
        plt.xlim([0,30000])
        plt.ylim([-1,41])
    def plotMsgs(self,st=0):
        ''' plot messages from the baby data'''
        ax=plt.gca()
        row=self.vp
        plt.plot([st,st],[row,row+1],'r')
        
        for ev in self.bev:
            s=ev[0];e=ev[1]
            r=mpl.patches.Rectangle((st+s,row+0.25),e-s,0.5,color='k')
            ax.add_patch(r)
        for ev in self.fev:
            s=ev[0];e=ev[1]
            r=mpl.patches.Rectangle((st+s,row+0.25),e-s,0.5,color='g')
            ax.add_patch(r)
        for ev in self.sev:
            e=ev[1]-1
            plt.plot([st+e,st+e],[row+0.2,row+0.8],'b')
            #r=mpl.patches.Rectangle((,row+0.25),
            #        ev[1]-ev[0],0.25,color='b')
            #ax.add_patch(r)
        for msg in self.msgs:
            temp=msg[2].split('th ')
            if len(temp)==2:
                plt.plot(st+msg[3]-5,row+int(temp[0])/12.,'r.')
            elif len(temp)>2:
                print 'plotMsgs:',msg[2],temp
                raise         
        #plt.ylim([69,86])
        plt.xlim([0, 6*60*60])
        plt.grid(b=False)
        #print self.gaze.shape[0], self.bev[-1][1],self.gaze[-1,0],self.gaze[self.bev[-1][1],0]
        #print self.gaze.shape, self.fs.shape,self.fs[-1], self.revfs.shape
        return st+self.te-self.ts


##################################################################
# getter functions
    def selectPhase(self,data,phase,hz):
        """ used by the getter functions below to select a portion 
            (phase) of the data based on the time.
            data - data from which the portion is selected
            phase - determines which portion of the data is selected,
                following options are available:
                -1 - select all portions (used with opur, cpur)
                0 - drift correction
                1 - presentation (from presentation onset to presentation end)
                2 - target selection (if subject detected chase)
                3 - [-200,200] ms around trial start (used for
                    checking drift correction)
            hz - data rate of the output, will be interpolated
                if different from the recording rate
        """
        dat=np.array([self.gaze[:,0]-self.t0[0],data]).T
        if phase ==-1:  out=np.array([self.gaze[self.ts:self.te,0]-self.t0[0],data]).T
        elif phase==0: out= dat[:self.ts]
        elif phase==1:
            if self.ts>=0: out= dat[self.ts:self.te]
            else: out= np.zeros((0,9))
        elif phase==2:
            if self.te>=0: out= dat[self.te:]
            else: out= np.zeros((0,9))
        elif phase==3:
            if self.ts>=0: out= dat[self.ts-50:self.ts+50]
            else: out= np.zeros((0,9))
        else: print 'Invalid Phase'
        out=ETData.resample(out,hz)
        return out[:,1]
    def getTraj(self,hz=None):
        '''returns agent trajectories
            3d ndarray (sample size x nr agents x 2)  '''
        t=self.fs[:,1]
        try: out=self.oldtraj[:t.size,:,:]
        except AttributeError: self.loadTrajectories(); out=self.oldtraj[:t.size,:,:]
        res=[]
        res.append(ETData.resample(np.concatenate((np.array(t,ndmin=2).T,out[:,:,0]),axis=1),hz))
        res.append(ETData.resample(np.concatenate((np.array(t,ndmin=2).T,out[:,:,1]),axis=1),hz))
        res= np.array(res)
        res=np.rollaxis(res,0,3)
        return res[:,1:,:]
    def getGaze(self,phase=1,hz=None):
        ''' returns gaze information
            2d ndarray (sample size x 9), columns give
            0-time,1-left eye x, 2 - left eye y,
            3 - left eye pupil size, 4 - right eye x,
            5 - right eye y, 6 - right eye pupil size
            7 - gaze point left, 8 - gaze point right
            phase - see ETData.selectPhase() for details
            hz - data rate of the output, will be interpolated
                if different from the recording rate'''
        if phase==-1: out=self.gaze
        elif phase==0: out= self.gaze[:self.ts,:]
        elif phase==1:
            if self.ts>=0:
                out= np.copy(self.gaze[self.ts:self.te,:])
                out[:,0]-= (self.t0[0])
            else: return np.zeros((0,9))
        elif phase==2:
            if self.te>=0:
                out= np.copy(self.gaze[self.te:,:])
                out[:,0]-= (self.t0[1])
            else: return np.zeros((0,9))
        elif phase==3:
            if self.ts>=0:
                out= np.copy(self.gaze[self.ts-100:self.ts+100,:])
                out[:,0]-= (self.t0[0])
            else: out= np.zeros((0,9))
        else: print 'Phase not supported'
        out=ETData.resample(out,hz)
        return out
    def getVelocity(self,phase= 1,hz=None):
        return self.selectPhase(self.vel,phase,hz)
    def getAcceleration(self,phase=1,hz=None):
        return self.selectPhase(self.acc,phase,hz)
    def getFixations(self,phase=1,hz=None):
        return self.selectPhase(self.isFix,phase,hz)
    def getSaccades(self,phase=1,hz=None):
        return self.selectPhase(self.isSac,phase,hz)
    def getCLP(self,phase=1,hz=None):#fast smooth eye movement
        return self.selectPhase(self.cpur,-1,hz)
    def getOLP(self,phase=1,hz=None):# slow smooth eye movement
        return self.selectPhase(self.opur,-1,hz)
    def getHEV(self,phase,hz=None):
        out=np.zeros(hz.size)
        for tr in self.hev:
            if tr[2]: out[t2f(tr[0]/float(self.hz)*1000,hz):t2f(tr[1]/float(self.hz)*1000,hz) ]=1
        return out
    def getTracking(self,phase,hz=None):# pursuit events
        out=np.zeros(hz.size)
        for tr in self.track:
            out[t2f(tr[0]/float(self.hz)*1000,hz):t2f(tr[1]/float(self.hz)*1000,hz) ]=1
        return out
    def getAgent(self,t):
        ''' returns tuple with ids of the agents that are
            focused at time t by any of the basic events'''
        g=self.gaze[self.ts:self.te]
        for p in self.opev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        for p in self.cpev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        for p in self.fev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        return []
    def getTrackedAgent(self,t):
        ''' returns tuple with ids of the agents that are
            focused at time t during a pursuit'''
        g=self.gaze[self.ts:self.te,0]
        gmax=g.size-1
        for tr in self.track:
            if t>g[tr[0]]-self.t0[0] and t<g[min(gmax,tr[1])]-self.t0[0]: 
                return tr[2]
        return []
#########################################################
# functions for manual drift correction
def plotDC(vp,block,trial):
    ''' plot gaze during drift correction'''
    from Preprocess import readEyelink
    plt.interactive(False)
#    vp=1
#    from readETData import readEyelink
#    for b in range(4,23):
#        print 'block ', b
#        data=readEyelink(vp,b)
#        for i in range(0,len(data)):
    b=block;i=trial
    data=readEyelink(vp,b)
    d=data[i]
    gg=d.getGaze(phase=3)
    plt.plot(gg[:,0],gg[:,1],'g--')
    plt.plot(gg[:,0],gg[:,2],'r--')
    plt.plot(gg[:,0],gg[:,4],'b--')
    plt.plot(gg[:,0],gg[:,5],'k--')
    d.extractBasicEvents()
    d.driftCorrection(jump=manualDC(vp,b,i))
    gg=d.getGaze(phase=3)
    plt.plot(gg[:,0],gg[:,1],'g')
    plt.plot(gg[:,0],gg[:,2],'r')
    plt.plot(gg[:,0],gg[:,4],'b')
    plt.plot(gg[:,0],gg[:,5],'k')
    plt.plot([gg[0,0],gg[-1,0]],[0,0],'k')
    plt.plot(d.dcfix,[-0.45,-0.45],'k',lw=2)
    plt.grid()
    plt.ylim([-0.5,0.5])
    plt.legend(['left x','left y','right x','right y'])
    plt.savefig(PATH+'dc'+os.path.sep+'vp%03db%02dtr%02d'%(vp,b,i))
    plt.cla()

def plotMD():
    '''plots the effect of drift correction '''
    vp=1
    from Preprocess import readEyelink
    for b in range(1,22):
        plt.cla()
        print 'block ', b
        data=readEyelink(vp,b)
        for d in data:
            d.driftCorrection(jump=manualDC(vp,b,i))
            d.plotMissingData()
        plt.show()
        plt.savefig('fb%02d'%(d.block))
        
def manualDC(vp,b,t):
    """ the automatic drift correction failed on some trials
        manualDC returns time in ms relative to trial onset
        the fixation at this time point is used to perform
        the drift correction,
        for out= -1 the online DC during experiment will be used
    """
    out=0
    if vp==1:
        if b==3 and t==25: out=-10
        elif b==5 and t==10: out=50
        elif b==6 and t==19: out=20
        elif b==6 and t==24: out=20
        elif b==8 and t==14: out=50
        elif b==12 and t==37: out=0
    elif vp==2:
        if b==7 or b==8 or b==22 or b==21 or b==10 or b==11 or (b>12 and b<20) or b<5: out=-1
        if b==5 and t==1: out=-50
        if b==5 and t==14: out=100
        if b==5 and t==17: out=-1
        if b==5 and t>18 and t<26: out=-1
        if b==5 and t==26: out=50
        if b==5 and t>=29: out=-1
        if b==5 and t==32: out=50
        if b==6 and t==2: out=-1
        if b==6 and t==7: out=-1
        if b==6 and t==22: out=-1
        if b==6 and t==23: out=-1
        if b==6 and t==25: out=-1
        if b==6 and t>=27: out=-1
        if b==6 and t==33: out=0
        if b==9 and t==1: out=-20
        if b==9 and t==5: out=100
        if b==9 and t==13: out=100
        if b==9 and t==14: out=-50
        if b==9 and t==16: out=50
        if b==9 and t==17: out=100
        if b==9 and t==22: out=100
        if b==9 and t==23: out=100
        if b==9 and t==26: out=100
        if b==9 and t==38: out=-50
        if b==22 and t==23: out=-50
        if b==20 and t==14: out=-1
        if b==20 and t==17: out=100
        if b==20 and t==24: out=100
        if b==20 and t==37: out=-1
        if b==19 and t==10: out=0
        if b==19 and t==32: out=0
        if b==18 and t==1: out=0
        if b==18 and t==3: out=100
        if b==18 and t==21: out=100
        if b==18 and t==39: out=0
        if b==11 and t==15: out=0
        if b==12 and t==3: out=-50
        if b==12 and t==13: out=100
        if b==12 and t==22: out=-50
        if b==12 and t==24: out=-50
        if b==12 and t==29: out=100
        if b==12 and t==33: out=100
        if b==12 and t==37: out=100
        if b==12 and t==38: out=-1
        if b==12 and t==1: out=-1
        if b==12 and t==11: out=-1
        if b==12 and t==19: out=-1
        if b==12 and t==27: out=-1
        if b==13 and t==11: out=0
        if b==14 and t==32: out=0
        if b==4 and t==31: out=0
        if b==3 and t==5: out=0
    elif vp==3:
        out=-1
        if b==1 and t==23: out=0
        if b==4 and t==22: out=0
        if b==7 and t==36: out=0
        if b==9 and t==23: out=0
        if b==11 and t==36: out=0
        if b==14 and t==38: out=-50
        if b==20 and t==27: out=0
        if b==21 and t==27: out=-50
        
    elif vp==4:
        out=-1
        if b==1 and t==3: out=0
        if b==1 and t==5: out=0
        if b==1 and t==14: out=0
        if b==1 and t==27: out=0
        if b==1 and t==37: out=0
        if b==3 and t==0: out=0
        if b==5 and t==9: out=0
        if b==11 and t==2: out=0
        if b==15 and t==25: out=0
        if b==15 and t==32: out=0
        if b==16 and t==13: out=0
        if b==17 and t==22: out=0
        if b==18 and t==20: out=0
        if b==19 and t==19: out=0
        if b==20 and t==29: out=0
        if b==5 and t==37: out=0
    print out
    return out
            
if __name__ == '__main__':
    # following routine was used to check correctness of drift correction
    plotDC(4,7,31)
   
  
 


