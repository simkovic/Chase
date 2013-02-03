import numpy as np
import pylab as plt
import matplotlib as mpl
from Settings import *
from Constants import *
from psychopy import core,event,visual
from scipy import signal
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.interpolate import interp1d
from copy import copy
import time,os


# TODO: readTobii() for reading drift correction data

plt.ion()

def norm(x,axis=0):
    x=np.power(x,2)
    x=x.sum(axis)
    return np.sqrt(x)

def interpRange(xold,yold,xnew):
    ynew=np.float32(xnew.copy())
    f=interp1d(xold,yold)
    for i in range(xnew.size):
        ynew[i] =f(xnew[i])
    return ynew


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
    ''' avoids invoking monitors'''
    cm= myPix2cm(pix,cent,width)
    #return np.arctan((pix/37.795275591)/float(dist))*180/np.pi
    return myCm2deg(cm,dist)
def myDeg2pix(deg,dist,cent,width):
    cm = myDeg2cm(deg,dist)
    return myCm2pix(cm,cent,width)



##def filterButterworth(self,x,cutoff=50):
##        
##        #normPass = 2*np.pi*cutoff/self.hz*2
##        normPass = cutoff / (self.hz/2.0)
##        normStop = 1.5*normPass
##        print normPass, normStop
##        (N, Wn) = signal.buttord(wp=normPass, ws=normStop,
##                    gpass=2, gstop=30, analog=0)
##        (b, a) = signal.butter(N, Wn)
##        print b.shape
##        #b *= 1e3
##        # return signal.lfilter(b, a, x[::-1])[::-1]
##        return signal.lfilter(b, a, x)
##
##    def filterCausal(self,x,theta=0.6):
##        y=np.zeros(x.shape)
##        y[0]=x[0]
##        for i in range(1,x.size):
##            if np.isnan(y[i-1]): y[i]=x[i]
##            else: y[i]=(theta)*y[i-1] +(1- theta)*x[i]
##        return y

# computation functions
###tobii baby settings
##SACVTH=15 # velocity threshold deg/sec
##SACATH=1500 # acceleration threshold deg/sec^2
##FIXVTH=18
##FIXATH=1500
##FILTERCUTOFF = 25 # hz

# eyelink settings
LSACVTH=50
LSACATH=8000
LSACMINDUR=0.06

SACVTH=21
SACATH=4000
SACMINDUR=0.02

FIXVTH=8
FIXATH=800
FIXMINDUR=0.1 #second

OLPURVTH= SACVTH
OLPURATH= SACATH
OLPURMD=0.06

CLPURVTHU=SACVTH
CLPURVTHL=FIXVTH
CLPURATHU=np.inf#FIXATH
CLPURATHL=np.inf
CLPURMD=0.14

FILTERCUTOFF = 50 #hz, cutoff of the gaussian filter
BLKMINDUR=0.05
NBLMINDUR=0.01
OLFOCUSRADIUS=3.5 # focus radius for agents
CLFOCUSRADIUS=4
MAXPHI=35#10 #
PLAG = 0 # lag between agent movement and pursuit movement
#used for agent tracking identification
EYEDEV=4
BLINK2SAC =2# deg, mininal blink before-after distance to create saccade
INTERPMD=0.250 # max duration for which blink interpolation is performed
def selectAgentCL(dist,dev):
    #a=(dist.max(0)<3).nonzero()[0]
    #b=norm(np.median(dev,axis=0),axis=1)<MAXPHI
    #b=np.mod(dev/np.pi*180.0+180,360)-180
    #print (np.abs(dev)<MAXPHI).mean(axis=0)
    b=(np.abs(dev)<MAXPHI).mean(axis=0)>0.5
    a=np.logical_and(dist.mean(0)<CLFOCUSRADIUS,b).nonzero()[0]
    return a
def selectAgentOL(dist):
    a=(np.mean(dist,axis=0)<OLFOCUSRADIUS).nonzero()[0]
    return a


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
    if filt: vel=filterGaussian(vel,hz)
    return vel
def computeAcceleration(tser,hz ,filt=False):
    vel = computeVelocity(tser,hz,filt)
    acc=np.concatenate((np.diff(vel*hz),[0]),0)
    if filt: acc=filterGaussian(acc,hz)
    return acc

def computeState(isFix,md):
    fixations=[]
    if isFix.sum()==0: return np.int32(isFix),[]
    fixon = np.bitwise_and(isFix,
        np.bitwise_not(np.roll(isFix,1))).nonzero()[0].tolist()
    fixoff=np.bitwise_and(np.roll(isFix,1),
        np.bitwise_not(isFix)).nonzero()[0].tolist()
    if fixon[-1]>fixoff[-1]:
        fixoff.append(isFix.shape[0]-1)
    if fixon[0]>fixoff[0]:
        fixon.insert(0,0)
    if len(fixon)!=len(fixoff):
        print 'invalid fixonoff'
        raise TypeError
    for f in range(len(fixon)):
        fs=fixon[f]
        fe=(fixoff[f]+1)
        dur=fe-fs
        if  dur<md[0] or dur>md[1]:
            isFix[fs:fe]=False
        else: fixations.append([fs,fe-1])
    #fixations=np.array(fixations)
    return isFix,fixations

def interpolateBlinks(t,d,hz):
    ''' interpolate short missing intervals 
    '''
    isblink= np.isnan(d)
    if isblink.sum()==0: return d
    blinkon = np.bitwise_and(isblink,np.bitwise_not(
        np.roll(isblink,1))).nonzero()[0].tolist()
    blinkoff=np.bitwise_and(np.roll(isblink,1),
        np.bitwise_not(isblink)).nonzero()[0].tolist()
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

    
def computeFixations(tser,vel,acc,hz):
    isFix=np.logical_and(vel<FIXVTH,np.abs(acc)<FIXATH)
    return computeState(isFix,[FIXMINDUR*hz,np.inf])

##def computePursuit(tser,vel,acc,hz):
##    isFix=np.logical_and(vel<FIXVTH,np.abs(acc)<FIXATH)
##    isFix,b=computeState(isFix,[FIXMINDUR*hz,np.inf])
##    isPur=np.logical_and(np.logical_and(vel<PURVTHU,np.abs(acc)<PURATHU),
##        np.logical_or(vel>PURVTHL,np.abs(acc)>PURATHL))
##    isPur=np.logical_and(isPur,~isFix)
##    return computeState(isPur,[PURMINDUR*hz,np.inf])
def computeOLpursuit(tser,vel,acc,hz):
    isFix=np.logical_and(vel<OLPURVTH,np.abs(acc)<OLPURATH)
    return computeState(isFix,[OLPURMD*hz,np.inf])
def computeCLpursuit(tser,vel,acc,hz):
    temp=np.logical_and(np.logical_and(vel<CLPURVTHU,np.abs(acc)<CLPURATHU),
        np.logical_or(vel>CLPURVTHL,np.abs(acc)>CLPURATHL))
    return computeState(temp,[CLPURMD*hz,np.inf])

def computeSaccades(tser,vel,acc,hz):
    isFix=np.logical_or(vel>SACVTH,np.abs(acc)>SACATH)
    return computeState(isFix,[SACMINDUR*hz,np.inf])

def computeLongSaccades(tser,vel,acc,hz):
    isFix=np.logical_or(vel>LSACVTH,np.abs(acc)>LSACATH)
    return computeState(isFix,[LSACMINDUR*hz,np.inf])

def computeBlinks(tser,hz):
    isFix=np.isnan(tser[:,1])
    return computeState(isFix,[BLKMINDUR*hz*0,np.inf])
        
class ETBlockData():
    def __init__(self,etdata):
        self.etdata=etdata
        self.vp=self.etdata[0].vp
        self.block=self.etdata[0].block
        
    def getTrial(self,trialid):
        return self.etdata[trialid]
        
        
    def loadBehavioralData(self):
        path = getcwd()
        path = path.rstrip('code')
        dat=np.loadtxt(path+'behavioralOutput/vp%03d.res'%self.vp)
        self.behdata=dat[dat[:,1]==self.block,:]

        print 'Checking consistence with behavioral data'
        dev=np.zeros(40)

        for t in range(40):
            self.etdata[t].behdata=self.behdata[t,:]
            a=self.behdata[t,6]*1000
            b=self.etdata[t].gaze[1][-1,0]
            dev[t]=abs(a-b)
            if dev[t]>5: print '\tt=%d deviation %.3f'%(t,dev[t])
        if np.all(dev<5): print '\tdetection times ok (abs error <5 msec)' 
            
        
##    def driftCorrection(self):
##        
##        i=0
##        remove=[]
##        print 'subject %d, block %d, SUMMARY'%(self.vp,self.block)
##        for d in self.etdata:
##            pass
##        # remove unsuccesful online dcorrs
##        remove.reverse()
##        for k in remove: self.etdata.pop(k)
##        #for d in self.etdata: d.extractAgentDistances()
            
class ETTrialData():
    def __init__(self,dat,calib,t0,info,recTime="0:0:0",
                 INTERPBLINKS=False,fcutoff=70,focus=BINOCULAR):
        self.calib=calib
        self.vp=int(info[0])
        self.block=int(info[1])
        self.trial=int(info[2])
        self.hz=float(info[3])
        self.eye=info[4]
        self.t0=[t0[1]-t0[0],t0[2]-t0[0]]
        self.focus=focus # which eye use to indicate focus, if binocular, use gaze average
        if self.t0[0]>0: self.ts=min((dat[:,0]>(self.t0[0])).nonzero()[0])
        else: self.ts=-1
        if self.t0[1]>0: self.te=min((dat[:,0]>(self.t0[1])).nonzero()[0])
        else: self.te=-1
        self.recTime=datetime.strptime(recTime,"%H:%M:%S")
        self.extractFixations(dat)
        
        
    def getTraj(self):
        return self.oldtraj
    def getDist(self,a=0):
        return self.dist[:,a]/20.0
    def getAgent(self,t):
        g=self.gaze[self.ts:self.te]
        for p in self.opev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        for p in self.cpev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        return []
        
    def computeAgentDistances(self):
        path = getcwd()
        path = path.rstrip('/code')
        order = np.load(path+'/input/vp%03d/ordervp%03db%d.npy'%(self.vp,self.vp,self.block))[self.trial]
        s=path+'/input/vp%03d/vp%03db%dtrial%03d.npy'%(self.vp,self.vp,self.block,order) 
        traj=np.load(s)
        self.oldtraj=traj
        #self.traj=traj
        g=self.getGaze()
        tt=np.arange(0,30001,10)
        
        self.dist=np.zeros((g.shape[0],traj.shape[1]))
        self.traj=np.zeros((g.shape[0],traj.shape[1],traj.shape[2]))
        self.dev=np.zeros((g.shape[0]-1,traj.shape[1]))
        for a in range(traj.shape[1]):
            #print tt.shape, traj[:,a,0].shape
            self.traj[:,a,0]=interpRange(tt, traj[:,a,0],g[:,0])
            self.traj[:,a,1]=interpRange(tt, traj[:,a,1],g[:,0])
            dx=np.roll(self.traj[:,a,0],int(0*self.hz),axis=0)-g[:,7]
            dy=np.roll(self.traj[:,a,1],int(0*self.hz),axis=0)-g[:,8]
            self.dist[:,a]= np.sqrt(np.power(dx,2)+np.power(dy,2))
            dg=np.diff(g[:,[7,8]],axis=0);
            #print int(PLAG*self.hz)
            dt=np.roll(np.diff(self.traj[:,a,:],axis=0),int(PLAG*self.hz),axis=0)
            self.dev[:,a]= (np.arctan2(dg[:,0],dg[:,1])
                -np.arctan2(dt[:,0],dt[:,1]))
        self.dev=np.mod(self.dev/np.pi*180.0+180,360)-180

            
    def driftCorrection(self):
        """ performs drift correction for each eye and axis
            separately based on the fixation location immediately
            preceding the trial onset
        """
        s= '\tt= %d, '%(self.trial)
        kk=(np.diff(self.gaze[:,0])>4).nonzero()[0]
        if len(kk)>0: print s, 'LAG >4, ',kk
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
        # find the latest fixation
        isFix=self.getFixations(phase=0)
        h= isFix.size-50
        while (not np.all(isFix[h:(h+50)]) and h>=0 ): h-=1
        #print i, d.isFix[0].size-h
        if h>=0:
            for j in [1,4,2,5]:
                dif=self.gaze[h:(h+50),j].mean()
                self.gaze[:,j]-=dif
            # recompute fixations
            self.extractFixations(self.gaze[:,:7],True)
        else: print s,'DRIFT CORRECTION FAILED', np.sum(d.isFix[0][-50:])
        
    def extractTracking(self):
        """ extracts high-level events - search and tracking"""
##        # reclassify some blinks as saccades
##        g=self.getGaze()
##        bsac=[]
##        for i in range(len(self.bev)): 
##            s=self.bev[i][0]; e=self.bev[i][1]
##            if norm(g[e,1:]-g[s-1,1:])>BLINK2SAC:
##                bsac.append([e-int(BLKMINDUR*self.hz), e])
##            if e-s>1000: print 't= %d, LONG SEQ OF MISSING DATA'% self.trial

        # merge events together
        self.events=[]
        ind=[0,0,0,0]
        #data[t].extractAgentDistances()
        d=[self.opev,self.cpev,self.sev]#,bsac]
        for f in range(int(30*self.hz)):
            for k in range(len(d)):
                if ind[k]< len(d[k]) and d[k][ind[k]][0]==f:
                    ev=copy(d[k][ind[k]]);
                    ev.append(k);ind[k]+=1;
                    self.events.append(ev)
        # build high-level events
        hev=[]
        lastlastA=[]
        lastA=[]
        lastEtype=-1
        info=[]
        trackon=False
        for ev in self.events:
            A= ev[2:-1]
            if ev[-1] in [FIX,PUR]:
                if len(set(lastA) & set(A)) and trackon:# and lastEtype==SAC:
                    info[2]+=int(lastEtype ==SAC);
                    info[1]=ev[1]
                else:
                    if len(info):hev.append(info);
                    info=[ev[0],ev[1],0,0,0,[],self.trial]
                    trackon=False
                #if not (ev[-1]==FIX and lastEtype==PUR):
                lastA=copy(A)  
            #if ev[-1]==SAC: info[2].append(ev)
            if ev[-1]==FIX: info[5].append(ev);info[3]+=1; 
            if ev[-1]==PUR:
                
                info[5].append(ev)
                info[4]+=1;
                trackon=True# (ev[1]-ev[0])>100
                #np.any(self.dist[ev[0]:ev[1],A].mean(0))
                
            lastEtype=ev[-1]
        if len(info)>0: hev.append(info)
        self.hev=hev
        # identify search and tracking
        track=[]
        for h in hev:
            flag=False
            if h[4]>0:
                for f in h[5]:
                    if f[-1]==PUR and f[1]-f[0]>250: flag=True
            if h[2]>0 or flag: track.append([h[0],h[1],h,True])
            else: track.append([h[0],h[1],h,False])
        self.track=track

        
    def plotTracking(self):
        ''' plots events '''
        #plt.figure()
        ax=plt.gca()
        for f in self.track:
            if f[-1]:r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.4,color='k')
            else: r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.8,color='r')
            ax.add_patch(r)
        plt.xlim([0,10000])
        plt.ylim([-1,41])
        plt.show()
    def exportTracking(self):
        out=[]
        ei=0
        for tr in self.track:
            for k in tr[2][5]:
                out.append([int(k[0]*2),int(k[1]*2),k[-1],ei,
                            int(tr[-1])])   
            ei+=1
        np.savetxt('vp%02db%02dt%02d.trc'%(self.vp,self.block,self.trial),np.array(out).T,fmt='%d')
        
            

                        
        
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

    
        
    def plotEvents(self):
        ''' plots events '''
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
        
    def extractFixations(self,dat,PUR=False):
        """ extract and filter gaze location,
            compute velocity, acceleration
            and find fixations
        """
        def helpf(fev,inds):
            out=[]
            for ff in fev:
                if ((ff[0]>inds[0] and ff[0]<inds[1])
                    or (ff[1]>inds[0] and ff[1]<inds[1])):
                    temp=[max(ff[0]-inds[0],0),min(ff[1]-inds[0],inds[1]-inds[0]-1)]+ff[2:]
                    out.append(temp)
            return out 
        # add two columns with gaze point
        if self.focus==BINOCULAR:
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
        #self.fev=helpf(fev,inds[k])
        self.sev=helpf(sev,[self.ts,self.te])
        #self.lev=helpf(lev,inds[k])
        self.bev=helpf(bev,[self.ts,self.te])
                
        # find pursuit states for phase 2
        if PUR:
            self.computeAgentDistances()
            opur,opev=computeOLpursuit(self.gaze,self.vel,self.acc,self.hz)
            cpur,cpev=computeCLpursuit(self.gaze,self.vel,self.acc,self.hz)
            self.opev=helpf(opev,[self.ts,self.te])
            self.cpev=helpf(cpev,[self.ts,self.te])
            self.opur=opur[self.ts:self.te];
            self.cpur=cpur[self.ts:self.te]
            i=0
            while i<len(self.cpev):
                s=self.cpev[i][0];e=self.cpev[i][1]-1
                a=selectAgentCL(self.dist[s:e,:],self.dev[s:e,:])
                if len(a)>0:
                    self.cpev[i].extend(a); i+=1
                else:self.cpev.pop(i);self.cpur[s:(e+1)]=False

            b= np.logical_and(self.opur,~self.cpur)
            self.opur,self.opev= computeState(b,[OLPURMD*self.hz,np.inf])
            for i in range(len(self.opev)):
                s=self.opev[i][0];e=self.opev[i][1]-1
                a=selectAgentOL(self.dist[s:e,:])
                self.opev[i].extend(a)
    
    @staticmethod
    def rescale(g,hz):
        tm=np.arange(g[0,0],g[-1,0],1000/float(hz))
        out=[tm]
        for kk in [7,8]:
            out.append(interpRange(g[:,0], g[:,kk],tm))
        return np.array(out).T
    def selectPhase(self,dat,phase):
        if phase==0: return dat[:self.ts]
        elif phase==1:
            if self.ts>=0: return dat[self.ts:self.te]
            else: return np.zeros((0,9))
        elif phase==2:
            if self.te>=0: return dat[self.te:]
            else: return np.zeros((0,9))
        else: print 'Invalid Phase'
    def getGaze(self,phase=1,hz=None):
        if phase==0: out= self.gaze[:self.ts,:]
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
        else: print 'Phase not supported'
        return out

    def getVelocity(self,phase= 1,t=None):
        return self.selectPhase(self.vel,phase)
    def getAcceleration(self,phase=1,t=None):
        return self.selectPhase(self.acc,phase)
    def getFixations(self,phase=1,t=None):
        return self.selectPhase(self.isFix,phase)
    def getSaccades(self,phase=1,t=None):
        return self.selectPhase(self.isSac,phase)
    def getCLP(self,phase=1,t=None):
        return self.cpur
    def getOLP(self,phase=1,t=None):
        return self.opur
    def getTracking(self,phase):
        out=np.zeros(self.te-self.ts)
        for tr in self.track:
            if tr[-1]: out[tr[0]:tr[1]]=1
        return out
    def getSearch(self,phase):
        out=np.zeros(self.te-self.ts)
        for tr in self.track:
            if not tr[-1]: out[tr[0]:tr[1]]=1
        return out
    
def discardInvalidTrials(data):
    bb=range(len(data))
    bb.reverse()
    for i in bb:
        if data[i].ts<0: data.pop(i)
    return data

def readEdf(vp,block):
    def reformat(trial,cent,tstart,distance):
        if len(trial)==0: return np.zeros((0,7))
        trial=np.array(trial)
        if type(trial) is type( () ): print 'error in readEdf'
        trial[:,0]-=tstart
        trial[:,1]=myPix2deg(trial[:,1],distance,cent[0],40)
        trial[:,2]=-myPix2deg(trial[:,2],distance,cent[1],32)
        if trial.shape[1]==7:
            trial[:,4]=myPix2deg(trial[:,4],distance,cent[0],40)
            trial[:,5]=-myPix2deg(trial[:,5],distance,cent[1],32)
        return trial
    cent=(0,0)
    path = getcwd()
    path = path.rstrip('code')
    f=open(path+'eyelinkOutput/VP%03dB%d.asc'%(vp,block),'r')
    LBLINK=False; RBLINK=False
    ende=False
    try:
        line=f.readline()
        data=[]
        PHASE= -1 # 0-DCORR, 1-TRIAL, 2-DECISION, 4-CALIB
        etdat=[]
        t0=[0,0,0]
        calib=[]
        i=0
        t=0
        size=7
        while True:   
            words=f.readline().split()
            i+=1            
            #if i%100==0: print i
            #if i<200: print i, words
            #if i>1300: break
            if len(words)>2 and words[0]=='EVENTS':
                for w in range(len(words)):
                    if words[w]=='RATE':
                        hz=float(words[w+1])
                        break
            if len(words)>5 and words[2]=='DISPLAY_COORDS':
                cent=(float(words[5])/2.0,float(words[6])/2.0)
            if len(words)==4 and words[2]=='MONITORDISTANCE':
                distance=float(words[3])
            if len(words)>2 and words[2]=='TRIALID': t=float(words[3])
            if len(words)==0:
                if not ende:
                    ende=True
                    continue
                break
            if len(words)>2 and words[2]=='!CAL':
                if len(words)==3: PHASE=4
                elif words[3]=='VALIDATION' and not (words[5]=='ABORTED'):
                    if words[6] == 'RIGHT':
                        calib[-1].append([t,float(words[9]),float(words[11]),RIGHTEYE])
                        PHASE= -1
                    else:
                        calib.append([])
                        calib[-1].append([t,float(words[9]),float(words[11]),LEFTEYE])
           
            if len(words)>2 and words[2]=='PRETRIAL':
                eye=words[5]
                PHASE=0;t0[0]=float(words[1])
            if len(words)>2 and words[2]=='START':
                PHASE=1;t0[1]=float(words[1])
            if len(words)>2 and (words[2]=='DETECTION'):
                PHASE=2; t0[2]= float(words[1])
            #if len(words)>2 and words[2]=='POSTTRIAL':
            if len(words)>2 and (words[2]=='POSTTRIAL' or words[2]=='OMISSION'):
                etdat = reformat(etdat,cent,t0[0],distance)
                if etdat.size>0:
                    et=ETTrialData(etdat,calib,t0,[vp,block,t,hz,eye])
                    data.append(et)
                
                etdat=[];t0=[0,0,0]
                calib=[];PHASE= -1
                LBLINK=False; RBLINK=False
            if PHASE== -1 or PHASE==4: continue
            if words[0]=='SBLINK' or words[0]=='EBLINK':
                if words[1]=='L': LBLINK= not LBLINK
                else : RBLINK= not RBLINK
            #if words[0]=='2133742': print words, LBLINK,RBLINK
            try: # to gather data
                if len(words)>5:
                    # we check whether the data gaze position is on the screen
                    if words[1]=='.': xleft=np.nan; yleft=np.nan
                    else:
                        xleft=float(words[1]); yleft=float(words[2])
                        if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                            xleft=np.nan; yleft=np.nan;
                    if words[4]=='.': xright=np.nan; yright=np.nan
                    else:
                        xright=float(words[4]); yright=float(words[5])
                        if xright>cent[0]*2 or xright<0 or yright<0 or yright>cent[1]*2:
                            xright=np.nan; yright=np.nan;
                    meas=(float(words[0]),xleft,yleft,float(words[3]),
                        xright,yright,float(words[6]))
                    size=7
                else:
                    if LBLINK or RBLINK: xleft=np.nan; yleft=np.nan
                    else:
                        xleft=float(words[1]); yleft=float(words[2])
                        if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                            xleft=np.nan; yleft=np.nan;
                    meas=(float(words[0]),xleft,yleft,float(words[3]))
                    size=4
                etdat.append(meas)
            except: pass
            #if len(trial)>0:print len(trial[0])
            #if len(dcorr)>1: print dcorr[-1]
        f.close()
    except: f.close(); raise
    data=discardInvalidTrials(data)
    return data

#data=readEdf(vp=7,block=1)

#tr=data[0]
#vel=tr.getVelocity()
#acc=tr.getAcceleration()
#sac=tr.getSaccades()

##d=np.loadtxt('VP001B1.csv')
##d[d==-1]=np.nan
###d=d[~np.any(d[:,[1,2,4,5]]==-1,axis=1),:]
##et=ETTrialData([],60,3,1,1,1)
##d[:,[1,4]]=pix2deg(d[:,[1,4]]-640,60)
##d[:,[2,5]]=pix2deg(d[:,[2,5]]-512,60)
##
##dd=np.copy(d)
##D=np.zeros((130,50))
##for f in range(10,140):
##    for i in [1,2,4,5]:
##        dd[:,i]=et.filterGaussian(d[:,i],f)
##    et=ETTrialData(dd,60,3,1,1,1)
##    vel=et.getVelocity()
##    D[f-10,:],discard=np.histogram(np.log(vel[~np.isnan(vel)]),50,range=(-2,6))
##plt.imshow(D,extent=(-2,6,140,10),aspect=0.05)
##
##plt.figure()
##fs=[10,20,24,28,32,36,40]
##for j in range(len(fs)):
##    if j<len(fs):
##        for i in [1,2,4,5]:
##            dd[:,i]=et.filterGaussian(d[:,i],fs[j])
##    else: dd=np.copy(d)
##    et=ETTrialData(dd,60,3,1,1,1)
##    vel=et.getVelocity()
##    plt.subplot(3,3,j+1)
##    plt.hist(np.log(vel[~np.isnan(vel)]),50,range=(-2,6))
##    plt.xlim([-2,6])
##    plt.ylim([0,300])
##    plt.title(str(fs[j]))

##    ws=50
##
##    ax1 = plt.subplot2grid((3,2), (0,0), colspan=2,rowspan=2)
##    ax1.set_aspect('equal')
##    ax2 = plt.subplot2grid((3,2), (2,0), colspan=2)
##    ax2.plot(np.arange(0,vel.size*4,4)/1000.0,np.log(vel))
##    ax2.set_ylim([0,5])
##    for v in range(vel.size-ws):
##        
##        ax1.scatter(tr.gaze[v+ws,1],tr.gaze[v+ws,2])
##        ax1.set_xlim([-10,10])
##        ax1.set_ylim([-10,10])
##        
##        ax2.set_xlim([v/250.0,(v+ws)/250.0])
##        plt.draw()
##        if v % 2 ==0:
##            print v       
def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
  
def readTobii(vp,block,lagged=False):
    ''' function for reading the tobii controller outputs list of ETDataTrial instances

        Each trial starts with line '[time]\tTrial\t[nr]'
        and ends with line '[time]\tOmission'

        lagged - return time stamp when the data was made available (ca. 30 ms time lag)
    '''
    path = getcwd()
    path = path.rstrip('/code')
    f=open(path+'/tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    #f=open('tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    try:
        data=[];trial=[]; theta=[];t=0;msgs=[]
        on=False
        while True:
            words=f.readline()
            if len(words)==0: break
            words=words.strip('\n').strip('\r').split('\t')
            if not on: # collect header information
                if len(words)==2 and  words[0]=='Monitor Distance': 
                    distance=float(words[1])
                if len(words)==2 and  words[0]=='Monitor Width': 
                    monwidth=float(words[1])
                if len(words)==2 and words[0]=='Recording time:':
                    recTime=words[1]
                if len(words)==2 and words[0]=='Recording refresh rate: ':
                    hz=float(words[1])
                if len(words)==2 and words[0]=='Recording resolution': 
                    cent=words[1].rsplit('x')
                    cent=(int(cent[0])/2.0,int(cent[1])/2.0)
                    ratio=cent[1]/float(cent[0])
                    print cent
                if len(words)==4 and words[2]=='Trial':
                    on=True; tstart=float(words[0])
            elif len(words)>=11: # record data
                # we check whether the data gaze position is on the screen
                xleft=float(words[2]); yleft=float(words[3])
                if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                    xleft=np.nan; yleft=np.nan;
                xright=float(words[5]); yright=float(words[6])
                if xright>cent[0]*2 or xright<0 or yright<0 or yright>cent[1]*2:
                    xright=np.nan; yright=np.nan;

                if lagged: tm =float(words[0])+float(words[8]);ff=int(words[1])
                else: tm=float(words[0]);ff=int(words[1])-2
                tdata=(tm,xleft,yleft,float(words[9]),
                    xright,yright,float(words[10]),ff)
                trial.append(tdata)
            elif (words[2]=='Detection' or words[2]=='Omission'):
                # we have all data for this trial, transform to deg and append
                on=False
                trial=np.array(trial)
                trial[:,0]-= tstart
                trial[trial==-1]=np.nan # TODO consider validity instead of coordinates
                
                trial[:,1]=myPix2deg(trial[:,1],distance,cent[0],monwidth)
                trial[:,2]=-myPix2deg(trial[:,2],distance,cent[1],monwidth*ratio)
                trial[:,4]=myPix2deg(trial[:,4],distance,cent[0],monwidth)
                trial[:,5]=-myPix2deg(trial[:,5],distance,cent[1],monwidth*ratio)
                et=ETTrialData(trial,[],[],hz,'BOTH',vp,block,t,recTime=recTime)
                et.theta=np.array(theta);theta=[]
                et.msg=msgs; msgs=[]
                data.append(et)
                bla
                t+=1
                trial=[]
            #elif len(words)==4 and words[1]=='Theta': theta.append([float(words[0]),float(words[2])])
            elif len(words)==6: msgs.append([float(words[0])-tstart,words[2]+' '+words[5]])
            elif words[2]!='Phase': msgs.append([float(words[0])-tstart,int(words[1]),words[2]])
    except: f.close(); raise
    f.close()
    return data

def checkDrift():
    # lets check whether there is a drift in the tobii data
    plt.close('all')
    d=np.array(dcorr)
    print d.shape
    # left
    for t in [0,1,2,7,8,9]:
        plt.plot(((d[t,:,1]-640)**2+ (d[t,:,2]-512)**2)**0.5 )
    plt.legend(['0','1','2','7','8','9'])
    # right
    plt.figure()
    for t in [0,1,2,7,8,9]:
        plt.plot(((d[t,:,4]-640)**2+ (d[t,:,5]-512)**2)**0.5 )
    plt.legend(['0','1','2','7','8','9'])
    plt.figure()
    plt.plot(((d[:,:,1].mean(1)-640)**2+(d[:,:,2].mean(1)-512)**2)**0.5)
    plt.plot(((d[:,:,4].mean(1)-640)**2+(d[:,:,5].mean(1)-512)**2)**0.5)
    plt.legend(['left','right'])

def findDriftEyelink(vp,block):
    plt.close('all')
    data=readEdf(vp,block)
    #print data[1].dcorr.shape
    for dd in range(8):
        S=dd*5
        E=(dd+1)*5
        N=E-S
        plt.figure(figsize=(16,12))
        for i in range(N):
            dat=data[S+i].dcorr
            both=dat.shape[1]>4
            plt.subplot(N,3,3*i+1)
            plt.plot(dat[:,0],dat[:,1],'g')
            if both: plt.plot(dat[:,0],dat[:,4],'r')
            plt.title('X axis')
            plt.xlim([0, 1200])
            plt.ylim([-3, 3])
            plt.ylabel('trial %d'% (S+i))
            plt.plot([0,1200],[0,0],'k')
            plt.subplot(N,3,3*i+2)
            plt.plot(dat[:,0],dat[:,2],'g')
            if both: plt.plot(dat[:,0],dat[:,5],'r')
            plt.title('Y axis')
            plt.xlim([0, 1200])
            plt.ylim([-3, 3])
            plt.plot([0,1200],[0,0],'k')

            
            if both:
                difx=np.abs(np.diff(dat[:,[1,4]].mean(axis=1),3))
                dify=np.abs(np.diff(dat[:,[2,5]].mean(axis=1),3))
            else:
                difx=np.abs(np.diff(dat[:,1],3))
                dify=np.abs(np.diff(dat[:,2],3))
            if ((vp==31 and block==4 and S+i==17) or
                (vp==31 and block==4 and S+i==16) or
                (vp==31 and block==4 and S+i<=12) or
                (vp==33 and block==1 and (S+i==12 or S+i==16 or S+i==17 or S+i==22 or S+i==23))):
                difx=np.abs(np.diff(dat[:,1],3))
                dify=np.abs(np.diff(dat[:,2],3))
            dif=np.sqrt(difx**2+dify**2)
            plt.subplot(N,3,3*i+3)
            plt.plot(dat[2:-1,0],dif)
            plt.xlim([0, 1200])
            start=(dat[2:-1,0]>190).nonzero()[0][0]
            mx=np.nanmax(dif[start:])
            if mx>0.2:
                k=(dif==mx).nonzero()[0][0]
                if vp==30 and block==3 and S+i==14: k=103
                if vp==31 and block==2 and S+i==26: k=104
                if vp==31 and block==4 and S+i==20: k=94
                dx=dat[k+1,1]-dat[k+2,1]
                dy=dat[k+1,2]-dat[k+2,2]
                plt.plot(dat[k+2,0],mx,'ko')
                #print dat[k+2,0],np.max(dif[80:]),np.argmax(dif[80:]),dat[-1,0]
            else: dx=0;dy=0
            if ((vp==30 and block==3 and S+i==14) or
                (vp==30 and block==3 and S+i==36) or
                (vp==32 and block==1 and S+i==3) or
                (vp==33 and block==1 and S+i==26) or
                (vp==33 and block==1 and S+i==7)): dx=0; dy=0
            plt.title('dx=%.3f, dy=%.3f'%(dx,dy))
            #if S+i==7: bla

def plotLTbabyPilot(vpn=range(101,112),maxTrDur=120):
    plt.close('all')
    labels=[]
    #vpn=range(101,112)#[102,106,113]#range(101,106)
    N=len(vpn)
    for ii in range(N): labels.append('vp %d'%vpn[ii])
    
    D=np.zeros((N,12))*np.nan
    DC=np.zeros((N,2))
    kk=0
    os.chdir('..')
    
    for vp in vpn:
        plt.figure()
        data=readTobii(vp,0)
        ordd=np.load(os.getcwd()+'/input/vp%d/ordervp%db0.npy'%(vp,vp))
        print vp,ordd
        for i in range(len(data)):
            cond=np.isnan(data[i].gaze[:,1])==False
            D[kk,i]=cond.sum()/60.0
            x=data[i].gaze[cond,0]/1000.0
            plt.plot(x,
                     data[i].gaze[cond,4]*0+i+1,'.b')
            ls=np.nan
            for msg in data[i].msg:
                if msg[1]=='Reward On':
                    ls=float(msg[0])/1000.0
                elif msg[1]=='Reward Off':
                    if np.isnan(ls): print 'error ls is nan'
                    plt.plot([ls, float(msg[0])/1000.0],[i+1.2,i+1.2],lw=4)
                    ls=np.nan
                elif msg[1]=='10 th saccade ':
                    plt.plot(float(msg[0])/1000.0,i+1,'xr')
            if not np.isnan(ls):
                plt.plot([ls,x[-1]],[i+1.2,i+1.2],lw=4)
                #print msg[1]    
        #DC[kk,0]=D[kk,ordd<5].mean()
        #DC[kk,1]=D[kk,ordd>=5].mean()
        plt.ylabel('Trial')
        plt.xlabel('Time in seconds')
        plt.xlim([0, maxTrDur])
        plt.ylim([0.5,len(data)+0.5])
        plt.title('VP %d' % vp)
        kk+=1
    plt.figure()
    plt.plot(D.T)
    plt.ylabel('Total Looking Time')
    plt.ylim([0, maxTrDur])
    plt.xlabel('Trial')
    plt.legend(labels)
    plt.figure()
    plt.plot(np.repeat([[6],[8]],4,axis=1),DC.T,'x',markersize=10)
    plt.xlim([5,12])
    plt.xlabel('Number of rings')
    plt.ylabel('Average looking time per trial in seconds')
    plt.legend(labels)
    plt.figure()
    plt.imshow(D,interpolation='nearest',vmin=0,
        vmax=maxTrDur,extent=[0.5,12.5,vpn[-1]+0.5,vpn[0]-0.5])
    ax=plt.gca()
    ax.set_yticks(np.arange(vpn[0],vpn[-1]+1))
    plt.show()
    plt.colorbar()
    
def checkEyelinkDatasets():
    for vp in range(20,70):
        for block in range(0,5):
            try:
                data=readEdf(vp,block)
                if not (len(data) == 40 or (len(data)==10 and  block==0)):
                    print 'error ', vp, block, len(data)
                else:
                    print '    ok', vp, block, len(data)
            except:
                print 'missing ', vp, block

if __name__ == '__main__':

    
    
    data=readEdf(18,9)

    #evs=[]
    for i in range(len(data)):
        if data[i].ts>=0:
            print i
            data[i].driftCorrection()
            data[i].extractTracking()
            data[i].exportTracking()
            #evs.extend(data[i].track)
            data[i].plotTracking()

##    
##    #data = readTobii(129,0)
##    #print data[1].msg
##    #plotLTbabyPilot(range(125,126))
##    #time.sleep(5)
    
