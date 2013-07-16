import numpy as np
import pylab as plt
import matplotlib as mpl
from Settings import *
from Constants import *
from EvalSettingsBaby import *
from psychopy import core,event,visual
from scipy import signal
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import nanmean
from copy import copy
import time,os


plt.ion()
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
    return out

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

def computeState(isFix,md,nfm=np.inf):
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

def interpolateBlinks(t,d,hz):
    ''' interpolate short missing intervals 
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

def tseries2eventlist(tser):
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
    
def computeFixations(tser,vel,acc,hz):
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
    temp=np.logical_and(vel>OLPURVTHL,
        np.logical_and(vel<OLPURVTHU,np.abs(acc)<OLPURATH))
    return computeState(temp,[OLPURMD*hz,np.inf])
def computeCLpursuit(tser,vel,acc,hz):
    temp=np.logical_and(vel>CLPURVTHL,
        np.logical_and(vel<CLPURVTHU,np.abs(acc)<CLPURATH))
    return computeState(temp,[CLPURMD*hz,np.inf])

def computeSaccades(tser,vel,acc,hz):
    isFix=np.logical_or(np.isnan(vel),
        np.logical_or(vel>SACVTH,np.abs(acc)>SACATH))
    discard, b = computeState(np.logical_or(vel>SACVTH,np.abs(acc)>SACATH),[SACMINDUR*hz,np.inf])
    a,discard= computeState(isFix,[SACMINDUR*hz,np.inf])
    return a,b

def computeLongSaccades(tser,vel,acc,hz):
    isFix=np.logical_or(vel>LSACVTH,np.abs(acc)>LSACATH)
    return computeState(isFix,[LSACMINDUR*hz,np.inf])

def computeBlinks(tser,hz):
    isFix=np.isnan(tser[:,7])
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

            
class ETTrialData():
    def __init__(self,dat,calib,t0,info,recTime="0:0:0",fs=None,
                 INTERPBLINKS=False,fcutoff=70,focus=BINOCULAR,msgs=[]):
        self.calib=calib
        self.vp=int(info[0])
        self.block=int(info[1])
        self.trial=int(info[2])
        self.hz=float(info[3])
        self.eye=info[4]
        self.t0=[t0[1]-t0[0],t0[2]-t0[0]]
        self.fs=fs
        self.msgs=msgs
        if self.fs==None: self.fs= self.computeFs()
        self.focus=focus # which eye use to indicate focus, if binocular, use gaze average
        if self.t0[0]>=0: self.ts=min((dat[:,0]>(self.t0[0])).nonzero()[0])
        else: self.ts=-1
        #print 'hhh',self.t0, self.ts
        
        if  self.t0[1]>0 and len((dat[:,0]>(self.t0[1])).nonzero()[0])>0: 
            self.te=min((dat[:,0]>(self.t0[1])).nonzero()[0])
        elif self.t0[1]>0: self.te=dat.shape[0]-1
        self.recTime=datetime.strptime(recTime,"%H:%M:%S")
        
    def computeFs(self):
        path = getcwd()
        path = path.rstrip('code')
        f=open(path+'input/vp%03d/SettingsExp.pkl'%self.vp)
        while f.readline().count('refreshRate')==0: pass
        f.readline();
        monhz=float(f.readline().lstrip('F').rstrip('\r\n'))
        f.close()
        dur=self.t0[1]-self.t0[0]
        N=int(round((dur)*monhz/1000.0))
        return np.array([range(N),np.linspace(0,dur,N)]).T
        
    
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
        for p in self.fev:
            if t>g[p[0],0]-self.t0[0] and t<g[p[1]-1,0]-self.t0[0]:
                return p[2:]
        return []
    def getTrackedAgent(self,t):
        g=self.gaze[self.ts:self.te,0]
        gmax=g.size-1
        for tr in self.track:
            if t>g[tr[0]]-self.t0[0] and t<g[min(gmax,tr[1])]-self.t0[0]: 
                return tr[2]
        return []
    def loadTrajectories(self):
        path = getcwd()
        path = path.rstrip('/code')
        print self.trial
        order = np.load(path+'/input/vp%03d/ordervp%03db%d.npy'%(self.vp,self.vp,self.block))[self.trial]
        s=path+'/input/vp%03d/vp%03db%dtrial%03d.npy'%(self.vp,self.vp,self.block,order) 
        traj=np.load(s)
        self.oldtraj=traj
        
    def computeAgentDistances(self):
        self.loadTrajectories()
        #self.traj=traj
        g=self.getGaze()
        traj=self.oldtraj[:self.fs.shape[0],:,:]
        self.dist=np.zeros((g.shape[0],traj.shape[1]))
        self.traj=np.zeros((g.shape[0],traj.shape[1],traj.shape[2]))
        self.dev=np.zeros((g.shape[0]-1,traj.shape[1]))
        for a in range(traj.shape[1]):
            #print tt.shape, traj[:,a,0].shape
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

            
    def driftCorrection(self,jump=0):
        """ performs drift correction for each eye and axis
            separately based on the fixation location immediately
            preceding the trial onset
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
        if jump==-1: 
            print 'skipping drift correction'
            return
        # find the latest fixation
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
        # recompute events
        self.extractBasicEvents(self.gaze[:,:7])
        self.extractPursuitEvents()
        
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
        # build high-level events
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
                or ev[-1]==CLPUR and not len(set(lastA) & set(A)) and (ev[1]-ev[0])/self.hz>0.3 # no search but long pursuit
                ): # start tracking event
                if len(info):
                    if len(info[-1])==0: info.pop(-1)
                    hev.append(info);
                info=[ev[0],ev[1],True,[]]
                trackon=True
                #acounter=np.zeros(self.traj.shape[1])
            elif (len(set(lastA) & set(A)) or (len(A) and np.any(acounter[A]>len(info[3])/2))) and trackon: # continue tracking
                info[1]=ev[1]
            else: # terminate event
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
        # identify search and tracking, look 
        #for first block isolated by saccades wheter it contians pursuit
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
        self.track=track
        # identify tracked agents
        self.computeTrackedAgents()
        # and merge tracking events with similar agents
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
                    self.track[i-1].append(selectAgentTRACKING(self.track[i-1][0],
                        self.track[i-1][1],self.events))
                    self.track.pop(i)
            else: i+=1
            
    def computeTrackedAgents(self):
        for tr in self.track:
            if len(tr)==2:tr.append(selectAgentTRACKING(tr[0],tr[1],self.events))
            
    def plotTracking(self):
        ''' plots events '''
        #plt.figure()
        ax=plt.gca()
        for f in self.track:
            if f[-1]:r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.4,color='k')
            else: r=mpl.patches.Rectangle((f[0],self.trial-0.1),f[1]-f[0],0.8,color='r')
            ax.add_patch(r)
        plt.xlim([0,15000])
        plt.ylim([-1,41])
        plt.show()
    def exportEvents(self):
        out=[]
        for ev in self.events:
            if ev[-1]!=SAC:
                out.append([ev[0]*2,ev[1]*2,ev[-1]])
        np.savetxt('vp%02db%02dt%02d.evt'%(self.vp,self.block,self.trial),np.array(out).T,fmt='%d')
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

    def plotMissingData(self):
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
        
    @staticmethod
    def helpf(fev,inds):
        out=[]
        for ff in fev:
            if ((ff[0]>inds[0] and ff[0]<inds[1])
                or (ff[1]>inds[0] and ff[1]<inds[1])):
                temp=[max(ff[0]-inds[0],0),min(ff[1]-inds[0],inds[1]-inds[0]-1)]+ff[2:]
                out.append(temp)
        return out
        
    def extractBasicEvents(self,dat):
        """ extract and filter gaze location,
            compute velocity, acceleration
            and find fixations
        """
        
        # add two columns with binocular gaze point
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
        self.isLSac,discard=computeLongSaccades(self.gaze,self.vel,self.acc,self.hz)
        self.isBlink,bev=computeBlinks(self.gaze,self.hz)
        self.fev=ETTrialData.helpf(fev,[self.ts,self.te])
        self.sev=ETTrialData.helpf(sev,[self.ts,self.te])
        self.bev=ETTrialData.helpf(bev,[self.ts,self.te])
        self.computeAgentDistances()
        for i in range(len(self.fev)):
            s=self.fev[i][0];e=self.fev[i][1]-1
            a=selectAgentFIX(self.dist[s:e,:],self.hz)
            self.fev[i].extend(a)
        self.opev=[];self.cpev=[]
    def extractPursuitEvents(self):
        # find pursuit states for phase 2
        opur,opev=computeOLpursuit(self.gaze,self.vel,self.acc,self.hz)
        cpur,cpev=computeCLpursuit(self.gaze,self.vel,self.acc,self.hz)
        self.opev=ETTrialData.helpf(opev,[self.ts,self.te])
        self.cpev=ETTrialData.helpf(cpev,[self.ts,self.te])
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
    def selectPhase(self,data,phase,hz):
        """ phase:
            -1 - select all (used with opur, cpur)
            0 - drift correction
            1 - trial
            2 - target selection
            3 - [-200,200] ms around trial start (used for checking drift correction)
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
        out=ETTrialData.resample(out,hz)
        return out[:,1]
    def getTraj(self,hz=None):
        t=self.fs[:,1]
        try: out=self.oldtraj[:t.size,:,:]
        except AttributeError: self.loadTrajectories(); out=self.oldtraj[:t.size,:,:]
        res=[]
        res.append(ETTrialData.resample(np.concatenate((np.array(t,ndmin=2).T,out[:,:,0]),axis=1),hz))
        res.append(ETTrialData.resample(np.concatenate((np.array(t,ndmin=2).T,out[:,:,1]),axis=1),hz))
        res= np.array(res)
        res=np.rollaxis(res,0,3)
        return res[:,1:,:]
    
    def getGaze(self,phase=1,hz=None):
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
        out=ETTrialData.resample(out,hz)
        return out

    def getVelocity(self,phase= 1,hz=None):
        return self.selectPhase(self.vel,phase,hz)
    def getAcceleration(self,phase=1,hz=None):
        return self.selectPhase(self.acc,phase,hz)
    def getFixations(self,phase=1,hz=None):
        return self.selectPhase(self.isFix,phase,hz)
    def getSaccades(self,phase=1,hz=None):
        return self.selectPhase(self.isSac,phase,hz)
    def getLongSaccades(self,phase=1,hz=None):
        return self.selectPhase(self.isLSac,phase,hz)
    def getCLP(self,phase=1,hz=None):
        return self.selectPhase(self.cpur,-1,hz)
    def getOLP(self,phase=1,hz=None):
        return self.selectPhase(self.opur,-1,hz)
    def getHEV(self,phase,hz=None):
        out=np.zeros(hz.size)
        for tr in self.hev:
            if tr[2]: out[t2f(tr[0]/float(self.hz)*1000,hz):t2f(tr[1]/float(self.hz)*1000,hz) ]=1
        return out
    def getTracking(self,phase,hz=None):
        out=np.zeros(hz.size)
        for tr in self.track:
            out[t2f(tr[0]/float(self.hz)*1000,hz):t2f(tr[1]/float(self.hz)*1000,hz) ]=1
        return out


def manualDC(vp,b,t):
    """ do we need manual drift correction?"""
    out=0
    if vp==1:
        if b==3 and t==25: out=-10
        elif b==3 and t==34: out=-1
        elif b==5 and t==10: out=50
        elif b==6 and t==19: out=20
        elif b==6 and t==24: out=20
        elif b==8 and t==14: out=50
        elif b==11 and t==17: out=-1
        elif b==12 and t==37: out=-1
        elif b==14 and t==38: out=-1
        elif b==18 and t==8: out=-1
        elif b==19 and t==6: out=-1 
    return out
def plotDC():
    plt.interactive(False)
    vp=1
    from readETData import readEyelink
    for b in range(1,22):
        print 'block ', b
        data=readEyelink(vp,b)
        for i in range(len(data)):
            plt.figure()
            d=data[i]
            gg=d.getGaze(phase=3)
            plt.plot(gg[:,0],gg[:,1],'g--')
            plt.plot(gg[:,0],gg[:,2],'r--')
            plt.plot(gg[:,0],gg[:,4],'b--')
            plt.plot(gg[:,0],gg[:,5],'k--')
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
            plt.savefig(os.getcwd()+'/pic/dc/b%02dtr%02d'%(b,i))
            plt.close()
            i+=1 

def plotMD():
    vp=1
    from readETData import readEyelink
    for b in range(1,22):
        plt.cla()
        print 'block ', b
        data=readEyelink(vp,b)
        for d in data:
            d.driftCorrection(jump=manualDC(vp,b,i))
            d.plotMissingData()
        plt.show()
        plt.savefig(os.getcwd()+'/pic/missingdata/fb%02d'%(d.block))
            
