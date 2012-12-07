import numpy as np
import pylab as plt
from Settings import *
from Constants import *
from psychopy import core,event,visual
from scipy import signal
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.interpolate import interp1d
import time,os

def interpRange(xold,yold,xnew):
    ynew=xnew.copy()
    f=interp1d(xold,yold)
    for i in range(xnew.size):
        ynew[i] =f(xnew[i])
    return ynew  

# TODO: readTobii() for reading drift correction data
plt.ion()
class ETTrialData():
    LEFTEYE=1
    RIGHTEYE=2
    BINOCULAR=3
    SACVTH=15#22 # velocity threshold deg/sec
    SACATH=1500#4000 # acceleration threshold deg/sec^2
    FIXVTH=18#10
    FIXATH=1500#1000
    def __init__(self,gaze,dcorr,calib,hz,eye,vp,block,trial,recTime="0:0:0",
                 INTERPBLINKS=False,fcutoff=70):#70):
        self.gaze=gaze
        self.dcorr=dcorr
        self.calib=calib
        self.hz=hz
        self.eye=eye
        self.vp=vp
        self.block=block
        self.trial=trial
        self.recTime=datetime.strptime(recTime,"%H:%M:%S")
        if INTERPBLINKS:# blinks - do linear interpolation
            isblink=np.isnan(self.gaze)
            blinkon = np.bitwise_and(isblink,np.bitwise_not(
                np.roll(isblink,1))).nonzero()[0].tolist()
            blinkoff=np.bitwise_and(np.roll(isblink,1),
                np.bitwise_not(isblink)).nonzero()[0].tolist()
            if len(blinkon)!=len(blinkoff):
                print 'Blink Interpolation Failed'
                raise TypeError
            for b in range(len(blinkon)):
                bs=blinkon[b]-1
                be=(blinkoff[b]+1)
                dur=be-bs
                #print dur
                for c in range(self.gaze.shape[1]):
                    self.gaze[bs:be,c]=np.linspace(self.gaze[bs,c],
                        self.gaze[be-1,c],dur)
        #if fcutoff>0:
        #    for i in [1,2,4,5]:
        #        self.gaze[:,i]=self.filterGaussian(self.gaze[:,i],fcutoff)

    def getGazeData(self,t=[]):
        t=np.array(t,ndmin=1)
        if t.size==0: return self.gaze
        else:
            gp=np.ones((t.size,self.gaze.shape[1]))*np.nan
            for k in range(1,self.gaze.shape[1]):
                f=interp1d(self.gaze[:,0],self.gaze[:,k])
                for i in range(t.size):
                    gp[i,k] =f(t[i])
            gp[:,0]=t
            return gp
            
        
    def filterButterworth(self,x,cutoff=50):
        
        #normPass = 2*np.pi*cutoff/self.hz*2
        normPass = cutoff / (self.hz/2.0)
        normStop = 1.5*normPass
        print normPass, normStop
        (N, Wn) = signal.buttord(wp=normPass, ws=normStop,
                    gpass=2, gstop=30, analog=0)
        (b, a) = signal.butter(N, Wn)
        print b.shape
        #b *= 1e3
        # return signal.lfilter(b, a, x[::-1])[::-1]
        return signal.lfilter(b, a, x)
    
    def filterGaussian(self,x,cutoff=25):
        #return self.filterCausal(x)
        # cutoff in hz
        sigma=self.hz/float(cutoff)
        #print np.ceil(12*sigma), sigma# sigma, self.hz, cutoff
        fg=signal.gaussian(np.ceil(5*sigma),sigma,sym=False)
        fg=fg/fg.sum()
        #print fg.shape,x.shape
        return np.convolve(x,fg,mode='same')

    def filterCausal(self,x,theta=0.6):
        y=np.zeros(x.shape)
        y[0]=x[0]
        for i in range(1,x.size):
            if np.isnan(y[i-1]): y[i]=x[i]
            else: y[i]=(theta)*y[i-1] +(1- theta)*x[i]
        return y
            
    
    def getVelocity(self,filt=False):
        if not filt: 
            vel=(np.sqrt(np.diff(self.gaze[:,1])**2
                    +np.diff(self.gaze[:,2])**2)*self.hz)
            return vel
        else:
            try: return self.vel
            except AttributeError:
                self.vel=(np.sqrt(np.diff(self.gaze[:,1])**2
                        +np.diff(self.gaze[:,2])**2)*self.hz)
                self.vel= self.filterGaussian(self.vel)
                return self.vel

            else: return self.vel
    def getAcceleration(self, filt=False):
        try: return self.acc
        except AttributeError:
            self.acc=np.concatenate((np.diff((self.getVelocity(False))*self.hz),[0]),0)
            if filt: self.acc=self.filterGaussian(self.acc)
            return self.acc

  
        
    def getSaccades(self,filt=True):
        try: return np.int32(self.isSac)#self.saccades
        except AttributeError:
            self.saccades=[]
            isSac=np.bitwise_or(self.getVelocity(filt=filt)>ETTrialData.SACVTH,
                self.getAcceleration(filt=filt)>ETTrialData.SACATH)
            sacon = np.bitwise_and(isSac,
                np.bitwise_not(np.roll(isSac,1))).nonzero()[0].tolist()
            sacoff=np.bitwise_and(np.roll(isSac,1),
                np.bitwise_not(isSac)).nonzero()[0].tolist()
            #print sacon[0], sacoff[0]
            if sacon[-1]>sacoff[-1]:
                sacoff.append(self.gaze.shape[0]-1)
            if sacon[0]>sacoff[0]:
                sacon.insert(0,0)
            if len(sacon)!=len(sacoff):
                print 'invalid fixonoff'
                raise TypeError
            for f in range(len(sacon)):
                fs=sacon[f]
                fe=(sacoff[f]+1)
                dur=fe-fs
                #print dur
                if dur>0.06*self.hz:
                    #print dur
                    self.saccades.append([fs,dur,self.gaze[fs:fe,1].mean(),
                        self.gaze[fs:fe,2].mean()])
                else: 
                    isSac[fs:fe]=False
            self.saccades=np.array(self.saccades)
            self.isSac=isSac
            return np.int32(self.isSac)#self.saccades
            
    def getFixations(self,filt=True):
        try: return self.isFix#self.saccades
        except AttributeError:
            self.fixations=[]
            isFix=np.bitwise_and(self.getVelocity(filt=filt)<ETTrialData.FIXVTH,
                np.abs(self.getAcceleration(filt=filt))<ETTrialData.FIXATH)
            fixon = np.bitwise_and(isFix,
                np.bitwise_not(np.roll(isFix,1))).nonzero()[0].tolist()
            fixoff=np.bitwise_and(np.roll(isFix,1),
                np.bitwise_not(isFix)).nonzero()[0].tolist()
            if fixon[-1]>fixoff[-1]:
                fixoff.append(self.gaze.shape[0]-1)
            if fixon[0]>fixoff[0]:
                fixon.insert(0,0)
            if len(fixon)!=len(fixoff):
                print 'invalid fixonoff'
                raise TypeError
            for f in range(len(fixon)):
                fs=fixon[f]
                fe=(fixoff[f]+1)
                dur=fe-fs
                if dur<0.1*self.hz:
                    self.fixations.append([fs,dur,self.gaze[fs:fe,1].mean(),
                        self.gaze[fs:fe,2].mean()])
                    isFix[fs:fe]=False
            self.fixations=np.array(self.fixations)
            self.isFix=np.int32(isFix)
            return self.isFix
            
    def getPursuit(self):
        try: return self.isPur
        except AttributeError:
            isPur=(1-self.getFixations())*(1-self.getSaccades())>0.5
            puron = np.bitwise_and(isPur,
                np.bitwise_not(np.roll(isPur,1))).nonzero()[0].tolist()
            puroff=np.bitwise_and(np.roll(isPur,1),
                np.bitwise_not(isPur)).nonzero()[0].tolist()
            if puron[-1]>puroff[-1]:
                puroff.append(self.gaze.shape[0]-1)
            if puron[0]>puroff[0]:
                puron.insert(0,0)
            if len(puron)!=len(puroff):
                print 'invalid puronoff'
                raise TypeError
            for f in range(len(puron)):
                fs=puron[f]
                fe=(puroff[f]+1)
                dur=fe-fs
                if dur<0.150*self.hz:
                    isPur[fs:fe]=False
            self.isPur=np.int32(isPur)
            return self.isPur
            
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
    

def readEdf(vp,block):
    def reformat(trial,cent,tstart,distance):
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
    f=open(path+'eyelinkOutput\\VP%03dB%d.ASC'%(vp,block),'r')
    BLINK=0
    ende=False
    try:
        line=f.readline()
        data=[]
        trial=[]
        dcorr=[]
        calib=[]
        i=0
        TAKE=False
        dcorrOn=False
        calOn=False
        t=0
        size=7
        while True:   
            words=f.readline().split()
            i+=1            
            #if i%100==0: print i
            #if i>6400: print i, words
            if len(words)>2 and words[0]=='EVENTS':
                for w in range(len(words)):
                    if words[w]=='RATE':
                        hz=float(words[w+1])
                        break
            if len(words)>2 and words[2]=='PRETRIAL':
                eye=words[5]
                dcorrOn=True
                tsdcorr=float(words[1])
            if len(words)>2 and words[2]=='!CAL':
                if len(words)==3: calOn=True
                elif words[3]=='VALIDATION' and not (words[5]=='ABORTED'):
                    #print words
                    calib.append(float(words[9]))
                    calib.append(float(words[11]))
                    if words[6] is 'RIGHT': calOn=False
            if len(words)>2 and words[2]=='POSTTRIAL':
                dcorrOn=False; #BLINK=0
                dcorr=[]
            if len(words)==4 and words[2]=='MONITORDISTANCE':
                distance=float(words[3])
                #print 'dist',distance
        
            if len(words)==0:
                if not ende:
                    ende=True
                    continue
                break
            if len(words)>5 and words[2]=='DISPLAY_COORDS':
                cent=(float(words[5])/2.0,float(words[6])/2.0)
            if len(words)>2 and words[2]=='START':
                TAKE=True;tstart=float(words[1]); dcorrOn=False
                continue
            if len(words)>2 and (words[2]=='DETECTION' or words[2]=='OMISSION'):
                TAKE=False
                trial = reformat(trial,cent,tstart,distance)
                dcorr = reformat(dcorr,cent,tsdcorr,distance)
                et=ETTrialData(trial,dcorr,calib,hz,eye,vp,block,t)
                data.append(et)
                t+=1
                trial=[]
                dcorr=[]
                calib=[]
                continue
            if words[0]=='SBLINK':
                BLINK+=1
            if words[0]=='EBLINK':
                BLINK-=1
            
            if not TAKE and not dcorrOn: continue
            if BLINK:
                if not dcorrOn: trial.append([np.nan]*size)
                else: dcorr.append([np.nan]*size)
                continue
                
            try:
                if len(words)>5:
                    # we check whether the data gaze position is on the screen
                    xleft=float(words[1]); yleft=float(words[2])
                    if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                        xleft=np.nan; yleft=np.nan;
                    xright=float(words[4]); yright=float(words[5])
                    if xright>cent[0]*2 or xright<0 or yright<0 or yright>cent[1]*2:
                        xright=np.nan; yright=np.nan;
                    
                    meas=(float(words[0]),xleft,yleft,float(words[3]),
                        xright,yright,float(words[6]))
                    size=7
                else:
                    xleft=float(words[1]); yleft=float(words[2])
                    if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                        xleft=np.nan; yleft=np.nan;
                    meas=(float(words[0]),xleft,yleft,float(words[3]))
                    size=4
                if not dcorrOn: trial.append(meas)
                else: dcorr.append(meas)
            except:
                pass
        f.close()
    except: f.close(); raise
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
    data = readTobii(129,0)
    print data[1].msg
    #plotLTbabyPilot(range(125,126))
    #time.sleep(5)
    




            
            
               
    









