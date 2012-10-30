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
    def __init__(self,gaze,dcorr,hz,eye,vp,block,trial,recTime="0:0:0",
                 INTERPBLINKS=False,fcutoff=70):#70):
        self.gaze=gaze
        self.dcorr=dcorr
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
            
def pix2deg(pix,dist):
        return np.arctan((pix/37.795275591)/float(dist))*180/np.pi

def readEdf(vp,block):
    def reformat(trial,cent,tstart,distance):
        trial=np.array(trial)
        if type(trial) is type( () ): print 'error in readEdf'
        trial[:,0]-=tstart
        trial[:,1]=pix2deg(trial[:,1]-cent[0],distance)
        trial[:,2]=-pix2deg(trial[:,2]-cent[1],distance)
        if trial.shape[1]==7:
            trial[:,4]=pix2deg(trial[:,4]-cent[0],distance)
            trial[:,5]=-pix2deg(trial[:,5]-cent[1],distance)
        return trial
    cent=(0,0)
    path = getcwd()
    path = path.rstrip('code')
    f=open(path+'eyelinkOutput/VP%03dB%d.ASC'%(vp,block),'r')
    BLINK=False
    ende=False
    try:
        line=f.readline()
        data=[]
        trial=[]
        dcorr=[]
        i=0
        TAKE=False
        dcorrOn=False
        t=0
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
            if len(words)>2 and words[2]=='POSTTRIAL':
                dcorrOn=False
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
                et=ETTrialData(trial,dcorr,hz,eye,vp,block,t)
                data.append(et)
                t+=1
                trial=[]
                dcorr=[]
                continue
            if words[0]=='SBLINK':
                BLINK=True
            if words[0]=='EBLINK':
                BLINK=False
            
            if not TAKE and not dcorrOn: continue
            if BLINK:
                trial.append([np.nan]*size)
                continue
                
            try:
                if len(words)>5:
                    
                    meas=(float(words[0]),float(words[1]),
                        float(words[2]),float(words[3]),
                        float(words[4]),float(words[5]),float(words[6]))
                    size=7
                else:
                    meas=(float(words[0]),float(words[1]),
                        float(words[2]),float(words[3]))
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
  
def readTobii(vp,block):
    ''' function for reading the tobii controller outputs list of ETDataTrial instances

        Each trial starts with line '[time]\tTrial\t[nr]'
        and ends with line '[time]\tOmission'
    '''
    f=open('tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    try:
        data=[];trial=[]; theta=[];t=0
        on=False
        while True:
            words=f.readline()
            if len(words)==0: break
            words=words.strip('\n').strip('\r').split('\t')
            if not on: # collect header information
                if len(words)==2 and  words[0]=='Monitor Distance': 
                    distance=float(words[1])
                if len(words)==2 and words[0]=='Recording time:':
                    recTime=words[1]
                if len(words)==2 and words[0]=='Recording refresh rate: ':
                    hz=float(words[1])
                if len(words)==2 and words[0]=='Recording resolution': 
                    cent=words[1].rsplit('x')
                    cent=(int(cent[0])/2.0,int(cent[1])/2.0)
                if len(words)==3 and words[1]=='Trial':
                    on=True; tstart=float(words[0])
            elif len(words)>=12: # record data
                # we check whether the data gaze position is on the screen
                xleft=float(words[1]); yleft=float(words[2])
                if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                    xleft=np.nan; yleft=np.nan;
                xright=float(words[1]); yright=float(words[2])
                if xright>cent[0]*2 or xright<0 or yright<0 or yright>cent[1]*2:
                    xright=np.nan; yright=np.nan;
                tdata=(float(words[0]),xleft,yleft,float(words[9]),
                    xright,yright,float(words[10]))
                trial.append(tdata)
            elif (words[1]=='Detection' or words[1]=='Omission'):
                # we have all data for this trial, transform to deg and append
                on=False
                trial=np.array(trial)
                trial[:,0]-= tstart
                trial[trial==-1]=np.nan # TODO consider validity instead of coordinates
                trial[:,1]=pix2deg(trial[:,1]-cent[0],distance)
                trial[:,2]=-pix2deg(trial[:,2]-cent[1],distance)
                trial[:,4]=pix2deg(trial[:,4]-cent[0],distance)
                trial[:,5]=-pix2deg(trial[:,5]-cent[1],distance)
                et=ETTrialData(trial,hz,'BOTH',vp,block,t,recTime=recTime)
                et.theta=np.array(theta);theta=[]
                data.append(et)
                t+=1
                trial=[]
            elif len(words)==3 and words[1]=='Theta': theta.append([float(words[0]),float(words[2])])
    except: f.close(); raise
    f.close()
    return data

##d=np.loadtxt('VP001B1.csv')
##d[d==-1]=np.nan
##d[:,[1,4]]=pix2deg(d[:,[1,4]]-640,60)
##d[:,[2,5]]=pix2deg(d[:,[2,5]]-512,60)
##et=ETTrialData(d,60,3,1,1,1)
##fix=et.getFixations()
##plt.plot(d[:,1],d[:,2],'.b',markersize=1)
##plt.plot(et.fixations[1:,2],et.fixations[1:,3],'.r')

##plt.close('all')
##data=readTobii(85,0)
##d=data[0]
##plt.plot(d.gaze[:,1])
##vel=d.getVelocity()
##nvel=[]
##for i in range(vel.size):
##    if not np.isnan(vel[i]):
##        nvel.append(vel[i])
##vel=np.array(nvel)
##d=data[3]
##plt.plot(d.getVelocity())
##plt.figure()
##plt.plot(d.getAcceleration())
##f=d.getSaccades()
###print f.sum()

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

if __name__ == '__main__':
    if False: # evaluate looking times
        plt.close('all')
        labels=[]
        vpn=range(106,107)#[102,106,113]#range(101,106)
        N=len(vpn)
        for ii in range(N): labels.append('vp %d'%vpn[ii])
        
        D=np.zeros((N,12))*np.nan
        DC=np.zeros((N,2))
        vp=104
        kk=0
        for vp in vpn:
            plt.figure()
            data=readTobii(vp,0)
            ordd=np.load('input/vp%d/ordervp%db0.npy'%(vp,vp))
            print vp,ordd
            for i in range(len(data)):
                D[kk,i]=(np.isnan(data[i].gaze[:,1])==False).sum()/60.0
                ende=data[i].gaze.shape[0]/60.0
                plt.plot(np.linspace(0,ende,data[i].gaze.shape[0]),
                         data[i].gaze[:,1]*0+i+0.5,'.b')
            DC[kk,0]=D[kk,ordd<5].mean()
            DC[kk,1]=D[kk,ordd>=5].mean()
            plt.ylabel('Trial')
            plt.xlabel('Time in seconds')
            plt.xlim([0, 30])
            plt.title('VP %d' % vp)
            kk+=1
        plt.figure()
        plt.plot(D.T)
        plt.ylabel('Total Looking Time')
        plt.ylim([0, 30])
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
            vmax=30,extent=[0.5,12.5,113.5,100.5])
        ax=plt.gca()
        ax.set_yticks(np.arange(101,114))
        plt.show()
        plt.colorbar()

    data=readEdf(40,0)
    print data[1].dcorr.shape
    np.save('dcorr.npy',data[1].dcorr)
    plt.show()
    for i in range(len(data)):
        plt.subplot(2,10,2*i+1)
        plt.plot(data[i].dcorr[:,0],data[i].dcorr[:,1],'g')
        plt.plot(data[i].dcorr[:,0],data[i].dcorr[:,4],'r')
        plt.title('X axis')
        plt.subplot(2,10,2*i+2)
        plt.plot(data[i].dcorr[:,0],data[i].dcorr[:,2],'g')
        plt.plot(data[i].dcorr[:,0],data[i].dcorr[:,5],'r')
        plt.title('Y axis')
    









