import numpy as np
from Settings import *
from os import getcwd
from evalETdata import ETTrialData, interpRange
import pylab as plt
plt.ion()

def _isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def _discardInvalidTrials(data):
    bb=range(len(data))
    bb.reverse()
    for i in bb:
        if data[i].ts<0: data.pop(i)
    return data
def _reformat(trial,tstart,Qexp):
    if len(trial)==0: return np.zeros((0,7))
    trial=np.array(trial)
    ms=np.array(Qexp.monitor.getSizePix())/2.0
    if type(trial) is type( () ): print 'error in readEdf'
    trial[:,0]-=tstart
    trial[:,1]=Qexp.pix2deg(trial[:,1]-ms[0])
    trial[:,2]=-Qexp.pix2deg(trial[:,2]-ms[1])
    if trial.shape[1]>4:
        trial[:,4]=Qexp.pix2deg(trial[:,4]-ms[0])
        trial[:,5]=-Qexp.pix2deg(trial[:,5]-ms[1])
    return trial

def readEyelink(vp,block):
    cent=(0,0)
    path = getcwd()
    path = path.rstrip('code')
    try:
        f=open(path+'eyelinkOutput/VP%03dB%d.asc'%(vp,block),'r')
    except:
        f=open(path+'eyelinkOutput/VP%03dB%d.ASC'%(vp,block),'r')
    Qexp=Settings.load(Q.inputPath+'vp%03d'%vp+Q.delim+'SettingsExp.pkl' )
    LBLINK=False; RBLINK=False
    ende=False
    try:
        line=f.readline()
        data=[]
        PHASE= -1 # 0-DCORR, 1-TRIAL, 2-DECISION, 4-CALIB
        etdat=[]
        t0=[0,0,0]
        calib=[]
        i=0;t=0;size=7;fr=0;ftriggers=[]
        while True:   
            words=f.readline().split()
            i+=1            
            #if i%100==0: print i
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
            if len(words)>2 and words[2]=='FRAME':
                ftriggers.append([fr,float(words[1]),float(words[4])])
                fr=int(words[3])+1
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
                etdat=[];t0=[0,0,0];ftriggers=[];fr=0
                LBLINK=False; RBLINK=False
                PHASE=0;t0[0]=float(words[1])
            if len(words)>2 and words[2]=='START':
                PHASE=1;t0[1]=float(words[1])
            if len(words)>2 and (words[2]=='DETECTION'):
                PHASE=2; t0[2]= float(words[1])
            if len(words)>2 and words[2]=='OMISSION':
                PHASE=2; t0[2]= float(words[1])
            #if len(words)>2 and words[2]=='POSTTRIAL':
            if len(words)>2 and (words[2]=='POSTTRIAL' and PHASE==2):
                etdat = _reformat(etdat,t0[0],Qexp)
                ftriggers=np.array(ftriggers)
                #print 'ftriggers.shape ',ftriggers.shape
                if ftriggers.size>0: ftriggers[:,1] -= t0[1]
                if etdat.size>0:
                    et=ETTrialData(etdat,calib,t0,
                        [vp,block,t,hz,eye],fs=ftriggers)
                    #et.extractBasicEvents(etdat)
                    data.append(et)
                
                etdat=[];t0=[0,0,0];ftriggers=[];fr=0
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
    data= _discardInvalidTrials(data)
    return data

    
def readSMI(vp,block):
    # blinks?
    # latency during the transport of messages?
    
    path = getcwd()
    path = path.rstrip('code')
    f=open(path+'smiOutput/VP%03dB%d Samples.txt'%(vp,block),'r')
    Qexp=Settings.load(Q.inputPath+'vp%03d'%vp+Q.delim+'SettingsExp.pkl' )
    ms= Qexp.monitor.getSizePix()
    try:
        line=f.readline()
        data=[]
        PHASE= -1 # 0-DCORR, 1-TRIAL, 2-DECISION, 4-CALIB
        etdat=[]
        t0=[0,0,0]
        calib=[]
        i=0;fr=0;t=0;ftriggers=[]
        while True:   
            words=f.readline().split()
            
            #if i%100==0: print i
            #if i<200: print i, words
            #if i>1300: break
            i+=1
            if len(words)>2 and words[2]=='Rate:':
                hz=float(words[3]);
            if len(words)>2 and words[2]=='Area:':
                cent=(float(words[3])/2.0,float(words[4])/2.0)
            if len(words)>5 and words[5]=='MONITORDISTANCE':
                distance=float(words[6])
            if len(words)==0: break
            # todo
##            if len(words)>2 and words[2]=='!CAL':
##                if len(words)==3: PHASE=4
##                elif words[3]=='VALIDATION' and not (words[5]=='ABORTED'):
##                    if words[6] == 'RIGHT':
##                        calib[-1].append([t,float(words[9]),float(words[11]),RIGHTEYE])
##                        PHASE= -1
##                    else:
##                        calib.append([])
##                        calib[-1].append([t,float(words[9]),float(words[11]),LEFTEYE])         
            if len(words)>5 and words[5]=='FRAME':
                ftriggers.append([fr,float(words[0])/1000.0,float(words[7])])
                fr=int(words[6])+1
            if len(words)>5 and words[5]=='TRIALID':
                eye='BOTH';t=int(words[6])
                PHASE=0;t0[0]=float(words[0])/1000.0
            if len(words)>5 and words[5]=='START':
                PHASE=1;t0[1]=float(words[0])/1000.0
            if len(words)>5 and (words[5]=='DETECTION'):
                PHASE=2; t0[2]= float(words[0])/1000.0
            #if len(words)>2 and words[2]=='POSTTRIAL':
            if len(words)>5 and (words[5]=='POSTTRIAL' or words[5]=='OMISSION'):
                etdat = _reformat(etdat,t0[0],Qexp)
                ftriggers=np.array(ftriggers)
                print ftriggers.shape
                if ftriggers.size>0: ftriggers[:,1] -= t0[1]
                if etdat.size>0:
                    et=ETTrialData(etdat,calib,t0,
                        [vp,block,t,hz,eye],fs=ftriggers)
                    data.append(et)
                
                etdat=[];t0=[0,0,0];ftriggers=[];fr=0
                calib=[];PHASE= -1
                #LBLINK=False; RBLINK=False
            if PHASE== -1 or PHASE==4: continue
            if len(words)>2 and words[1]=='SMP':
                # we check whether the data gaze position is on the screen
                if words[7]=='.': xleft=np.nan; yleft=np.nan
                else:
                    xleft=float(words[7]); yleft=float(words[8])
                    #if xleft>ms[0] or xleft<0 or yleft>ms[1] or yleft<0:
                    if yleft==0 or xleft==0:
                        xleft=np.nan; yleft=np.nan;
                if words[9]=='.': xright=np.nan; yright=np.nan
                else:
                    xright=float(words[9]); yright=float(words[10])
                    #if xright>ms[0] or xright<=0 or yright<=0 or yright>ms[1]:
                    if xright==0 or yright==0:
                        xright=np.nan; yright=np.nan;
                meas=(float(words[0])/1000.0,xleft,yleft,
                    float(words[3]),xright,yright,float(words[5]))
                etdat.append(meas)
        f.close()
    except: f.close(); raise
    data=_discardInvalidTrials(data)
    return data
  
def readTobii(vp,block,lagged=False):
    ''' function for reading the tobii controller outputs list of ETDataTrial instances

        Each trial starts with line '[time]\tTrial\t[nr]'
        and ends with line '[time]\tOmission'

        lagged - return time stamp when the data was made available (ca. 30 ms time lag)
    '''
    print 'Reading Tobii Data'
    path = getcwd()
    path = path.rstrip('/code')
    f=open(path+'/tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    Qexp=Settings.load(Q.inputPath+'vp%03d'%vp+Q.delim+'SettingsExp.pkl' )
    
    #f=open('tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    ms= Qexp.monitor.getSizePix()
    try:
        data=[];trial=[]; theta=[];t=0;msgs=[]; t0=[0,0,0];reward=[]
        on=False
        while True:
            words=f.readline()
            if len(words)==0: break
            words=words.strip('\n').strip('\r').split('\t')
            if len(words)==2: # collect header information
                if words[0]=='Recording time:':
                    recTime=words[1]; t0[0]=0; on=True
                if words[0]=='Subject: ':on=False
                if words[0]=='Recording refresh rate: ':
                    hz=float(words[1])
            elif len(words)==4 and words[2]=='Trial':
                t0[1]=trial[-1][0] # perturb
            elif len(words)==4 and words[2]=='Phase':
                phase=int(words[3])
            elif len(words)>=11 and on: # record data
                # we check whether the data gaze position is on the screen
                xleft=float(words[2]); yleft=float(words[3])
                if xleft>ms[0] or xleft<0 or yleft>ms[1] or yleft<0:
                    xleft=np.nan; yleft=np.nan;
                xright=float(words[5]); yright=float(words[6])
                if xright>ms[0] or xright<0 or yright<0 or yright>ms[1]:
                    xright=np.nan; yright=np.nan;
                if lagged: tm =float(words[0])+float(words[8]);ff=int(words[1])
                else: tm=float(words[0]);ff=int(words[1])-2
                tdata=(tm,xleft,yleft,float(words[9]),
                    xright,yright,float(words[10]),ff)
                trial.append(tdata)
            elif len(words)>2 and (words[2]=='Detection' or words[2]=='Omission'):
                # we have all data for this trial, transform to deg and append
                on=False;t0[2]=trial[-1][0]
                trial=np.array(trial)
                trial[trial==-1]=np.nan # TODO consider validity instead of coordinates
                trial=_reformat(trial,t0[0],Qexp)
                #print t0, trial.shape, trial[0,0]
                et=ETTrialData(trial[:,:-1],[],t0,
                    [vp,block,t,hz,'BOTH'],fs=[],recTime=recTime,msgs=msgs)
                fs=trial[et.ts:et.te,[-1,0]]
                fs[:,1]-=t0[1]
                for fff in range(fs.shape[0]-2,-1,-1):
                    if fs[fff+1,0]<fs[fff,0]:
                        fs[fff,0]=fs[fff+1,0]
                et.fs=np.zeros((fs[-1,0],2))
                et.fs[:,0]=range(int(fs[-1,0]))
                et.fs[:,1]=interpRange(fs[:,0],fs[:,1],et.fs[:,0])
                et.extractBasicEvents(trial[:,:-1]);
                et.phase=phase;et.reward=reward
                data.append(et)
                t+=1;trial=[];msgs=[];reward=[]
            elif on and len(words)==6:
                msgs.append([float(words[0])-t0[1],words[2]+' '+words[5]]) 
            elif on and len(words)>2:
                msgs.append([float(words[0])-t0[1],int(words[1]),words[2]])
                if words[2]=='Reward On': reward=[float(words[0])-t0[1]]
                if words[2]=='Reward Off': reward.append(float(words[0])-t0[1])
    except: f.close(); print words; raise
    f.close()
    print 'Finished Reading Data'
    return data


# some raw scripts

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



def saveSearchSacInfo():
    vp=1
    path=getcwd().rstrip('code')+'evaluation/searchSaccades/'
    si=[]
    for b in range(1,22):
        data=readEyelink(vp,b)
        for i in range(len(data)):
            if data[i].ts>=0:
                print 'block ',b,'trial',i
                data[i].extractBasicEvents()
                data[i].driftCorrection()
                data[i].importComplexEvents()
                if  data[i].search!=None:
                    g=data[i].getGaze()
                    for ev in data[i].search:
                        si.append([ev[0],g[ev[0],0],g[ev[0],7],g[ev[0],8],
                            ev[1],g[ev[1],0],g[ev[1],7],g[ev[1],8],ev[2],ev[3],
                            ev[4],data[i].t0[1]-data[i].t0[0],b,i])
    np.save(path+'SIvp%03d.npy'%vp,si)  
    
if __name__ == '__main__':
    #data=readEyelink(1,1)
    #data[1].extractBasicEvents()
    #data[1].loadTrajectories()
##    data[1].driftCorrection()
##    data[1].importTrackingFromCoder()
##    print data[1].sev
##    print data[1].search

    saveSearchSacInfo()
    
            
##    res=np.zeros(2549)
##    for b in range(1,22):
##        data=readEyelink(1,b)
##        for i in range(40):
##            data[i].extractBasicEvents()
##            data[i].loadTrajectories()
##            D=data[i].oldtraj
##            print i, D.shape
##            for a in range(14):
##                res+=np.logical_or(np.abs(np.diff(D[:,a,1],2))>0.0001,
##                        np.abs(np.diff(D[:,a,0],2))>0.0001)
##            

        
##    data[1].extractBasicEvents()
##    data[1].driftCorrection()
##    data[1].importComplexEvents()
