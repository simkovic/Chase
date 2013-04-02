import numpy as np
from Settings import Q
from os import getcwd
from evalETdata import ETTrialData

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
    if type(trial) is type( () ): print 'error in readEdf'
    trial[:,0]-=tstart
    trial[:,1]=Qexp.pix2deg(trial[:,1])
    trial[:,2]=-Qexp.pix2deg(trial[:,2])
    if trial.shape[1]==7:
        trial[:,4]=Q.exp.pix2deg(trial[:,4])
        trial[:,5]=-Q.exp.pix2deg(trial[:,5])
    return trial

def readEdf(vp,block):
    cent=(0,0)
    path = getcwd()
    path = path.rstrip('code')
    try:
        f=open(path+'eyelinkOutput/VP%03dB%d.asc'%(vp,block),'r')
    except:
        f=open(path+'eyelinkOutput/VP%03dB%d.ASC'%(vp,block),'r')
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
                etdat = _reformat(etdat,cent,t0[0],distance)
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
    data= _discardInvalidTrials(data)
    return data

def readSMI(vp,block):
    # blinks?
    # latency during the transport of messages?
    
    path = getcwd()
    path = path.rstrip('code')
    f=open(path+'smiOutput/VP%03dB%d Samples.txt'%(vp,block),'r')
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
                etdat = _reformat(etdat,cent,t0[0],distance)
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
                    #if xleft>cent[0]*2 or xleft<0 or yleft>cent[1]*2 or yleft<0:
                    if yleft==0 or xleft==0:
                        xleft=np.nan; yleft=np.nan;
                if words[9]=='.': xright=np.nan; yright=np.nan
                else:
                    xright=float(words[9]); yright=float(words[10])
                    #if xright>cent[0]*2 or xright<=0 or yright<=0 or yright>cent[1]*2:
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
    path = getcwd()
    path = path.rstrip('/code')
    f=open(path+'/tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    Qexp=Q.loadSettings(vp)
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
                trial[trial==-1]=np.nan # TODO consider validity instead of coordinates
                trial=_reformat(trial,tstart,Qexp)
                et=ETTrialData(trial[:,:-1],[],[trial[0,0],tstart,trial[-1,0]],
                    [vp,block,t,hz,'BOTH'],fs=trial[:,[0,-1]],recTime=recTime,msgs=msgs)
                data.append(et)
                t+=1;trial=[];msgs=[]
            elif len(words)==6: msgs.append([float(words[0])-tstart,words[2]+' '+words[5]])
            elif words[2]!='Phase': msgs.append([float(words[0])-tstart,int(words[1]),words[2]])
    except: f.close(); raise
    f.close()
    return data
data=readTobii(150,0)

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
def eyelinkScript():
    path='/home/matus/Desktop/pylink/evaluation/sacTargets/'
    for b in range(0,25):
        data=readEdf(18,b)
        sactot=0
        #evs=[]
        for i in range(len(data)):
            if data[i].ts>=0:
                print i
                data[i].driftCorrection()
                for ev in data[i].sev: sactot+=1# int(ev[0]-50>=0 and ev[0]+50<data[i].traj.shape[0])
                #data[i].extractTracking()
                #data[i].exportEvents()
                #data[i].plotEvents()
                #data[i].exportTracking()
                #evs.extend(data[i].track)
                #data[i].plotTracking()
        
        sevall=[]
        D=np.zeros((sactot,200,15,2))*np.nan
        k=0
        print 'sactot=',sactot
        
        for i in range(len(data)):
            if data[i].ts>=0:
                g=data[i].getGaze()
                for ev in data[i].sev:
                    si=max(ev[0]-100,0)
                    ei=min(ev[0]+100,data[i].traj.shape[0]-1)
                    ssi= si-ev[0]+100
                    eei= 100+ei -ev[0]
                    #print si,ei,ssi, eei
                    D[k,ssi:eei,:14,:]=data[i].traj[si:ei,:,:]
                    D[k,ssi:eei,-1,:]=g[si:ei,[7,8]]
                    k+=1
                    sevall.append([[ev[0],g[ev[0],0],g[ev[0],7],g[ev[0],8]],
                                    [ev[1],g[ev[1],0],g[ev[1],7],g[ev[1],8]]])
                    #if ev[0]-50<0 or ev[0]+50>=data[i].traj.shape[0]:stop
                    #else: print 'warn ', i
        np.save(path+'vp%03db%d.npy'%(data[0].vp,data[0].block),D)
        np.save(path+'SIvp%03db%d.npy'%(data[0].vp,data[0].block),sevall)                    
           


