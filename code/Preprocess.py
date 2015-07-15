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
from Settings import *
import os,pickle
from ETData import ETData, interpRange
plt.ion()
##########################################################
# helper functions
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
##########################################################
# read eyetracking data
def readEyelink(vp,block):
    ''' reads Eyelink text file (.asc) of subject VP and block BLOCK
        the file should be at /eyelinkOutput/VP<VP>B<BLOCK>.asc
        requires the corresponding input files on the input path
        vp - subject id
        block - experiment block
        returns ETData instance
    '''
    cent=(0,0)
    path = os.getcwd()
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
        while True: # read all lines  
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
                PHASE=2; t0[2]= float(words[1]);msg=words[2]
            if len(words)>2 and words[2]=='OMISSION':
                PHASE=2; t0[2]= float(words[1]);msg=words[2]
            #if len(words)>2 and words[2]=='POSTTRIAL':
            if len(words)>2 and (words[2]=='POSTTRIAL' and PHASE==2):
                etdat = _reformat(etdat,t0[0],Qexp)
                ftriggers=np.array(ftriggers)
                #print 'ftriggers.shape ',ftriggers.shape
                if ftriggers.size>0: ftriggers[:,1] -= t0[1]
                if etdat.size>0:
                    et=ETData(etdat,calib,t0,
                        [vp,block,t,hz,eye],fs=ftriggers,msgs=msg)
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
            except:pass
            #if len(trial)>0:print len(trial[0])
            #if len(dcorr)>1: print dcorr[-1]
        f.close()
    except: f.close(); raise
    data= _discardInvalidTrials(data)
    return data

    
def readSMI(vp,block):
    ''' reads SMI text file (.txt) of subject VP and block BLOCK
        the file should be at /smiOutput/VP<VP>B<BLOCK>.asc
        requires the corresponding input files on the input path
        vp - subject id
        block - experiment block
        returns ETData instance
        NOTE: this code is experimental
        TODO blinks?, latency during the transport of messages?
    '''
    path = os.getcwd()
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
                    et=ETData(etdat,calib,t0,
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
  
def readTobii(vp,block,path,lagged=False,verbose=False):
    '''
        reads Tobii controller output of subject VP and block BLOCK
        the file should be at <PATH>/VP<VP>B<BLOCK>.csv
        requires the corresponding input files on the input path
        vp - subject id
        block - experiment block
        path - path to the eyetracking data
        lagged - log time stamp when the data was made available
            (ca. 30 ms time lag), useful for replay
        verbose - print info
        returns ETData instance
        
        Each trial starts with line '[time]\tTrial\t[nr]'
        and ends with line '[time]\tOmission'  
    '''
    from Settings import Qexp
    if verbose: print 'Reading Tobii Data'
    #path = os.getcwd()
    #path = path.rstrip('code')
    f=open(path+'VP%03dB%d.csv'%(vp,block),'r')
    #Qexp=Settings.load(Q.inputPath+'vp%03d'%vp+Q.delim+'SettingsExp.pkl' )
           
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
                et=ETData(trial[:,:-1],[],t0,
                    [vp,block,t,hz,'BOTH'],fs=np.array([np.nan,np.nan]),recTime=recTime,msgs=msgs)
                fs=trial[et.ts:et.te,[-1,0]]
                fs[:,1]-=t0[1]
                for fff in range(fs.shape[0]-2,-1,-1):
                    if fs[fff+1,0]<fs[fff,0]:
                        fs[fff,0]=fs[fff+1,0]
                et.fs=np.zeros((fs[-1,0],3))
                et.fs[:,0]=range(int(fs[-1,0]))
                et.fs[:,1]=interpRange(fs[:,0],fs[:,1],et.fs[:,0])
                et.fs[:,2]=interpRange(fs[:,1],range(fs.shape[0]),et.fs[:,1])
                et.fs[:,2]=np.round(et.fs[:,2])
                for msg in et.msgs:
                    if msg[2]=='Omission': msg[0]=float(msg[0]);msg[1]=float(msg[1])
                    if msg[2]=='Drift Correction':
                        msg[1]=int(round(msg[0]*75/1000.))
                        msg.append(msg[0]*et.hz/1000.)
                    elif msg[1]-et.fs.shape[0]>0:
                        val=(msg[1]-et.fs.shape[0])*75/et.hz+et.fs[-1,2]
                        msg.append(int(round(val)))
                    else: msg.append(int(et.fs[int(msg[1]),2]))
                #et.extractBasicEvents(trial[:,:-1]);
                et.phase=phase;et.reward=reward
                data.append(et)
                t+=1;trial=[];msgs=[];reward=[]
            elif on and len(words)==6:
                msgs.append([float(words[0])-t0[1],words[2]+' '+words[5]]) 
            elif on and len(words)>2:
                msgs.append([float(words[0])-t0[1],int(words[1]),words[2]])
                if words[2]=='Reward On': reward=[float(words[0])-t0[1]]
                if words[2]=='Reward Off': reward.append(float(words[0])-t0[1])
    except: f.close(); print 'Words: ',words; raise
    f.close()
    if verbose:print 'Finished Reading Data'
    return data

#######################################
def saveETinfo(vp=1,coderid=4,suf=''):
    ''' extract and save eyetracking information for template analyses
        vp - subject id
        output: for each subect saves following files
            ti.npy, si.npy both npy files are  2 dimensional ndarray
            with a row giving info on a single sample from the
            template analysis
            the columns of each array give 
        si.npy: saccade information
            0- sac onset in frames,
            1- sac onset in sec,
            2- sac onset posx degrees,
            3- sac onset posy degrees,
            4- sac end in frames,
            5- sac end in sec,
            6- sac end posx degrees,
            7- sac end posy degrees,
            8- sac speed,
            9- sac duration,
            10-event type of the consecutive event,
            11-start of tracking event in frames,
            12-trial duration in sec,
            13-sac order positon within pursuit event counted backwards
            14-sac order position within pursuit event,
            15-block id,
            16-trial id
        ti.npy: smooth eye movement episodes information
            0 - sac order positon within pursuit event counted backwards
            1 - sac order position within pursuit event,
            2 - block id
            3 - trial id
            4 - episode onset in frames
            5 - episode end in frames
        sacxy.pickle, trackxy.pickle and purxy.pickle are nested lists
            that give the gaze coordinates during saccades, smooth eye
            movement episodes and catch-up saccades 
            
        blinks are not included as saccades
        may throw some runtime warnings due to nans in the gaze data 
    '''
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    si=[];sacxy=[];finalev=[];ti=[];trackxy=[]
    pi=[];purxy=[];
    for b in range(1,24):
        try:data=readEyelink(vp,b)
        except IOError:
            print 'block %d is missing'%b
            continue
        for i in range(0,len(data)):
            if data[i].ts>=0:
                print 'vp',vp,'block ',b,'trial',i
                data[i].extractBasicEvents()
                data[i].driftCorrection()
                data[i].importComplexEvents(coderid=coderid)
                if  data[i].search!=None:
                    g=data[i].getGaze()
                    for ev in data[i].search:
                        si.append([ev[0],g[ev[0],0],g[ev[0],7],g[ev[0],8],
                            ev[1],g[ev[1],0],g[ev[1],7],g[ev[1],8],ev[2],ev[3],
                            ev[4],np.nan,data[i].t0[1]-data[i].t0[0],0,0,b,i])
                        sacxy.append(g[ev[0]:ev[1],[7,8]].flatten().tolist())
                    gg=0
                    for tr in data[i].track:
                        if False:#len(tr[4:])>0:
                            # neg difference indicates missing initial saccades,
                            # i.e. tracking started with a blink or directly with pursuit 
                            print '\tdif', g[tr[0],0],g[tr[4][1],0],tr[0]-tr[4][1]
                        kk=1
                        for ev in tr[4]:
                            si.append([ev[0],g[ev[0],0],g[ev[0],7],g[ev[0],8],
                                ev[1],g[ev[1],0],g[ev[1],7],g[ev[1],8],ev[2],ev[3],
                                ev[4],tr[0],data[i].t0[1]-data[i].t0[0],
                                kk-len(tr[4])-1,kk,b,i]);
                            sacxy.append(g[ev[0]:ev[1],[7,8]].flatten().tolist())
                            if len(ti) and len(ti[-1])==5:
                                ti[-1].extend([ev[0]])
                                gxy=g[max(0,ti[-1][4]-5):min(ti[-1][5]+5,g.shape[0]-1),[0,7,8]]
                                trackxy.append([tr[2],tr[3],gxy.flatten().tolist()])
                            ti.append(si[-1][13:]+[ev[1]])
                            if len(ev)>5:
                                pi.append([b,i,kk-len(tr[4])-1,kk,
                                           ev[-1][0],g[ev[-1][0],0],ev[-1][1],g[ev[-1][1],0]])
                                purxy.append(g[ev[-1][0]:ev[-1][1],[7,8]].flatten().tolist())
                            kk+=1
                        if len(tr[4])==0:
                            mll=min(g.shape[0]-1,tr[1])
                            ti.append([-1,1,b,i,tr[0],mll])
                        else: ti[-1].append(min(g.shape[0]-1,tr[1]))
                        assert len(ti[-1])==6
                        gxy=g[max(0,ti[-1][4]-5):min(ti[-1][5]+5,g.shape[0]-1),[0,7,8]]
                        trackxy.append([tr[2],tr[3],gxy.flatten().tolist()])
                        gg+=1
                    if data[i].msgs=='DETECTION':
                        si[-1][13]=1;ti[-1][0]=1
                        res=[[b,i,0]]
                        for ttt in [-251,-201,-151,-101,-51,-26]:
                            if data[i].isFix[ttt] or data[i].opur[ttt] or data[i].cpur[ttt]:
                                res.append(g[ttt,[0,7,8]])
                            else: res.append(np.zeros(3)*np.nan)
                        finalev.append(res)  
                    elif data[i].msgs!='OMISSION':raise ValueError('Unexpected message')
    np.save(path+'ti%s.npy'%suf,ti)
    f=open(path+'trackxy%s.pickle'%suf,'wb')
    pickle.dump(trackxy,f);f.close()
    np.save(path+'finalev%s'%suf,finalev)
    f=open(path+'purxy%s.pickle'%suf,'wb')
    pickle.dump(purxy,f);f.close()
    f=open(path+'sacxy%s.pickle'%suf,'wb')
    pickle.dump(sacxy,f);f.close()
    np.save(path+'si%s.npy'%suf,si)
    np.save(path+'pi%s.npy'%suf,pi)
    
    
if __name__ == '__main__':
    import sys
    saveTrackedAgs(vp=2,coderid=4)
    bla
    vp=int(sys.argv[1])
    coder=int(sys.argv[2])
    saveETinfo(vp=vp,coderid=coder,suf='coder%d'%coder)

            
    #print np.int32(np.isnan(data[0].gaze[:1000,1]))
    #data[0].extractBasicEvents()
    #for dat in data: dat.extractBasicEvents()
