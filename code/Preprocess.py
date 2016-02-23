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

import warnings
warnings.filterwarnings("ignore")
from Settings import *
import os,pickle
from ETData import ETData, interpRange
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
    if type(trial) is type( () ): print 'error in _reformat'
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


#######################################
def savePredictors(data):
    ''' compute velocity quantiles for each inter-saccade event
    '''
    import numpy as np
    try:
        #velquant=[[data.vp,data.block,data.trial]+(len(perc)-2)*[0]];
        preds=[]
        data.extractBasicEvents()
        data.driftCorrection()
        importFailed=data.importComplexEvents(coderid=4)
        if importFailed: return [[-1]]
        g=data.getGaze()
        vel=data.getVelocity()
        for si in range(len(data.sev)):
            preds.append([data.vp,data.trial,data.sev[si][-1]])
            if si+2<len(data.sev): e=data.sev[si+1][0]
            else: e=-1
            s=data.sev[si][1];d=e-s
            tps=[s,s+d/4.,s+d/2.,s+3*d/4.,e]
            tps=np.int32(np.round(tps))
            for ti in range(len(tps)-1):
                preds[-1].append(np.nanmedian(vel[tps[ti]:tps[ti+1]]))
                dist=np.nanmedian(data.dist[tps[ti]:tps[ti+1],:],0)
                di=np.argsort(dist)[:4]#take four nearest agents
                preds[-1].extend(dist[di])
                dev=np.abs(data.dev[tps[ti]:tps[ti+1],:])
                dev=np.nanmedian(dev[:,di],0)
                preds[-1].extend(dev)
        return preds   
    except:
        print 'Error at vp %d b %d t %d'%(data.vp,data.block,data.trial)
        raise   
def savePredictorsScript(suf=''):
    from multiprocessing import Pool
    pool=Pool(8)
    vpn=[1,2,3,4];
    blocks=range(1,24);
    data=[]
    for vp in vpn:
        for b in blocks:
            try:
                dt=readEyelink(vp,b)
                for i in range(0,len(dt)):
                    if dt[i].ts>=0: data.append(dt[i])
            except IOError:
                print 'vp %d, block %d is missing'%(vp,b)
                continue
    print 'savePredictorsScript:Start'
    #res=dview.map_sync(savePredictors,data)
    res=pool.map(savePredictors,data)
    print 'savePredictorsScript:Finished'
    for vp in vpn:
        path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
        out=filter(lambda x: x[0][0]==vp,res)
        f=open(path+'preds%s.pickle'%suf,'wb')
        pickle.dump(out,f);f.close()             
    
if __name__ == '__main__':
    #import sys
    #vp=int(sys.argv[1])
    #coder=int(sys.argv[2])
    #saveETinfo(vp=vp,coderid=coder,suf='coder%d'%coder)
    savePredictorsScript(suf='MA')
