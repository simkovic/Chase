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
from psychopy import visual,core
from psychopy.misc import deg2pix
from Constants import *
from Settings import Q
import random, Image,ImageFilter, os,pyglet, pickle,commands
from scipy.ndimage.filters import convolve,gaussian_filter
from ImageOps import grayscale
from time import time, sleep
from multiprocessing import Process,Pool
import os as oss

# each function that performs data analysis loads the data from hard drive
# and saves its results to the hard drive
# most functions feature arguments vp, event and suf which indicate:
# vp - subject id
# event - which samples were analyzed, samples are locked either at the onset of
#   0: exploration saccade, 1: 1st catch-up saccade, 2: 2nd catch-up saccade etc.
#   or
#   95-99: sample 500,400,300,200,100 and 50 ms before the button press respectively
# suf - output file suffix

def initPath(vpp,eventt):
    #global event,vp,path,inpath,figpath
    event=eventt;vp=vpp
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    if event>=0: inpath=path+'E%d/'%event
    else: inpath=path+'E%d/'%(100+event)
    figpath=os.getcwd().rstrip('code')+'figures/Pixel/'
    return path,inpath,figpath
#########################################################
# functions for Translating Coordinates to Images
def position2image(positions,elem=None,wind=None):
    '''transforms vector of agent positions to display snapshot
        output format is HxW matrix of light intensity values (uint8)
    '''
    if type(wind)==type(None):
        close=True; wind=Q.initDisplay()
    else: close=False
    if type(elem)==type(None):
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=positions.shape[0], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    try:
        elem.setXYs(positions)      
        elem.draw()    
        wind.getMovieFrame(buffer='back')
        ret=wind.movieFrames[0]
        wind.movieFrames=[]
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT | pyglet.gl.GL_DEPTH_BUFFER_BIT)
        wind._defDepth=0.0
        if close: wind.close()
        return grayscale(ret)# make grey, convert to npy
    except:
        if close: wind.close()
        raise

def traj2movie(traj,width=5,outsize=64,elem=None,wind=None,rot=2,
               hz=85.0,SX=0.3,SY=0.3,ST=20):
    ''' extracts window at position 0,0 of width WIDTH deg
        from trajectories and subsamples to OUTSIZExOUTSIZE pixels
        HZ - trajectory sampling frequency
        ROT - int number of rotations to output or float angle in radians 
        SX,SY,ST - standard deviation of gaussian filter in deg,deg,ms
        
    '''
    if type(wind)==type(None):
        close=True; wind=Q.initDisplay()
    else: close=False
    if type(elem)==type(None):
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=traj.shape[1], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    try:
        sig=[ST/1000.0*hz]
        sig.append(deg2pix(SX,wind.monitor))
        sig.append(deg2pix(SY,wind.monitor))
        w=int(np.round(deg2pix(width,wind.monitor)/2.0))
        D=np.zeros((traj.shape[0],outsize,outsize,rot),dtype=np.uint8)
        Ims=[]
        for f in range(0,traj.shape[0]):
            Im=position2image(traj[f,:,:],wind=wind)
            cx=int(Im.size[0]/2.0);cy=int(Im.size[1]/2.0)
            Im=Im.crop(np.int32((cx-1.5*w,cy-1.5*w,cx+1.5*w,cy+1.5*w)))
            Im=np.asarray(Im,dtype=np.float32)
            Ims.append(Im)
        Ims=np.array(Ims)
        if np.any(np.array(sig)!=0):Ims=gaussian_filter(Ims,sig)
        if np.any(Ims>255): print 'warning, too large'
        if np.any(Ims<0): print 'warning, too small'
        Ims=np.uint8(np.round(Ims))
        for f in range(Ims.shape[0]):
            Im=Image.fromarray(np.array(Ims[f,:,:]))
            bb=int(Im.size[0]/2.0)
            I=Im.crop((bb-w,bb-w,bb+w,bb+w))
            I=np.asarray(I.resize((outsize,outsize),Image.ANTIALIAS))
            D[f,:,:,0]=I
            for r in range(1,rot):
                I2=Im.rotate(90/float(rot)*r)
                I2=I2.crop((bb-w,bb-w,bb+w,bb+w))
                I2=np.asarray(I2.resize((outsize,outsize),Image.ANTIALIAS))
                D[f,:,:,r]=I2
        if close: wind.close()
        return D
    except:
        if close: wind.close()
        raise

def traj2avi(traj,fn='test'):
    ''' outputs trajectory as avi
        traj - trajectory matrix (rows-frames,cols-agents)
        fn - filename'''
    for f in range(traj.shape[0]):
        plt.plot(traj[f,:,0],traj[f,:,1],'o')
        plt.xlim([-5,5]);plt.ylim([-5,5])
        ax=plt.gca();
        ax.set_aspect(1); 
        ax.set_xticks([])
        ax.set_yticks([]);
        plt.savefig('fig%03d.png'%f,bbox_inches='tight')
        plt.cla()
    commands.getstatusoutput('ffmpeg -i fig%03d.png -r 50 -y '+fn+'.avi')
    commands.getstatusoutput('rm fig***.png');

def pf2avi(pf,fn='test'):
    ''' outputs template movie as a avi
        pf - template movie,
        fn - ouput file name'''
    for f in range(pf.shape[2]):
        plt.imshow(pf[:,:,f],vmax=255,vmin=0)
        plt.grid()
        ax=plt.gca();
        ax.set_aspect(1); 
        ax.set_xticks([])
        ax.set_yticks([]);
        plt.savefig('fig%03d.png'%f,bbox_inches='tight')
        plt.cla()
    commands.getstatusoutput('ffmpeg -i fig%03d.png -r 50 -y '+fn+'.avi')
    commands.getstatusoutput('rm fig***.png');
########################################3
# routines that translate trajectories to movie representation
# which is used to compute template movies
# this code does parallel computation in a clumsy way
# as I wrote it before I figured out how python's multiprocessing module works
def PFextract(E,part=[0,1],wind=None,elem=None,inpath='',suf=''):
    """
        part[0] - current part
        part[1] - total number of parts in the paralel computation
        wind - psychopy window, elem - psychopy ElementArrayStim
        inpath - input path
        suf - output name suffix
    """
    f=open(inpath+'PF%s.pars'%suf,'r');dat=pickle.load(f);f.close()
    inc=E.shape[0]/part[1]
    start=part[0]*inc
    ende=min((part[0]+1)*inc,E.shape[0])
    print start,ende,E.shape
    os=dat['os'];rot=dat['rot']

    phis=np.load(inpath+'phi%s.npy'%suf)
    D=np.zeros((ende-start,E.shape[1],os,os,rot),dtype=np.uint8)
    try:
        if type(wind)==type(None):
            close=True; wind=Q.initDisplay()
        else: close=False
        if elem==None:
            elem=visual.ElementArrayStim(wind,fieldShape='sqr',
                nElements=E.shape[1], sizes=Q.agentSize,
                elementMask=RING,elementTex=None,colors='white')
        for i in range(ende-start):
            phi=phis[i+start]# rotate clockwise by phi
            R=np.array([[np.cos(phi),np.sin(phi)],
                        [-np.sin(phi),np.cos(phi)]])
            temp=np.copy(E[i+start,:,:,:])
            for a in range(14):temp[:,a,:]=R.dot(temp[:,a,:].T).T
            D[i,:,:,:,:]=traj2movie(temp,outsize=os,
                elem=elem,wind=wind,rot=rot,width=dat['width'],
                hz=dat['hz'],SX=dat['SX'],SY=dat['SY'],ST=dat['ST'])
            #from matustools.matusplotlib import ndarray2gif
            #outt=np.float32(D[i,:,:,:,0].T)
            #outt-= np.min(outt)
            #outt/= np.max(outt)
            #ndarray2gif('test%d'%i,outt)
            #if i==3: bla
        if close: wind.close()
        PF=np.rollaxis(D,1,5)
        if not oss.path.exists(inpath+'PF%s/'%suf):
            oss.makedirs(inpath+'PF%s/'%suf)
        if len(part)==2: np.save(inpath+'PF%s/PF%03d.npy'%(suf,part[0]),PF)
        else: np.save('PF.npy',PF)
    except:
        if close: wind.close()
        raise

def PFinit(vp,event,suf=''):
    ''' suf - output name suffix'''
    path,inpath,fp=initPath(vp,event)
    if event>=0: N=[50,15,8,2][event]
    else: N=1
    dat={'N':N,'os':64,'rot':1,
         'width':10,'hz':85.0,'SX':0.3,'SY':0.3,'ST':40}
    np.save(inpath+'stackPF.npy',range(dat['N']+1))
    Q.save(inpath+'PF%s.q'%suf)
    f=open(inpath+'PF%s.pars'%suf,'w')
    pickle.dump(dat,f)
    f.close()
    
def PFparallel(vp,event,suf=''):
    ''' please run PFinit() first
        suf - output name suffix
    '''
    path,inpath,fp=initPath(vp,event)
    E=np.load(inpath+'DG%s.npy'%suf)[:,:,:,:2]
    print E.shape
    stack=np.load(inpath+'stackPF.npy').tolist()
    f=open(inpath+'PF%s.pars'%suf,'r');dat=pickle.load(f);f.close()
    N=dat['N']
    wind=Q.initDisplay()
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=E.shape[1], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    while len(stack):
        jobid=stack.pop(0)
        np.save(inpath+'stackPF.npy',stack)
        PFextract(E,[jobid,N],wind=wind, elem=elem,inpath=inpath,suf=suf)
        loaded=False
        while not loaded:
            try:
                stack=np.load(inpath+'stackPF.npy').tolist()
                loaded=True
            except IOError:
                print 'IOError'
                core.wait(1)
    wind.close()

#########################################################
# support vector machine classification

def createMask(P,F):
    # create mask with circular aperture
    mid=(P-1)/2.0
    mask=np.zeros((P,P,F),dtype=np.bool8)
    for i in range(P):
        for j in range(P):
            if np.sqrt((i-mid)**2+(j-mid)**2)<=P/2.0: mask[i,j,:]=True
    return mask 
######################################
# svm data preparation
def pfExport(vp,evA,evB,suf=''):
    ''' compute similarity matrix between perceptive fields
        vp - subject id
        evA - id of event A
        evB - id of event B
    '''
    strid ='pfExport, vp=%d, evA=%d, evB=%d: '%(vp,evA,evB)
    print strid+'started'
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    inpa=path+'E%d/'%evA
    inpb=path+'E%d/'%evB
    f=open(inpa+'PF%s.pars'%suf,'r');
    dat=pickle.load(f);f.close();N1=dat['N']+1
    f=open(inpb+'PF%s.pars'%suf,'r');
    dat=pickle.load(f);f.close();N2=dat['N']+1
    D1=np.load(inpa+'PF%s/PF000.npy'%(suf))
    D2=np.load(inpb+'PF%s/PF000.npy'%(suf))
    ds1=D1.shape[0];ds2=D2.shape[0];
    assert dat['os']==D1.shape[1]
    P=D1.shape[1];F=D1.shape[4]
    dga=np.load(inpa+'DG%s.npy'%suf).shape[0]
    dgb=np.load(inpb+'DG%s.npy'%suf).shape[0]
    evsuf=['ev%d'%evB,''][int(evA==evB)]
    #print dga,dgb,ds1,ds2
    S=np.zeros([dga,dgb])*np.nan
    mask=createMask(P,F) 
    # compute similarity
    for pf1 in range(0,N1):
        for pf2 in range(pf1*int(evA==evB),N2):
            D1=np.load(inpa+'PF%s/PF%03d.npy'%(suf,pf1))
            D2=np.load(inpb+'PF%s/PF%03d.npy'%(suf,pf2))
            Spart=np.zeros((D1.shape[0],D2.shape[0]))*np.nan
            for n1 in range(D1.shape[0]):
                for n2 in range(D2.shape[0]):
                    a=np.float32(D1[n1,:,:,0,:])
                    b=np.float32(D2[n2,:,:,0,:])
                    Spart[n1,n2]=(np.square(a-b)*mask).sum()
            Spart=np.sqrt(Spart);sps=Spart.shape
            S[pf1*ds1:(pf1*ds1+sps[0]),pf2*ds2:(pf2*ds2+sps[1])]=Spart
            if evA==evB: S[pf2*ds2:(pf2*ds2+sps[1]),pf1*ds1:(pf1*ds1+sps[0])]=Spart.T
        print strid + 'pf1=%d'%pf1
    assert np.all(~np.isnan(S))
    np.save(inpa+'S'+suf+evsuf,S)
    print strid+'finished'
    
def pfSubsample(vp,ev,s=2,suf=''):
    ''' subsamples the output of PFextract() for svm classification
        s - multiplicative subsampling factor'''
    strid='pfSubsample vp=%d, ev=%d,s=%d: '%(vp,ev,s)
    print strid+'started'
    path,inpath,figpath=initPath(vp,ev)
    f=open(inpath+'PF%s.pars'%suf,'r');
    dat=pickle.load(f);f.close();N=dat['N']+1
    out=[]
    for h in range(0,N):
        D=np.load(inpath+'PF%s/PF%03d.npy'%(suf,h))
        P=D.shape[1]; F=D.shape[4]
        pfnew=np.zeros([D.shape[0],P/s,P/s,F/s])*np.nan
        for n in range(D.shape[0]):
            pf=D[n,:,:,0,:]
            for i in range(pfnew.shape[1]):
                for j in range(pfnew.shape[2]):
                    for f in range(pfnew.shape[3]):
                        temp=pf[i*s:(i+1)*s,j*s:(j+1)*s,f*s:(f+1)*s].mean()
                        pfnew[n,i,j,f]=temp
        out.append(pfnew)
        print strid+'h=%d finished'%h
    out=np.concatenate(out,axis=0)
    np.save(inpath+'sPF%s%d.npy'%(suf,s),out)
    print strid+'finished'
    
def exportScript(suf=''):
    '''script does data preprocessing'''
    pool=Pool(processes=8)
    vps=[1,2,3,4];
    for ags in [[0,0],[0,1],[1,1]]:
        for vp in vps:
            pool.apply_async(pfExport,[vp]+ags+[suf])       
    for ev in [0,1]:
        for vp in vps:
            pool.apply_async(pfSubsample,[vp,ev,2,suf])   
    pool.close()
    pool.join()

##################################################
# train SVM
def SsimilSvm(vp,ev,beta,fn,suf=''):
    ''' computes and saves similarity matrix
        beta - value of beta parameter on log scale
        fn - output file name'''
    strid='SexportSvm vp=%d, ev=%d,beta=%.1f: '%(vp,ev,beta)
    #print strid+'started'
    e1=ev; e2=ev+1
    pth=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    S1=np.load(pth+'E%d/S%s.npy'%(e1,suf))
    S2=np.load(pth+'E%d/S%s.npy'%(e2,suf))
    Scross=np.load(pth+'E%d/S%sev%d.npy'%(e1,suf,e2))
    n1=S1.shape[0];n2=S2.shape[0]
    S=np.zeros((n1+n2,n1+n2))*np.nan
    S[:n1,:n1]=S1
    S[:n1,n1:]=Scross
    S[n1:,:n1]=Scross.T
    S[n1:,n1:]=S2
    del S1,S2,Scross
    # use radial basis function, note values in S are already squared
    S=np.exp(-np.exp(beta)*S/5000.)
    f=open(pth+'E%d/svm%s/svm%d.in'%(e1,suf,int(beta*10)),'w')
    for row in range(n1+n2):
        s='%d 0:%d'%(int(row<n1,row+1)
        for col in range(n1+n2):
            s+=' %d:%.4f'%(col+1,S[row,col])
        s+='\n'
        f.write(s)
        f.flush()
    f.close()
    print strid+'finished'

def SevalSvm(vp,ev,b,fn,suf):
    '''calls libsvm to train the svm
        libsvm 3.12 was used'''
    strid='SevalSvm vp=%d, ev=%d,beta=%.1f: '%(vp,ev,b)
    print strid+'started'
    cs=np.arange(-10,10,0.5) # range of C values
    SsimilSvm(vp,ev,b,fn,suf)
    fn= fn+'%d'%int(b*10)
    logf=open(fn+'.log','w')
    for c in cs:
        status,output=commands.getstatusoutput('svm-train -s '+
            '0 -v 5 -t 4 -c %f -m 6000 %s.in'%(np.exp(c),fn))
        if status:
            print output
            raise RuntimeError(output)
        logf.write('b=%.1f\nC=%.1f\n'%(b,c))
        logf.write(output+'\n')
        logf.flush()
        #temp=float(output.rsplit(' ')[-1].rstrip('%'))/100.
    logf.close()
    print strid+'finished'
    
def gridSearchScript(suf=''):
    ''' script that performs training'''
    pool=Pool(processes=8)
    vps=[1,2,3,4]# subject ids
    betas=np.arange(-5,10,0.5)# range of beta values
    for ev in [0]:
        for vp in vps:
            path,inpath,figpath=initPath(vp,ev)
            fn=inpath+'svm%s/svm'%suf
            for beta in betas:
                pool.apply_async(SevalSvm,[vp,ev,beta,fn,suf])
                if ev==0: sleep(20)
    pool.close()
    pool.join()
############################################3
# analyzing and validating the svm solution
def getWeights(vp,event,suf):
    ''' returns the weights of the svm solution'''
    # validate the model and save it
    path,inpath,fp=initPath(vp,event)
    opt=np.load(inpath+'svm%s/opt.npy'%suf)
    fnm=inpath+'svm%s/svm'%suf;fn=fnm+'%d'%int(opt[0]*10)
    if not os.path.isfile(fn+'.in'):SexportSvm(vp,event,opt[0],fn,suf)
    status,output=commands.getstatusoutput('svm-train -s '+
        '0 -t 4 -c %f %s.in %s.model'%(np.exp(opt[1]),fn,fnm))
    if status: print output
    # save support vector indices to npy file for later use
    f=open(fnm+'.model','r')
    svs=[]
    weights=[]
    svon=False
    k=0
    for line in f.readlines():
        words=line.rstrip('\n')
        if words == 'SV':
            svon=True
            continue
        words=words.rsplit(' ')
        if words[0] == 'rho': weights.append(float(words[1]))
        if svon:
            weights.append(float(words[0]))
            words=words[1].rsplit(':')[1]
            svs.append(int(words))
    f.close()
    weights=np.array(weights)
    svs=np.array(svs)-1
    np.save(inpath+'svm%s/weights.npy'%suf,weights)
    np.save(inpath+'svm%s/svs.npy'%suf,svs)
    info=[]
    sPF=np.load(inpath+'sPF%s2.npy'%suf)
    info.append(sPF.shape[0])
    sPF=np.load(path+'E%d/sPF%s2.npy'%(event+1,suf))
    info.append(sPF.shape[0])
    del sPF
    info.append(svs.size)
    return np.array(info)

def plotSvm(event=0,suf=''):
    ''' plots the grid search results '''
    print 'plotSvm started'
    plt.figure()
    infos=[]
    for vp in [1,2,3,4]:
        path,inpath,figpath=initPath(vp,event)
        fns=os.listdir(inpath+'svm%s/'%suf)
        fns=filter(lambda s: s.endswith('.log'),fns)
        dat=[]
        for fn in fns:
            f=open(inpath+'svm%s/'%suf+fn,'r')
            txt=f.read()
            f.close()
            txt='\n'+txt
            txt=txt.rsplit('%')
            for tx in txt[:-1]:
                lines=tx.rsplit('\n')
                #print len(lines),len(lines[0]),lines[0]
                b= float(lines[1].rsplit('=')[1])
                C= float(lines[2].rsplit('=')[1])
                f= float(lines[-1].rsplit('=')[1])/100.
                dat.append([b,C,f])
           
        dat=np.array(dat)
        betas=np.unique(dat[:,0]).tolist()
        Cs= np.unique(dat[:,1]).tolist()
        fun=np.zeros((len(betas),len(Cs)))#*np.nan
        for d in dat.tolist():
            fun[betas.index(d[0]),Cs.index(d[1])]=d[2]

        inc=(betas[1]-betas[0])
        betas.append(betas[-1]+inc)
        Cs.append(Cs[-1]+inc)
        betas=np.array(betas)-inc/2.;Cs=np.array(Cs)-inc/2.
        am= (fun==np.max(fun)).nonzero()
        iam=np.argmin(am[1])
        opt=[betas[am[0][iam]]+inc/2.,Cs[am[1][iam]]+inc/2.]

        # sanity check
        oi=np.logical_and(opt[0]==dat[:,0],opt[1]==dat[:,1])
        assert np.max(fun)==dat[oi.nonzero()[0][0],2]
        np.save(inpath+'svm%s/opt'%suf,opt)
        nf=getWeights(vp,event,suf)
        chnc=max(nf[0],nf[1])/float(nf[0]+nf[1])
        infos.append([vp,np.max(fun)*100,chnc*100]+nf.tolist()
            +[nf[2]/float(nf[0]+nf[1])*100,opt[1],opt[0]])
        plt.subplot(2,2,vp)
        plt.pcolor(betas,Cs,fun.T,cmap='hot')
        plt.xlabel('beta');plt.ylabel('C')
        plt.xlim([betas[0],betas[-1]]);plt.ylim([Cs[0],Cs[-1]])
        plt.colorbar()
        plt.plot(opt[0],opt[1],'rx',mew=2)
        plt.title('b=%.1f, C=%.1f,fm=%.2f,ch=%.2f'%(opt[0],opt[1],np.max(fun),chnc))
    plt.savefig(figpath+'svm%sfitEv%d.png'%(suf,event))
    from matustools.matusplotlib import ndarray2latextable
    ndarray2latextable(np.array(infos),decim=[0,2,2,0,0,0,2,1,1])
    return infos
#################################################
# hill climbing the objective function
def svmObjFun(*args):
    [wid,np,P,F,svvs,beta,weights,D,inpath,suf,invert,x]=args
    SMAX=128
    '''
    compute similarity between x and the selected perc fields
    then compute the svm objective function i.e. (w^T K(x,svs) - b)
    weights[1:] are w and b is in weights[0]
    svs gives the indices of selected perc fields
    '''

    S=np.zeros(D.shape[0],dtype=np.float64)*np.nan
    for n in range(S.size):
        if svvs[n]: S[n]=np.linalg.norm(D[n,:,:,:]-np.float64(x)*SMAX)
    S=S[~np.isnan(S)]
    K=np.exp(-np.exp(beta)*S/5000.)
    res=weights[1:].dot(K)-weights[0]
    return  (-1)**invert * res

def inithc(vp,event,s,suf=''):
    ''' initialize hillclimbing
        s- subsampling factor'''
    path,inpath,fp=initPath(vp,event)
    D0=np.load(inpath+'sPF%s%d.npy'%(suf,s))
    D1=np.load(path+'E%d/sPF%s%d.npy'%(event+1,suf,s))
    D=np.float64(np.concatenate([D0,D1],axis=0))
    P=D.shape[1];F=D.shape[3]
    mask=createMask(P,F)
    D*=mask
    weights=np.load(inpath+'svm%s/weights.npy'%suf)
    svs=np.load(inpath+'svm%s/svs.npy'%suf)
    svvs=np.zeros(D.shape[0])
    svvs[svs]=1
    svvs=svvs>0.5
    [beta,c]=np.load(inpath+'svm%s/opt.npy'%suf).tolist()
    return [0,np,P,F,svvs,beta,weights,D,inpath,suf]



def hillClimb(*args):
    ''' perform hillclimbing'''
    # these are read-only vars
    [wid,np,P,F,svvs,beta,ww,D,inpath,suf]=args
    seed=wid/2-1
    invert=wid%2
    args=list(args)+[invert]
    mask= createMask(P,F)
    print 'worker %d: running, seed=%d,invert=%d' % (wid,seed, invert)
    if seed==-1: x=np.zeros((P,P,F))>0
    else:
        np.random.seed(seed)
        xmin=np.logical_and(np.random.rand(P,P,F)>0.9,mask)
        t0=time()
        fmin=svmObjFun(*tuple(args+[xmin]))
        for k in range(1000):
            x=np.logical_and(np.random.rand(P,P,F)>0.9,mask)
            f=svmObjFun(*tuple(args+[x]))
            if f<fmin: xmin=x
        print 'worker %d: prelim grid search finished: fmin='%(wid), fmin, time()-t0
    fmin=svmObjFun(*tuple(args+[x]))
    loops=20
    t0=time()
    fk=np.inf
    for k in range(loops):
        for h in np.random.permutation(x.size).tolist():
            a,b,c=[h/(P*F),(h%(P*F))/F,h%F]
            #assert (a*P*F+b*F+c)==h
            if mask[a,b,c]:
                x[a,b,c]= not x[a,b,c]
                f=svmObjFun(*tuple(args+[x]))
                if fmin>f:  fmin=f
                else: x[a,b,c]=not x[a,b,c]
            else: x[a,b,c]=False
        if fk==fmin:
            print 'worker %d: converged, f=%f'%(wid,fmin)
            break
        fk=fmin
        print 'worker %d: loop=%d, t=%.3f, fmin=%f'%(wid,k,np.round((time()-t0)/3600.,3),fmin)
        np.save(inpath+'svm%s/hc/hcWorker%d'%(suf,wid),x)


def hcscript(vp,event,nworkers=8,s=2,suf=''):
    '''script that performs the hill climbing '''
    ags=inithc(vp,event,s,suf)
    ps=[]
    for wid in range(0,nworkers):
        ags[0]=wid
        p=Process(target=hillClimb,args=ags)
        p.start();ps.append(p)
    for p in ps: p.join()



#########################################################
# principal component analysis

def PF2X(merge=1,verbose=True):
    ''' translates samples into format for PCA
        merge - # of pf files to merge into one x file
                by default merge=1 and no files are merged
    '''
    pfpath=inpath.rstrip('/X/')+'/PF/'
    fs=os.listdir(pfpath)
    fs.sort();k=0;x=[];h=0

    if verbose: print 'PF2X: merging'
    for f in fs:
        pf=np.load(pfpath+f)
        if pf.shape[0]==0: continue
        #pf=pf[:,32:-32,32:-32,0,:].squeeze()
        pf=pf[:,:,:,0,:].squeeze()
        x.append(pf.reshape((pf.shape[0],pf.size/pf.shape[0])))
        if k%merge==merge-1:
            out=np.concatenate(x,0)
            np.save(inpath+'X%d'%h,out)
            x=[];h+=1
        k+=1
    if len(x)>0:
        out=np.concatenate(x,0)
        np.save(inpath+'X%d'%h,out);h+=1
    if N==1: return np.float64(out)/255.0
    assert h==N
    inc=int(np.ceil(out.shape[1]/float(h)))
    for g in range(h):
        out=[]
        if verbose: print 'PF2X: computing transpose %d/%d'%(g+1,h)
        for j in range(h):
            x=np.load(inpath+'X%d.npy'%j)
            out.append(x[:,g*inc:(g+1)*inc].copy())
            if g==h-1:
                x=np.float64(x)/255.0
                np.save(inpath+'X%d.npy'%j,x)
            del x
        out =np.concatenate(out,0).T
        out=np.float64(out)/255.0
        np.save(inpath+'XT%d'%g,out)
  
def pcaScript(vp,ev):
    ''' performs pca, reference:
        Reference: Bishop, C. (2006) PRML, Chap. 12.1'''
    global inpath,N
    path,inpath,figpath=initPath(vp,ev)
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    inpath=inpath+'X/'
    if not oss.path.exists(inpath): oss.makedirs(inpath)
    mrg=[50,15,8][ev]+1;N=dat['N']/mrg+1
    N=1
    X=PF2X(merge=mrg)
    print 'demeaning'
    X = (X-X.mean(1)[:,np.newaxis])
    print 'computing cov matrix'
    C=np.cov(X)#C=Xcov()
    np.save(inpath+'C',C)
    #C=np.load(inpath+'C.npy')
    print 'shape of Cov matrix is ',C.shape
    print 'computing eigenvalue decomposition'
    [latent,coeff]=np.linalg.eigh(C)
    print 'eig finished'
    indx=np.argsort(latent)[::-1][:100]
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    assert np.allclose(np.linalg.norm(coeff,axis=1),1)
    coeff=coeff[:,indx]
    coeff=X.T.dot(coeff)
    denom=np.sqrt((64*64*68-1)*latent[np.newaxis,indx])
    coeff/=denom #make unit vector length
    np.save(inpath+'coeff.npy',coeff)
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    latent/=latent.sum()
    latent=latent[indx]
    np.save(inpath+'latent',latent)
    print 'computing score'
    score = X.dot(coeff)
    np.save(inpath+'score',score)

#########################################################
# Find optimal sample rotation
def weight(traj):
    ''' third order polynomial spatial window
        no time-dependent weighting
        traj - agent trajectories
    '''
    traj=(traj[:,:-1,:,:]+traj[:,1:,:,:])/2.
    out=np.zeros((traj.shape[0],traj.shape[1],traj.shape[2]))
    dist=np.linalg.norm(traj,axis=3)
    #out=np.float32(dist<5.0)
    out=np.maximum(1-np.power(dist/6,3),0)
    return out

def radialkde(x,y,weights=None,bandwidth=np.pi/6,kernel=None):
    ''' kernel density estimation
        see Wasserman (2004) All of Statistics, chap 20.3  '''
    if weights is None: weights=np.ones(y.size)
    if kernel is None: 
        kernel= lambda x:(2*np.pi)**(-0.5)*np.exp(-np.square(x)/2)
    x=np.atleast_2d(x)
    y=np.atleast_2d(y).T
    weights=np.atleast_2d(weights).T
    dif=np.abs(x-y)
    dif[dif>np.pi]=2*np.pi-dif[dif>np.pi]
    out=np.sum(weights*kernel(dif/bandwidth),axis=0)/bandwidth/x.size
    return out

def computeRotation(vp,event):
    path,inpath,fp=initPath(vp,event)
    D=np.load(inpath+'DG.npy')[:,:,:,:2]
    x=np.linspace(-1,1,3601)*np.pi
    phis=np.zeros(D.shape[0])
    dd=np.diff(D,axis=1)
    movdir=np.arctan2(dd[:,:,:,1],dd[:,:,:,0])
    w=weight(D)
    print 'Computing Rotation'
    for n in range(D.shape[0]):
        pr=D.shape[0]/10
        if n%pr==0: print '%d/10 finished'%(n/pr)
        a=radialkde(x,movdir[n].flatten(),weights=w[n].flatten())
        phis[n]=x[np.argmax(a)]
    np.save(inpath+'phi',phis)

def plotTraj(D,clr='k',rad=5):
    ax=plt.gca()
    for a in range(D.shape[1]):
        plt.plot(D[:,a,0],D[:,a,1],'-'+clr,linewidth=2)
        ar=plt.arrow(D[-2,a,0],D[-2,a,1],D[-1,a,0]-D[-2,a,0],D[-1,a,1]-D[-2,a,1],
              length_includes_head=False,head_width=0.2,fc=clr)
        ax.add_patch(ar)
    c=plt.Circle((0,0),rad,fc='r',alpha=0.1,ec=None)
    ax.add_patch(c)
    m=D.shape[0]/2
    plt.plot(D[m,:,0],D[m,:,1],'or')
    ax.set_aspect(1);lim=3*rad/2
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])

def rotateTraj(traj,phi,PLOT=False):
    ''' rotates trajectory traj by phi radians '''
    c=np.cos(phi);s=np.sin(phi)
    R=np.array([[c,-s],[s,c]])
    if PLOT:
        plotTraj(traj)
        plt.plot([4*c,-4*c],[4*s,-4*s],'g')
        plt.figure()
    for a in range(14): traj[:,a,:]=traj[:,a,:].dot(R) 
    if PLOT: plotTraj(traj)
    return traj


if __name__ == '__main__':
    # extract images
    for vp in range(1,5):
        for ev in range(3)+range(96,100):
            computeRotation(vp,ev)
            PFinit(vp,ev)
            PFparallel(vp,ev)
            # perform pca
            pcaScript(vp,ev)
            
    # compute svm
    exportScript()
    gridSearchScript()
    for vp in range(1,5):
        hcscript(vp,0)
    
    # ideal observer
    vp=999
    computeRotation(vp,1)
    PFinit(vp,1)
    PFparallel(vp,1)
    pcaScript(vp,1)
