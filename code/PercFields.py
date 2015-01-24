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
def initPath(vpp,eventt):
    #global event,vp,path,inpath,figpath
    event=eventt;vp=vpp
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    if event>=0: inpath=path+'E%d/'%event
    else: inpath=path+'E%d/'%(100+event)
    figpath=os.getcwd().rstrip('code')+'figures/PercFields/'
    #print 'initPath: vp=%d, ev=%d'%(vp,event)
    return path,inpath,figpath
#########################################################
#                                                       #
#   Translate Coordinates to Perceptive Fields          #
#                                                       #
#########################################################

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

def PFextract(E,part=[0,1],wind=None,elem=None,inpath='',suf=''):
    """ part[0] - current part
        part[1] - total number of parts
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
#                                                       #
#                SVM                                    #
#                                                       #
#########################################################
def createMask(P,F):
    # create mask with circular aperture
    mid=(P-1)/2.0
    mask=np.zeros((P,P,F),dtype=np.bool8)
    for i in range(P):
        for j in range(P):
            if np.sqrt((i-mid)**2+(j-mid)**2)<=P/2.0: mask[i,j,:]=True
    return mask 

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
    ''' s - multiplicative subsampling factor'''
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
    pool=Pool(processes=8)
    vps=[1,2];
    for ags in [[1,1],[1,2],[2,2]]:
        for vp in vps:
            pool.apply_async(pfExport,[vp]+ags+[suf])       
    for ev in [1,2]:
        for vp in vps:
            pool.apply_async(pfSubsample,[vp,ev,2,suf])   
    pool.close()
    pool.join()


def SexportSvm(vp,ev,beta,fn,suf=''):
    ''' beta on log scale '''
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
        s='%d 0:%d'%(int(row<n1),row+1)
        for col in range(n1+n2):
            s+=' %d:%.4f'%(col+1,S[row,col])
        s+='\n'
        f.write(s)
        f.flush()
    f.close()
    print strid+'finished'

def SevalSvm(vp,ev,b,fn,suf):
    strid='SevalSvm vp=%d, ev=%d,beta=%.1f: '%(vp,ev,b)
    print strid+'started'
    cs=np.arange(-10,10,0.5)
    SexportSvm(vp,ev,b,fn,suf)
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
    pool=Pool(processes=8)
    vps=[1,2]
    betas=np.arange(-5,10,0.5)

    for ev in [1]:
        for vp in vps:
            path,inpath,figpath=initPath(vp,ev)
            fn=inpath+'svm%s/svm'%suf
            for beta in betas:
                pool.apply_async(SevalSvm,[vp,ev,beta,fn,suf])
                if ev==0: sleep(20)
    pool.close()
    pool.join()
#path,inpath,figpath=initPath(1,1)
#SevalSvm(1,1,-0.5,inpath+'svm2/svm','2')   

def getWeights(vp,event,suf):
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


#exportScript(suf='3')
#gridSearchScript(suf='3')
#plotSvm(event=0,suf='')
#plotSvm(event=1,suf='')
#plotSvm(event=1,suf='3')
#plotSvm(event=1,suf='2')

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
    ags=inithc(vp,event,s,suf)
    ps=[]
    for wid in range(0,nworkers):
        ags[0]=wid
        p=Process(target=hillClimb,args=ags)
        p.start();ps.append(p)
    for p in ps: p.join()

#hcscript(2,1,suf='2')



def svmPlotExtrRep(event=0,plot=True,suf=''):
    
    P=32;F=34
    dat=[]
    for vp in range(1,5):
        path,inpath,figpath=initPath(vp,event)
        fn= inpath+'svm%s/hc/hcWorker'%suf
        dat.append([])
        for g in range(2):
            for k in range(4):
                try:temp=np.load(fn+'%d.npy'%(k*2+g))
                except IOError:
                    print 'File missing: ',vp,event,suf
                    temp=np.zeros(P*P*F,dtype=np.bool8)
                temp=np.reshape(temp,[P,P,F])
                dat[-1].append(np.bool8(g-1**g *temp))
    if plot: plotGifGrid(dat,fn=figpath+'svm%sExtremaE%d'%(suf,event),bcgclr=0.5)
    return dat
def svmPlotExtrema():
    from matustools.matusplotlib import plotGifGrid
    out=[[],[],[],[]]
    for nf in [[0,''],[1,''],[1,'3'],[1,'2']]:
        dat=svmPlotExtrRep(nf[0],suf=nf[1])
        for vp in range(4):
            out[vp].extend([dat[vp][1],dat[vp][5]])
    path,inpath,figpath=initPath(1,0)
    txt=[['VP1',16,20,-16],['VP2',16,60,-16],['VP3',16,100,-16],['VP4',16,140,-16],
         ['A',16,-8,60],['B',16,-8,140],['C',16,-8,220],['D',16,-8,300]]
    plotGifGrid(out,fn=figpath+'svmExtrema',bcgclr=0.5,
                text=txt,duration=0.2,plottime=True,snapshot=True)  
        

##initPath(3,1)
##args=inithc(s=2)
##invert=1
##xx=dat[4][0].flatten()
##
##svs=args[-1][args[4],:,:,:]
##out=np.zeros(svs.shape[0],dtype=np.float128)
##for si in range(svs.shape[0]):
##    xx=svs[si].flatten()
##    #xx=np.random.rand(32*32*34)>0.9
##    out[si]= svmObjFun(*tuple(args+[xx]))
##    #print out[si],type(out[si])
##np.save('out',out)



##initPath(sid,ev)
###svmGridSearch()       
###svmFindSvs()    
##hillClimb(nworkers=8,s=2)

#########################################################
#                                                       #
#                       PCA                             #
#                                                       #
#########################################################

def PF2X(merge=1,verbose=True):
    ''' merge - # of pf files to merge into one x file
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

def split(X):
    inc=int(np.ceil(X.shape[0]/float(N)))
    for i in range(N):
        np.save(inpath+'X%d.npy'%i,X[i*inc:(i+1)*inc,:])
    inc=int(np.ceil(X.shape[1]/float(N)))
    for i in range(N):
        np.save(inpath+'XT%d.npy'%i,X[:,i*inc:(i+1)*inc].T)
        
def merge(fprefix='X',N=5):
    out=[]
    for i in range(N): out.append(np.load(inpath+fprefix+'%d.npy'%i))
    return np.concatenate(out,0)
def mergeT(fprefix='X'):
    out=[]
    for i in range(N): out.append(np.load(inpath+fprefix+'T%d.npy'%i).T)
    return np.concatenate(out,1)
def XgetColumn(k):
    inc=np.load(inpath+'XT0.npy').shape[0]
    a=k/inc;b=k%inc
    return np.matrix(np.load(inpath+'XT%d.npy'%a)[b,:]).T
    

def Xleftmult(A,transpose=False,verbose=True):
    ''' OUT=X*A or OUT=X.T*A '''
    out=[];A=np.matrix(A)
    for i in range(N):
        if verbose: print 'Xleftmult: %d/%d'%(i+1,N)
        X=np.load(inpath+'X%d.npy'%i)
        if transpose: X=np.load(inpath+'XT%d.npy'%i)
        out.append(X*A)
    return np.concatenate(out,0)

def XminusOuterProduct(A,B):
    ''' X=X-A*B.T todo save XT also'''
    out=[];A=np.matrix(A);B=np.matrix(B).T
    s=0
    for i in range(N):
        X=np.load(inpath+'X%d.npy'%i)
        e=s+X.shape[0]
        X= X- A[s:e,:]*B
        np.save(inpath+'X%d.npy'%i,X)
        s=e
    s=0
    for i in range(N):
        X=np.load(inpath+'XT%d.npy'%i)
        e=s+X.shape[0]
        X= X- (A*B[:,s:e]).T
        np.save(inpath+'XT%d.npy'%i,X)
        s=e
def XmeanCenter(axis=1,verbose=True):
    #compute mean
    if verbose: print 'XmeanCenter: computing mean'
    ss=['','T']
    tot=0
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[axis],i))
        if i==0: res=X.sum(0)
        else: res+=X.sum(0)
        tot+=X.shape[0]
    res/=tot
    np.save(inpath+'X%smean.npy'%ss[axis],res.squeeze())
    if verbose: print 'XmeanCenter: subtracting mean from X'
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[axis],i))
        np.save(inpath+'X%s%d.npy'%(ss[axis],i),X-np.matrix(res))
    inc=int(np.ceil(X.shape[1]/float(N)))
    if verbose: print 'XmeanCenter: subtracting mean from XT'
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[1-axis],i))
        np.save(inpath+'X%s%d.npy'%(ss[1-axis],i),X-np.matrix(res[i*inc:(i*inc+X.shape[0])]).T)  

def Xcov(transpose=False,verbose=True):
    ''' NOTE: Xcov demeans X on hard drive '''
    transpose=int(transpose)
    XmeanCenter(axis=1-transpose)
    ss=['','T']
    C=[[0]*N]*N
    for i in range(N):
        X1=np.load(inpath+'X%s%d.npy'%(ss[transpose],i))
        for j in range(N):
            print 'Xcov: Computing element [%d,%d]/%d'%(i,j,N)
            X2=np.load(inpath+'X%s%d.npy'%(ss[transpose],j))
            C[i][j]= X1.dot(X2.T)
        C[i]=np.concatenate(C[i],1)
    return np.concatenate(C,0)/float(X1.shape[1]-1)


def pcaEIG(A,highdim=None):
    """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to features/attributes. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the normalized eigenvalues (percent variance explained)
    of the covariance matrix of A.
    Reference: Bishop, C. (2006) PRML, Chap. 12.1
    """
    A=np.array(A)
    n=A.shape[0];m=A.shape[1]
    highdim = n<m
    assert n!=m
    M = (A-A.mean(1)[:,np.newaxis]) # mean-center data
    if highdim:
        [latent,coeff] = np.linalg.eigh(np.cov(M))
        coeff=M.T.dot(coeff)
        denom=np.sqrt((A.shape[1]-1)*latent[np.newaxis,:])
        coeff/=denom #make unit vector length
    else:
        [latent,coeff] = np.linalg.eigh(np.cov(M.T))
    score = M.dot(coeff)
    latent/=latent.sum()
    # sort the data
    indx=np.argsort(latent)[::-1]
    latent=latent[indx]
    coeff=coeff[:,indx]
    score=score[:,indx]
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    return coeff,score,latent
  
def pcaNIPALS(K=5,tol=1e-4,verbose=False):
    ''' Reference:
            Section 2.2 in Andrecut, M. (2009).
            Parallel GPU implementation of iterative PCA algorithms.
            Journal of Computational Biology, 16(11), 1593-1599.
            
    '''
    if verbose: print 'Mean centering columns'
    XmeanCenter(1)
    latent=[]
    for k in range(K):
        lam0=0;lam1=np.inf
        T=np.matrix(XgetColumn(k))
        if verbose: print 'Computing PC ',k
        h=0
        while abs(lam1-lam0)>tol and h<100:
            P=Xleftmult(T,True)
            P=P/np.linalg.norm(P)
            T=Xleftmult(P)
            lam0=lam1
            lam1=np.linalg.norm(T)
            if verbose: print '\t Iteration '+str(h)+', Convergence =', abs(lam1-lam0)
            h+=1
        latent.append(lam1)
        XminusOuterProduct(T,P)
        #np.save(inpath+'T%02d'%k,T)
        np.save(inpath+'coeffT%d'%k,P.T)
    np.save(inpath+'latent',latent)

def testPca():
    global inpath
    global N
    inpath=''
    N=5;M=21
    np.random.seed(1)
    X=np.random.rand(52,M)
    split(X)
    assert np.all(np.abs(X-merge())==0)
    assert np.all(np.abs(np.matrix(X[:,M/2]).T-XgetColumn(M/2))==0)
    Y1=(X-np.matrix(X.mean(1)).T)
    XmeanCenter(1)
    Y2=mergeT()
    assert np.mean(np.abs(Y1-Y2))<1e-8
    Y1=(X-np.matrix(X.mean(0)))
    split(X)
    XmeanCenter(0)
    Y2=mergeT()
    assert np.mean(np.abs(Y1-Y2))<1e-8
    Y3=merge()
    assert np.mean(np.abs(Y1-Y3))<1e-8
    split(X)
    A=np.matrix(np.random.rand(M,3))
    Y1=X*A
    Y2=Xleftmult(A)
    assert np.all(np.abs(Y1-Y2)==0)
    split(X)
    A=np.matrix(np.random.rand(52,3))
    Y1=X.T*A
    Y2=Xleftmult(A,True)
    assert np.all(np.abs(Y1-Y2)<0.001)
    A=np.matrix(np.random.rand(52,1))
    B=np.matrix(np.random.rand(M,1))
    Y1=X-A*B.T
    XminusOuterProduct(A,B)
    Y2=merge()
    assert np.all(np.abs(Y1-Y2)==0)
    Y3=mergeT()
    assert np.all(np.abs(Y1-Y3)==0)
    C1=np.cov(X.T)
    split(X)
    C2=Xcov(True)
    assert np.abs(C1-C2).mean()<1e-12
    global c1,s1,v1,c2,s2,v2
    X=np.random.rand(52,M)
    split(X)
    pcaNIPALS(K=M)
    a1=mergeT('coeff')
    c1=np.load(inpath+'latent.npy')
    a2,b2,c2=pcaEIG(X)
    print c2/c2.sum(), c2.sum()
    for k in range(a1.shape[1]):
        plt.figure()
        plt.plot(a2[:,k]);
        if np.abs((a2[:,k]-a1[:,k])).mean()> np.abs((a2[:,k]+a1[:,k])).mean() : plt.plot(-a1[:,k])
        else: plt.plot(a1[:,k])
    plt.show()
#testPca()

def pcaScript(vp,ev):
    global inpath,N
    path,inpath,figpath=initPath(vp,ev)
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    inpath=inpath+'X/'
    if not oss.path.exists(inpath): oss.makedirs(inpath)
    mrg=[50,15,8][ev]+1;N=dat['N']/mrg+1
    N=1
    X=PF2X(merge=mrg)
    #X=np.load(inpath+'X.npy')
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
    coeff=X.T.dot(coeff)#coeff=Xleftmult(coeff,True)
    # this is the formula from Bishop, C. (2006) PRML, Chap. 12.1
    denom=np.sqrt((64*64*68-1)*latent[np.newaxis,indx])
    coeff/=denom #make unit vector length
    np.save(inpath+'coeff.npy',coeff)
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    latent/=latent.sum()
    latent=latent[indx]
    np.save(inpath+'latent',latent)
    print 'computing score'
    score = X.dot(coeff)#score=Xleftmult(coeff)
    np.save(inpath+'score',score)



#for ev in [0]:
#    for vp in range(1,5):   
#        pcaScript(vp,ev)

def controlSimulationsPIX():
    P=64;T=50
    I=np.random.rand(P,P,T)*1e-2
    for t in range(T):
        I[t:t+3,P/2-2:P/2+2,t] = 1
        #I[[t+10,t+11],P/2,t] = 1
        #I[[t+10,t+11],P/2+1,t] = 1
    it=10
    D=[np.matrix((1-I).flatten())]
    for phi in range(it,360,it):
        print phi
        Im=np.copy(I)
        for t in range(T):
            M=Image.fromarray(Im[:,:,t])
            M=M.rotate(float(phi))
            Im[:,:,t]=np.asarray(M)
        D.append(np.matrix((1-Im).flatten()))
    D=np.concatenate(D)
    ndarray2gif('dat',1-I.T,addblank=True)
    D=D-np.matrix(D.mean(0))

    coeff,score,latent=pcaSVD(D)
    for k in range(coeff.shape[1]):
        cff=coeff[:,k].reshape((P,P,T)).T
        cff-=np.min(cff)
        cff/=np.max(cff)
        ndarray2gif('pc%02d'%k,cff,addblank=True)

def controlSimulationsXY():
    def rotate(x,phi):
        phi=phi/180.0*np.pi
        R=np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
        return R.dot(x)
    def traj2gif(K,T,A,P,fn='pf'):
        K=np.reshape(np.array(K),(T,A,2))
        I=[]
        for t in range(T):
            Im=np.zeros((P,P,1))
            for a in range(A):
                Im[P/2+K[t,a,1],P/2+K[t,a,0],0]=1
            I.append(Im)
        I=np.concatenate(I,axis=2)
        PFlist2gif(I,fname=fn)
    plt.close('all')
    A=2
    T=20; P=50
    I=np.zeros((T,A,2))
    for t in range(T):
        I[t,0,0] = t
        I[t,1,0] = t-T
    #I[:,:,1]=10
    it=10
    D=[np.matrix(I.flatten())]
    for phi in range(it,360,it):
        Im=np.copy(I)
        for t in range(T):
            Im[t,:,:]=rotate(Im[t,:,:].T,float(phi)).T
            Im[t,:,1]+=10
        D.append(np.matrix(Im.flatten()))
    D=np.concatenate(D)
       
    coeff,score,latent=pcaSVD(D)
    traj2gif(coeff[:,0]*50,T,A,P,fn='PC0')
    traj2gif(coeff[:,1]*50,T,A,P,fn='PC1')
    plt.plot(score[0,:],score[1,:],'o')

#########################################################
#                                                       #
#           Find optimal perc field rotation            #
#                                                       #
#########################################################


def weight(traj):
    ''' third order polynomial spatial window
        no time-dependent weighting
    '''
    traj=(traj[:,:-1,:,:]+traj[:,1:,:,:])/2.
    out=np.zeros((traj.shape[0],traj.shape[1],traj.shape[2]))
    dist=np.linalg.norm(traj,axis=3)
    #out=np.float32(dist<5.0)
    out=np.maximum(1-np.power(dist/6,3),0)
    return out

def radialkde(x,y,weights=None,bandwidth=np.pi/6,kernel=None):
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
    c=np.cos(phi);s=np.sin(phi)
    R=np.array([[c,-s],[s,c]])
    if PLOT:
        plotTraj(traj)
        plt.plot([4*c,-4*c],[4*s,-4*s],'g')
        plt.figure()
    for a in range(14): traj[:,a,:]=traj[:,a,:].dot(R) 
    if PLOT: plotTraj(traj)
    return traj




##for iev in [-1,-2,-3,-4,-5,-6]:
##    for ivp in [4]:#range(1,5):
##        print 'vp = %d, ev = %d'%(ivp,iev)
##        computeRotation(ivp,iev)
##        PFinit(ivp,iev)
##        PFparallel(ivp,iev)
##        #pcaScript()
##        plotCoeff()

# ideal observer
vp=999
#computeRotation(vp,1)
#PFinit(vp,1)
#PFparallel(vp,1)
pcaScript(vp,1)
