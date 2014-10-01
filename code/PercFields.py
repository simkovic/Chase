import numpy as np
import pylab as plt
from psychopy import visual,core
from psychopy.misc import deg2pix
from Constants import *
from Settings import Q
import random, Image,ImageFilter, os,pyglet, pickle,commands
from scipy.ndimage.filters import convolve,gaussian_filter
from ImageOps import grayscale
from psychopy import core
from matustools.matusplotlib import ndarray2gif,plotGifGrid
from time import time, sleep
from multiprocessing import Process

def initPath(vpp,eventt):
    global event,vp,path,inpath,figpath
    event=eventt;vp=vpp
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    inpath=path+'E%d/'%event
    figpath=os.getcwd().rstrip('code')+'figures/PercFields/'
    print 'initPath: vp=%d, ev=%d'%(vp,event)

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

def PFextract(E,part=[0,1],wind=None,elem=None):
    """ part[0] - current part
        part[1] - total number of parts
    """
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    inc=E.shape[0]/part[1]
    start=part[0]*inc
    ende=min((part[0]+1)*inc,E.shape[0])
    print start,ende,E.shape
    os=dat['os'];rot=dat['rot']

    phis=np.load(inpath+'phi.npy')
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
        if len(part)==2: np.save(inpath+'PF/PF%03d.npy'%(part[0]),PF)
        else: np.save('PF.npy',PF)
    except:
        if close: wind.close()
        raise

def PFinit():
    dat={'N':[50,15,8,2][event],'os':64,'rot':1,
         'width':10,'hz':85.0,'SX':0.3,'SY':0.3,'ST':40}
    np.save(inpath+'stackPF.npy',range(dat['N']+1))
    Q.save(inpath+'PF.q')
    f=open(inpath+'PF.pars','w')
    pickle.dump(dat,f)
    f.close()
    
def PFparallel():
    ''' please run PFinit() first
    '''
    E=np.load(inpath+'DG.npy')[:,:,:,:2]
    print E.shape
    stack=np.load(inpath+'stackPF.npy').tolist()
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    N=dat['N']
    wind=Q.initDisplay()
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=E.shape[1], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    while len(stack):
        jobid=stack.pop(0)
        np.save(inpath+'stackPF.npy',stack)
        PFextract(E,[jobid,N],wind=wind, elem=elem)
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
    

def Scompute(vp,evA,evB,pfn='PF'):
    ''' compute similarity matrix between perceptive fields
        vp - subject id
        evA - id of event A
        evB - id of event B
    '''
    print 'Scompute: vp=%d, evA=%d, evB=%d'%(vp,evA,evB)
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    inpa=path+'E%d/'%evA
    inpb=path+'E%d/'%evB
    f=open(inpa+'%s.pars'%pfn,'r');
    dat=pickle.load(f);f.close();N1=dat['N']+1
    f=open(inpb+'%s.pars'%pfn,'r');
    dat=pickle.load(f);f.close();N2=dat['N']+1
    D1=np.load(inpa+'%s/%s000.npy'%(pfn,pfn))
    D2=np.load(inpb+'%s/%s000.npy'%(pfn,pfn))
    ds1=D1.shape[0];ds2=D2.shape[0];
    assert dat['os']==D1.shape[1]
    P=D1.shape[1];F=D1.shape[4]
    dga=np.load(inpa+'DG.npy').shape[0]
    dgb=np.load(inpb+'DG.npy').shape[0]
    print dga,dgb,ds1,ds2
    # create mask with circular aperture
    mid=(P-1)/2.0
    mask=np.zeros((P,P,F),dtype=np.bool8)
    suf=['ev%d'%evB,''][int(evA==evB)]
    S=np.zeros([dga,dgb])*np.nan
    for i in range(P):
        for j in range(P):
            if np.sqrt((i-mid)**2+(j-mid)**2)<=P/2.0: mask[i,j,:]=True
    # compute similarity
    for pf1 in range(0,N1):
        for pf2 in range(pf1*int(evA==evB),N2):
            D1=np.load(inpa+'%s/%s%03d.npy'%(pfn,pfn,pf1))
            D2=np.load(inpb+'%s/%s%03d.npy'%(pfn,pfn,pf2))
            Spart=np.zeros((D1.shape[0],D2.shape[0]))*np.nan
            for n1 in range(D1.shape[0]):
                for n2 in range(D2.shape[0]):
                    a=np.float32(D1[n1,:,:,0,:])
                    b=np.float32(D2[n2,:,:,0,:])
                    Spart[n1,n2]=(np.square(a-b)*mask).sum()
            Spart=np.sqrt(Spart);sps=Spart.shape
            S[pf1*ds1:(pf1*ds1+sps[0]),pf2*ds2:(pf2*ds2+sps[1])]=Spart
            if evA==evB: S[pf2*ds2:(pf2*ds2+sps[1]),pf1*ds1:(pf1*ds1+sps[0])]=Spart.T
    np.save(inpa+'S'+suf,S)

def SexportSvm(vp,ev,beta=1):
    ''' beta on log scale '''
    print 'Exporting'
    e1=ev; e2=ev+1
    fn1=range(0,602,5);fn2=range(152)
    pth=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    S1=np.load(pth+'E%d/S.npy'%(e1))
    S2=np.load(pth+'E%d/S.npy'%(e2))
    Scross=np.load(pth+'E%d/Sev%d.npy'%(e1,e2))
    n1=S1.shape[0];n2=S2.shape[0]
    S=np.zeros((n1+n2,n1+n2))*np.nan
    #print S[:n1,:n1].shape, S1.shape, n1/float(n1+n2)
    S[:n1,:n1]=S1
    S[:n1,n1:]=Scross
    #print S[:n1,n1:].shape,Scross.shape
    S[n1:,:n1]=Scross.T
    S[n1:,n1:]=S2
    del S1,S2,Scross
    ##print np.any(np.isnan(S)), np.min(S), np.max(S)
    ##ks=[0.125,0.25,0.5,1,2,4,8]
    ##ms=[100,500,1000,5000,10000]
    ##for k in range(len(ks)):
    ##    for m in range(len(ms)):
    ##        plt.subplot(5,7, m*7+k+1)
    ##        plt.hist(np.exp(-np.power(S.flatten()/float(ms[m]),ks[k])),1000)

    # use radial basis function, note values in S are already squared
    S=np.exp(-beta*S/5000.)
    #print np.any(np.isnan(S)),np.min(S),np.max(S)
    #print np.any(np.isinf(S))
    f=open(pth+'E%d/svm/svm.in'%(e1),'w')
    for row in range(n1+n2):
        s='%d 0:%d'%(int(row<n1),row+1)
        for col in range(n1+n2):
            s+=' %d:%.4f'%(col+1,S[row,col])
        s+='\n'
        f.write(s)
        f.flush()
    f.close()
    print 'Export Finished'
def svmGridSearch(betas=None, cs=None):
    fn=inpath+'svm/svm'
    logf=open(fn+'.log','w')
    if betas is None: betas=np.arange(-5,10,0.5)
    if cs is None: cs=np.arange(-5,15,0.5)
    logf.write('betas = '+str(betas)+'\ncs = '+str(cs)+'\n');
    res=[]
    for b in betas:
        res.append([])
        SexportSvm(vp,event,beta=np.exp(b))
        for c in cs:
            print 'beta = %.3f, C = %f'%(b,c)
            #print 'svm-train -s'+' 0 -v 5 -t 4 -c %f -m 6000 %s.in %s.model'%(c,fn,fn)
            status,output=commands.getstatusoutput('svm-train -s '+
                '0 -v 5 -t 4 -c %f -m 6000 %s.in'%(np.exp(c),fn))
            if status:
                print output
                raise RuntimeError(output)
            logf.write(output+'\n')
            logf.flush()
            #print output
            temp=float(output.rsplit(' ')[-1].rstrip('%'))/100.
            print temp
            res[-1].append(temp)
    res=np.array(res)
    logf.close()
    np.save(inpath+'svm/grid.npy',res)
    X,Y=np.meshgrid(cs,betas)
    print X.shape,Y.shape,res.shape
    plt.pcolor(X,Y,res);
    am=np.nonzero(res==np.max(res))
    iam=np.argmax(am[1])
    
    beta=betas[am[0][iam]]
    c=cs[am[1][iam]]
    plt.xlabel('cs'); plt.ylabel('betas')
    plt.savefig('grid.png')
    print 'beta=',beta, ',c=',c
    print np.max(res),np.min(res)
    np.savetxt(inpath+'svm/opt.par',[beta,c])
    

def svmFindSvs():
    [beta,c]=np.loadtxt(inpath+'svm/opt.par').tolist()
    # compute the model
    SexportSvm(vp,event,beta)
    fn=inpath+'svm/svm'
    status,output=commands.getstatusoutput('svm-train -s '+
            '0 -t 4 -c %f %s.in %s.model'%(np.exp(c),fn,fn))
    if status: print output
    # save support vectors to npy file for later use
    f=open(fn+'.model','r')
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
    print svs.shape, weights.shape
    np.save(inpath+'svm/weights.npy',weights)
    np.save(inpath+'svm/svs.npy',svs)

def pfSubsample(s=4):
    ''' s - multiplicative subsampling factor'''
    f=open(inpath+'PF.pars','r');
    dat=pickle.load(f);f.close();N=dat['N']+1
    print N
    out=[]
    for h in range(0,N):
        D=np.load(inpath+'PF/PF%03d.npy'%h)
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
        print h
    print out[0].shape,out[-1].shape
    out=np.concatenate(out,axis=0)
    print out.shape
    np.save(inpath+'sPF%d.npy'%s,out)

def hillClimb(nworkers=8,s=4): 
    D0=np.load(inpath+'sPF%d.npy'%s)
    D1=np.load(path+'E%d/sPF%d.npy'%(event+1,s))
    D=np.float64(np.concatenate([D0,D1],axis=0))
    print D.shape
    P=D.shape[1];F=D.shape[3]
    # create mask with circular aperture
    mid=(P-1)/2.0
    try: mask=np.load('mask.npy')
    except IOError:
        mask=np.zeros((P,P,F),dtype=np.bool8)
        for i in range(P):
            for j in range(P):
                if np.sqrt((i-mid)**2+(j-mid)**2)<=P/2.0: mask[i,j,:]=True
        np.save('mask',mask)

    #ww=np.zeros(dga+dgb)
    weights=np.load(inpath+'svm/weights.npy')
    svs=np.load(inpath+'svm/svs.npy')
    ww=weights[1:]
    svvs=np.zeros(D.shape[0])
    svvs[svs]=1
    svvs=svvs>0.5
    [beta,c]=np.loadtxt(inpath+'svm/opt.par').tolist()
    def worker(*args):
        # these are read-only vars
        [wid,np,P,F,svvs,beta,ww,D,mask]=args
        def svmObjFun(x,SMAX=128):
            '''
            compute similarity between x and the selected perc fields
            then compute the svm objective function i.e. (w^T K(x,svs) - b)
            weights[1:] are w and b is in weights[0]
            svs gives the indices of selected perc fields
            '''
            x=np.reshape(x,[P,P,F])
            S=np.zeros(D.shape[0])*np.nan
            for n in range(S.size):
                if svvs[n]: S[n]=np.linalg.norm(D[n,:,:,:]-np.float64(x)*SMAX)
            S=S[~np.isnan(S)]
            #print np.max(D),np.min(D)
            #print np.max(S),np.min(S),beta, np.max(weights),np.min(weights)
            K=np.exp(-np.exp(beta)*S/5000.)
            res=ww.dot(K)-weights[0]
            #print 'res=',res
            return (-1)**invert *res 
        seed=wid/2-1
        invert=wid%2
        print 'worker %d: running, seed=%d,invert=%d' % (wid,seed, invert)
        if seed==-1: x=np.zeros(P*P*F)>0
        else:
            np.random.seed(seed)
            xmin=np.random.rand(P*P*F)>0.9
            t0=time()
            fmin=svmObjFun(xmin)
            for k in range(1000):
                x=np.random.rand(P*P*F)>0.9
                f=svmObjFun(x)
                if f<fmin: xmin=x
            print 'worker %d: prelim grid search finished: fmin='%(wid), fmin, time()-t0
        fmin=svmObjFun(x)
        loops=20
        t0=time()
        fk=np.inf
        for k in range(loops):
            for h in np.random.permutation(x.size).tolist():
                x[h]= not x[h]
                f=svmObjFun(x)
                if fmin>f:  fmin=f
                else: x[h]=not x[h]
            if fk==fmin:
                print 'worker %d: converged, f=%f'%(wid,fmin)
                break
            fk=fmin
            print 'worker %d: loop=%d, t=%.3f, fmin=%f'%(wid,k,np.round((time()-t0)/3600.,3),fmin)
            np.save(inpath+'svm/hc/hcWorker%d'%(wid),x)
    ps=[]
    for wid in range(0,nworkers):
        p=Process(target=worker,args=(wid,np,P,F,svvs,beta,ww,D,mask))
        p.start();ps.append(p)
    for p in ps: p.join()

def svmPlotExtrema(event=0):
    P=32;F=34
    dat=[]
    print len(dat)
    for vp in range(1,5):
        initPath(vp,event)
        fn= inpath+'svm/hc/hcWorker'
        for g in range(2):
            dat.append([])
            for k in range(4):
                try:temp=np.load(fn+'%d.npy'%(k*2+g))
                except IOError:temp=np.ones(P*P*F)
                temp=np.reshape(temp,[P,P,F])
                dat[-1].append(temp)
    plotGifGrid(dat,fn=figpath+'svmExtremaE%d'%event)



#Scompute(sid,ev,ev+1)
##Scompute(sid,ev+1,ev+1)
##sid=1
##ev=1
##initPath(sid,ev)
##svmGridSearch()       
##svmFindSvs()
##for ev in [0]:
##    initPath(sid,ev)
##    pfSubsample(s=2)
    

##for sid py range(1,5):
##    initPath(sid,1)
##    hillClimb(nworkers=8,s=2)
#initPath(2,1)
#hillClimb(nworkers=8,s=2)
svmPlotExtrema(0)
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


def pcaSVD(A,highdim=None):
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
    n=A.shape[0];m=A.shape[1]
    A=np.array(A)
    if highdim==None: highdim=n<m
    M = (A-np.array(A.mean(1),ndmin=2).T) # mean-center data
    if highdim:
        [latent,coeff] = np.linalg.eigh(np.cov(M))
        coeff=M.T.dot(coeff)
    else:
        [latent,coeff] = np.linalg.eigh(np.cov(M.T))
    score = M.dot(coeff).T
    coeff/=np.sqrt(A.shape[int(highdim==False)]*np.array(latent,ndmin=2)) #unit vector length
    latent/=latent.sum()
    # sort the data
    indx=np.argsort(latent)[::-1]
    latent=latent[indx]
    coeff=coeff[:,indx]
    score=score[indx,:]
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
    c1,s1,v1=pcaSVD(X,True)
    c2,s2,v2=pcaSVD(X,False)
    print v1,v2

    X=np.random.rand(52,M)
    split(X)
    pcaNIPALS(K=M)
    a1=mergeT('coeff')
    c1=np.load(inpath+'latent.npy')
    a2,b2,c2=pcaSVD(X)
    print c2/c2.sum(), c2.sum()
    for k in range(a1.shape[1]):
        plt.figure()
        plt.plot(a2[:,k]);
        if np.abs((a2[:,k]-a1[:,k])).mean()> np.abs((a2[:,k]+a1[:,k])).mean() : plt.plot(-a1[:,k])
        else: plt.plot(a1[:,k])
    plt.show()

def pcaScript():
    global inpath,N
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    inpath=inpath+'X/'
    mrg=4;N=dat['N']/mrg+1
    PF2X(merge=mrg)

    C=Xcov()
    np.save(inpath+'C',C)

    C=np.load(inpath+'C.npy')
    print 'shape of Cov matrix is ',C.shape
    print 'computing eigenvalue decomposition'
    [latent,coeff]=np.linalg.eigh(C)
    print 'eig finished'
    print coeff.shape
    latent=latent[::-1]
    coeff=coeff[:,::-1]
    latent/=latent.sum()
    np.save(inpath+'latent',latent[:100])
    np.save(inpath+'coeff.npy',coeff[:,:100])

    coeff=np.load(inpath+'coeff.npy')
    coeff=Xleftmult(coeff,True)
    print coeff.shape
    np.save(inpath+'coeff.npy',coeff)

    score=Xleftmult(coeff)
    np.save(inpath+'score',score)
#initPath(3,0)
#pcaScript()

def _getPC(pf,h):
    if pf.shape[0]!=64:pf=pf[:,h].reshape((64,64,68))
    pf-= np.min(pf)
    pf/= np.max(pf)
    return pf.squeeze().T

def plotCoeff(rows=8,cols=5):
    coeff=np.load(inpath+'X/coeff.npy')
    offset=8 # nr pixels for border padding
    R=np.ones((69,(64+offset)*rows,(64+offset)*cols),dtype=np.float32)
    for h in range(coeff.shape[1]):
        if h>=rows*cols:continue
        c= h%cols;r= h/cols
        s=((offset+64)*r+offset/2,(offset+64)*c+offset/2)
        R[1:,s[0]:s[0]+64,s[1]:s[1]+64]=_getPC(coeff,h)
    ndarray2gif('pcAllvp%de%d'%(vp,event),np.uint8(R*255),duration=0.1)

def plotLatent():
    for ev in range(2):
        for vp in range(1,5):
            initPath(vp,ev)
            plt.plot(np.load(inpath+'X/latent.npy')[:20],['r','g'][ev])
    plt.show()


def plotScore(pcs=5,scs=3):
    score=np.load(inpath+'X/score.npy')
    coeff=np.load(inpath+'X/coeff.npy')
    offset=8 # nr pixels for border padding
    rows=scs*2+1; cols=pcs
    R=np.ones((69,(64+offset)*rows,(64+offset)*cols),dtype=np.float32)
    f=open(inpath+'PF.pars','r');dat=pickle.load(f);f.close()
    bd=score.shape[0]/dat['N']
    for h in range(pcs):
        s=((offset+64)*scs+offset/2,(offset+64)*h+offset/2)
        pc=_getPC(coeff,h)
        print pc.mean()
        if False and pc.mean()>=0.4: invert= -1
        else: invert= 1
        R[1:,s[0]:s[0]+64,s[1]:s[1]+64]=invert*pc
        
        ns=np.argsort(invert*score[:,h])[range(-scs,0)+range(scs)]
        for i in range(len(ns)):
            pf=np.load(inpath+'PF/PF%03d.npy'%(ns[i]/bd))[ns[i]%bd,:,:,0,:]
            s=((offset+64)*(i+int(i>=scs))+offset/2,(offset+64)*h+offset/2)
            #print h,i,ns[i],ns[i]/bd,ns[i]%bd, s
            R[1:,s[0]:s[0]+64,s[1]:s[1]+64]= _getPC(np.float32(pf),h)
    ndarray2gif('scoreVp%de%d'%(vp,event),np.uint8(R*255),duration=0.1)



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

def computeRotation():
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

##for iev in [1,0]:
##    for ivp in range(1,5):
##        print 'vp = %d, ev = %d'%(ivp,iev)
##        initPath(ivp,iev)
##        #computeRotation()
##        #PFinit()
##        #PFparallel()
##        #pcaScript()
##        plotCoeff()
