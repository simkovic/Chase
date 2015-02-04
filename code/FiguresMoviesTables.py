import numpy as np
import pylab as plt
from scipy import stats
from matustools.matusplotlib import *
from Coord import initVP
import os
DPI=300
FIG=(('behdata.png',('Detection Time','' ),
                     ('Subject','Mean Detection Time'),
                     ('Subject',),
                     ('','Mean Proportion Correct'),()),
     ('bdtime.png',('Trial','Proportion Correct',['S1','S2','S3','S4'])),
     ('Coord/agdens',('Radial Distance','Agent Density',['S1','S2','S3','S4','RT']),
      ('Time to Saccade Onset','Agent Density'),
      ('Radial Distance','Direction Changes'),
      ('Time to Saccade Onset','Direction Changes'),
      ('Radial Distance', 'Light Contrast Saliency'),
      ('Radial Distance', 'Motion Saliency'),
      ('Time to Saccade Onset', 'Light Contrast Saliency'),
      ('Time to Saccade Onset', 'Motion Saliency')),
     ('Pixel/buttonPress',()),
     ('Coord/trackEv',()),
     ('Coord/trackVel',('Time','Velocity',['S1','S2','S3','S4'])),
     ('Coord/trackPFrail',('','',['Start','Middle','End'],'S'))
     )


inpath = os.getcwd().rstrip('code')+'evaluation'+os.path.sep
figpath = os.getcwd().rstrip('code')+'figures'+os.path.sep
def plotBehData():
    plt.close()
    from matustools.matustats import lognorm
    rtmc=np.load(inpath+'rtmc.npy')
    rejs=np.load(inpath+'rejs.npy')
    rts=np.load(inpath+'rts.npy')
    acc=np.load(inpath+'acc.npy')
    figure(size=1,aspect=1.4)
    subplot(3,1,1)
    vp=0
    d=rts[vp,rejs[vp,:]==1]
    d=d[d>0]
    x=np.linspace(1,30,30)
    hist(d,bins=x,normed=True)
    plt.plot(x-0.5,lognorm(mu=rtmc[vp,-2,:].mean(),
                sigma=rtmc[vp,-1,:].mean()).pdf(x-0.5),'k')
    plt.xlabel(FIG[0][1][0])
    plt.ylabel(FIG[0][1][1])
    subplot_annotate(loc='ne',nr=0)
    #plt.subplot(2,2,i+1);i+=1
    for k in range(4):
        subplot(3,2,[3,4,5,6][k])
        if k<2: 
            plt.ylim([6,20]);
            plt.xlabel(FIG[0][2+k][0])
        else:plt.gca().set_xticklabels([])
        if k==0:
            print FIG[0][1+k]
            plt.ylabel(FIG[0][2+k][1])
        elif k>1: 
            plt.ylim([0.75,1])
        if k==2:plt.ylabel(FIG[0][2+k][1])
        if k%2==1: plt.gca().set_yticklabels([])
        
        errorbar(rtmc[:,k,:].T,x=range(1,5))
        subplot_annotate(loc='ne',nr=k+1)
    plt.subplots_adjust(wspace=0.01,hspace=-0.2)
    plt.savefig(figpath+FIG[0][0],dpi=DPI,bbox_inches='tight')

    figure(aspect=0.6)
    for vp in range(4):
        sel= ~np.isnan(rts[vp,:])
        d=np.int32(acc[vp,sel]==1)
        d=np.reshape(d,(d.size/10.,10))
        y=d.mean(1)
        x=np.arange(y.size)*10
        plt.plot(x+(vp-1.5)*1,y+(vp-1.5)*0.01,'+',ms=5,mew=1)
    plt.ylim([0.2,1.05])
    plt.gca().set_xticks(range(0,y.size*10,40))
    plt.grid(True,axis='x')
    plt.xlim([-4,250])
    plt.xlabel(FIG[1][1][0])
    plt.ylabel(FIG[1][1][1])
    leg=plt.legend(FIG[1][1][2],loc=0,numpoints=1,frameon=True,
                   ncol=4)#,fontsize='small',labelspacing=0)
    box=leg.get_frame()
    box.set_linewidth(0.)
    box.set_facecolor([0.9]*3)
    plt.savefig(figpath+FIG[1][0],dpi=DPI,bbox_inches='tight')
    plt.close('all')


def plotAnalysis(event=-1):
    plt.close()
    #lim=[[0.25,2.5],[0.37,2.5]]
    if event==0: lim=[[0.25,0.022],[0.37,0.022]]
    else: lim=[[0.25,0.025],[0.37,0.025]]
    for vp in range(1,5):
        plt.figure(0)
        if event>-1: sw=-400; ew=400
        else: sw=-800; ew=0
        hz=85.0 # start, end (in ms) and sampling frequency of the saved window
        fw=int(np.round((abs(sw)+ew)/1000.0*hz))
        bnR=np.arange(0,30,0.5)
        bnS=np.diff(np.pi*bnR**2)
        bnd= bnR+(bnR[1]-bnR[0])/2.0;bnd=bnd[:-1]
        path=inpath+'vp%03d/'%vp
        subplot(4,2,1)
        I=np.load(path+'E%d/agdens.npy'%event)
        I/=float(bnR.size)
        I*=100. # density in nr agents per 100 deg^2
        m=I.shape[2]/2
        plt.plot(bnd,I[0,:,m,0])
        if vp==4:
            #plt.plot(bnd,I[2,:,m,0])
            # show histogram
            x=np.concatenate([bnd,bnd[::-1]])
            ci=np.concatenate([I[2,:,m,2],I[2,:,m,1][::-1]])
            plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                        alpha=0.2,fill=True,fc='red',ec='red'))
        plt.xlabel(FIG[2][1][0])
        plt.ylabel(FIG[2][1][1])
        plt.legend(FIG[2][1][2],loc=1,fontsize=7)
        plt.grid();plt.ylim([0,lim[event][0]]);
        plt.xlim([0,14])

        subplot(4,2,2)
        plt.grid()
        x=np.linspace(sw,ew,I[0].shape[1])/1000.
        x=np.reshape(x,[x.size/2,2]).mean(1)
        hhh=0
        plt.plot(x,np.reshape(I[0,hhh,:,0],[x.size,2]).mean(1))
        if vp==4: plt.plot(x,np.reshape(I[2,hhh,:,0],[x.size,2]).mean(1))
        plt.ylim([0,lim[event][0]]);
        plt.xlim([-0.4,0.4])
        #plt.title('Radius=5 deg')
        plt.xlabel(FIG[2][2][0])
        plt.ylabel(FIG[2][2][1])#[a per deg^2]
        

        #direction change
        K=np.load(path+'E%d/dcK.npy'%event)
        nK=np.load(path+'E%d/dcnK.npy'%event)
        bn=np.arange(0,20,0.5)
        d= bn+(bn[1]-bn[0])/2.0;d=d[:-1]
        c=np.diff(np.pi*bn**2)
        # this code fragment shows the interaction
        #plt.figure(1)
        #plt.subplot(2,2,vp)
        #plt.imshow(np.nansum(K[0],2)/np.sum(nK[0],2),origin='lower',extent=[sw,ew,bn[0],bn[-1]],aspect=20,cmap='hot',vmax=11,vmin=2)     
        #plt.colorbar()
        ################
        plt.figure(0)
        subplot(4,2,3)
        hz=85.0
        mm=K[0].shape[1]/2
        plt.plot(d,np.nansum(K[0][:,mm,:],1)/ nK[0][:,mm,:].sum(1)*hz)
        if vp==4:# confidence band assuming iid binomial
            p=np.nansum(K[2][:,mm,:],1)/ nK[2][:,mm,:].sum(1)
            ci=1.96*np.sqrt(p*(1-p)/nK[2][:,mm,:].sum(1))
            l=(p-ci)*hz;h=(p+ci)*hz
            x=np.concatenate([d,d[::-1]])
            ci=np.concatenate([h,l[::-1]])
            plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                        alpha=0.2,fill=True,fc='red',ec='white'))
        plt.xlabel(FIG[2][3][0])  
        plt.ylabel(FIG[2][3][1])
        
        plt.xlim([0,14]);plt.grid();plt.ylim([0,12])
        subplot(4,2,4)
        x=np.linspace(sw,ew,K[0].shape[1])/1000.
        ss=2
        x=np.reshape(x,[x.size/ss,ss]).mean(1)
        kk=0
        y=np.nansum(K[0][kk,:,:],1)/ nK[0][kk,:,:].sum(1)*hz
        plt.plot(x,np.reshape(y,[x.size,ss]).mean(1))
        if vp==4:
            #plt.plot(x,np.nansum(K[2][kk,:,:],1)/ nK[2][kk,:,:].sum(1))
            p=np.nansum(K[2][kk,:,:],1)/nK[2][kk,:,:].sum(1)
            ci=1.96*np.sqrt(p*(1-p)/nK[2][kk,:,:].sum(1))
            l=(p-ci)*hz;h=(p+ci)*hz
            l=np.reshape(l,[x.size,ss]).mean(1);h=np.reshape(h,[x.size,ss]).mean(1)
            x=np.concatenate([x,x[::-1]])
            ci=np.concatenate([h,l[::-1]])
            
            plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                        alpha=0.2,fill=True,fc='red',ec='white'))
        plt.xlabel(FIG[2][4][0])  
        plt.ylabel(FIG[2][4][1])
        plt.xlim([-0.4,0.4])
        plt.ylim([0,12])
        #plt.title('Radius = 10 deg')
        plt.grid()

        hh=-1
        for chan in ['SOintensity','COmotion']:
            hh+=1
            K=np.load(path+'E%d/grd%s.npy'%(event,chan))
            I=np.load(path+'E%d/rad%s.npy'%(event,chan)).T
            if vp==4:IR=np.load(path+'E%d/radT%s.npy'%(event,chan)).T
                
            subplot(4,2,5+2*hh)
            plt.plot(np.arange(1,15),I[:,I.shape[1]/2])
            if vp==4: plt.plot(np.arange(1,15),IR[:,IR.shape[1]/2])
            plt.xlabel(FIG[2][5+hh][0])  
            plt.ylabel(FIG[2][5+hh][1])
            plt.ylim([0.008,lim[event][1]])
            
            plt.grid();plt.xlim([0,14])
            subplot(4,2,6+2*hh)
            x=np.linspace(sw,ew,I.shape[1])/1000.
            hhh=3
            #plt.plot(x,np.nanmean(I[:hhh,:],0))
            plt.plot(x,I[0,:])
            if vp==4: plt.plot(x,IR[0,:])
            plt.xlabel(FIG[2][7+hh][0])
            plt.ylabel(FIG[2][7+hh][1])
            plt.ylim([0.008,lim[event][1]])
            plt.xlim([-0.4,0.4])
            plt.grid()
        #plt.legend(['gaze','rand time','rand agent','rand pos'],loc=2)
        #plt.show()
    plt.savefig(figpath+FIG[2][0]+'E%d.png'%event,dpi=DPI,bbox_inches='tight')
    plt.close('all')

# PF pca

def _getPC(pf,h):
    if pf.shape[0]!=64:pf=pf[:,h].reshape((64,64,68))
    pf-= np.min(pf)
    pf/= np.max(pf)
    return np.rollaxis(pf.squeeze(),1).T

def plotCoeff(event,rows=8,cols=5,pcs=10):
    plt.close()
    panels=[];small=[]
    for vp in range(1,5):
        small.append([])
        path=inpath+'vp%03d/E%d/'%(vp,event)
        coeff=np.load(path+'X/coeff.npy')
        offset=8 # nr pixels for border padding
        R=np.ones((69,(64+offset)*rows,(64+offset)*cols),dtype=np.float32)
        for h in range(coeff.shape[1]):
            if h>=rows*cols:continue
            c= h%cols;r= h/cols
            s=((offset+64)*r+offset/2,(offset+64)*c+offset/2)
            pc= _getPC(coeff,h)
            if pc.mean()>=0.4: pc= 1-pc
            R[1:,s[0]:s[0]+64,s[1]:s[1]+64]= pc
            if h<pcs: small[-1].append(pc.T)
        panels.append(np.copy(R))
    plotGifGrid(small, fn=figpath+'Pixel/pcE%ds'%(event),tpL=True,
                duration=0.1,plottime=True,snapshot=1,bcgclr=1)   
    pad=20
    a,b,c=R.shape
    T=np.ones((a,(b+pad)*2,c*2+pad))
    T[:,pad:(b+pad),:c]=panels[0]
    T[:,pad:(b+pad),(c+pad):(2*c+pad)]=panels[1]
    T[:,(b+2*pad):(2*b+2*pad),:c]=panels[2]
    T[:,(b+2*pad):(2*b+2*pad),(c+pad):(2*c+pad)]=panels[3]
    labels=[]
    for i in range(4): labels.append(str2img('ABCD'[i],20))
    T[:,:labels[0].shape[0],:labels[0].shape[1]]-=labels[0]
    T[:,:labels[1].shape[0],(c+pad):(c+pad+labels[1].shape[1])]-=labels[1]
    T[:,(b+pad):(b+pad+labels[2].shape[0]),:labels[2].shape[1]]-=labels[2]
    T[:,(b+pad):(b+pad+labels[3].shape[0]),
      (c+pad):(c+pad+labels[3].shape[1])]-=labels[3]
    ndarray2gif(figpath+'Pixel/pcE%d'%(event),
                T,duration=0.1,plottime=True,snapshot=True)
def plotLatent():
    for ev in range(2):
        for vp in range(1,5):
            path=inpath+'vp%03d/E%d/X/'%(vp,ev)
            plt.subplot(1,2,ev+1)
            var=np.load(path+'latent.npy')
            plt.plot(range(1,21),var[:20])
            plt.xlabel('Principal components');plt.xlim([0,21])
            plt.ylabel('Proportion of explained variance')
            plt.grid(b=False,axis='x');plt.ylim([0,0.11])
            plt.gca().set_yticks(np.arange(0,0.12,0.01))
    plt.legend(range(1,5));print var.sum(),var[:20].sum(),var[:4].sum()
    plt.savefig(figpath+'Pixel/pcaVar.png',
                dpi=DPI,bbox_inches='tight')
def tabLatent(ev,pcs=5):
    dat=[]
    if ev==1: vps=range(1,5)+[999]
    else: vps=range(1,5)
    for vp in vps:
        path=inpath+'vp%03d/E%d/X/'%(vp,ev)
        dat.append(np.load(path+'latent.npy')[:pcs]*100)
    return ndarray2latextable(np.array(dat),decim=1)
        
def pcAddition():
    #BP S1
    out=[]
    for vp in [1,0]:
        path=inpath+'vp%03d/E%d/'%(vp+1,97)
        coeff=np.load(path+'X/coeff.npy')
        pc1=_getPC(coeff,0)
        if pc1.mean()>=0.4: pc1=1-pc1
        pc2=_getPC(coeff,1)
        if pc2.mean()>=0.4: pc2=1-pc2
        out.append([])
        out[-1].append(pc1)
        out[-1].append(pc2)
        out[-1].append((pc1-pc2+1)/2.)
        if False:
            out.append([])
            out[-1].append(pc1)
            out[-1].append(1-pc2)
            out[-1].append((pc1+pc2)/2.)
    plotGifGrid(out,fn=figpath+'Pixel/pcAddition',bcgclr=1,plottime=True)
             

def plotScore(vp,event,pcs=5,scs=3):
    plt.close()
    path=inpath+'vp%03d/E%d/'%(vp,event)
    score=np.load(path+'X/score.npy')
    coeff=np.load(path+'X/coeff.npy')
    r=np.corrcoef(score[:,:10].T)
    print np.round(r,2)
    print np.linalg.norm(coeff[:,:10],axis=0)
    #c=score[:,1]/score[:,0]*coeff[:,1].sum()/coeff[:,0].sum()
    #print 'vp%d, ev%d, r01=%.3f,c1/c0=%.3f'%(vp,event,r,np.median(c))
    offset=18 # nr pixels for border padding
    rows=scs*2+1; cols=pcs
    R=np.ones((69,(64+offset)*rows,(64+offset)*cols),dtype=np.float32)
    from pickle import load
    f=open(path+'PF.pars','r');dat=load(f);f.close()
    bd=score.shape[0]/dat['N']
    nss=[]
    for h in range(pcs):
        s=((offset+64)*scs+offset/2,(offset+64)*h+offset/2)
        pc=_getPC(coeff,h)
        print pc.mean()
        if pc.mean()>=0.4:
            R[1:,s[0]:s[0]+64,s[1]:s[1]+64]=1-pc
            ns=np.argsort(-score[:,h])[range(-scs,0)+range(scs)]
        else:
            R[1:,s[0]:s[0]+64,s[1]:s[1]+64]=pc
            ns=np.argsort(score[:,h])[range(-scs,0)+range(scs)]
        for i in range(len(ns)):
            pf=np.load(path+'PF/PF%03d.npy'%(ns[i]/bd))[ns[i]%bd,:,:,0,:]
            s=((offset+64)*(i+int(i>=scs))+offset/2,(offset+64)*h+offset/2)
            #print h,i,ns[i],ns[i]/bd,ns[i]%bd, s
            R[1:,s[0]:s[0]+64,s[1]:s[1]+64]= _getPC(np.float32(pf),h)
    ndarray2gif(figpath+'Pixel/scoreVp%de%d'%(vp,event),
                np.uint8(R*255),duration=0.1,plottime=True)

#########################################################
#                                                       #
#                  Button press                         #
#                                                       #
#########################################################
      
   

def plotBTmean(MAX=16):
    from matustools.matusplotlib import plotGifGrid
    dat=[]
    for vp in range(1,5):
        dat.append([])
        for event in range(-6,0):
            fn=inpath+'vp%03d/E%d/'%(vp,100+event)+'PF/PF000.npy'
            d=np.squeeze(np.load(fn))
            #print np.max(d.mean(axis=0)),np.min(d.mean(axis=0))
            dat[-1].append(d.mean(axis=0)/float(MAX))
    plotGifGrid(dat,fn=figpath+'buttonPressMean',bcgclr=1)
    return dat

def plotBTpt(vpn=range(1,5),pcaEv=97):
    from scipy.optimize import fmin
    def fun(x,D=None,verbose=False):
        nrlines=len(x)/3; p1=np.nan;s=0.15
        if nrlines==1: p0,v0,s0=tuple(x)
        elif nrlines==2: p0,v0,s0,p1,v1,s1=tuple(x)
        else: raise ValueError
        out=np.ones((P,T))
        dist=np.abs(pm+v0*tm-p0)/np.sqrt(1+v0**2)/s0
        out=np.maximum(1-np.power(dist,3),0)/float(nrlines)
        if nrlines==2:
            dist=np.abs(pm+v1*tm-p1)/np.sqrt(1+v1**2)/s1
            out+=np.maximum(1-np.power(dist,3),0)/float(nrlines)
        out/=out.sum()
        if D is None: return out
        fout=-np.corrcoef(D.flatten(),out.flatten())[0,1]# np.linalg.norm(D-out)**2
        if verbose: print 'p0=%.2f, v=%.2f, p1=%.2f, s=%.2f, f=%f'%(p0,v,p1,s,fout)
        return fout
    #dat=plotBTmean()
    T=68#dat[0][0].shape[-1]
    P=64#dat[0][0].shape[0]
    t=np.linspace(-0.8,0,T);p=np.linspace(-5,5,P)
    tm=np.repeat(t[np.newaxis,:],P,axis=0)
    pm=np.repeat(p[:,np.newaxis],T,axis=1)
    rows=len(vpn)
    #cols=len(dat[0])
    plt.figure(figsize=(6,6))
    #m=[-251,-201,-151,-101,-51,-1]
    bnds=[(1,None),(None,0),(None,None)]
    est=[]
    for i in range(rows):
        #for k in [1]:
        #est.append([])
        j=4# 200 ms 
        fn=inpath+'vp%03d/E%d/'%(vpn[i],pcaEv)+'X/coeff.npy'
        pc=_getPC(np.load(fn),0)
        if pc.mean()>=0.4: pc=1-pc
        D= pc.T[:,31:33,:].mean(1)
        #else: D=dat[i][j][31:33,:,:].mean(0)
        D/=D.sum()
        #print i,k,D.shape,D.sum()
        
        plt.subplot(2,2,i+1)
        plt.pcolor(p,t,D.T,cmap='gray')
        # below we set the initial guess 
        if vpn[i]==999: x0=np.array((3,-12,0.1,7,-12,0.1))
        elif vpn[i]==1: x0=np.array((1.3,-12,0.1))
        else: x0=np.array((3,-12,0.1,-2,-12,0.1))
        xopt=fmin(func=fun,x0=x0,args=(D,False))
        est.append(xopt.tolist())
        plt.plot(xopt[0]-xopt[1]*t,t,'r',lw=2,alpha=0.4)
        if vpn[i]!=1:
            plt.plot(xopt[3]-xopt[4]*t,t,'r',lw=2,alpha=0.4)
        else:est[-1].extend([np.nan]*3)
        plt.grid(True);
        plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]]);
        ax=plt.gca()
        #if i<2:ax.set_xticks([]);
        if i%2==0:ax.set_yticks([])
        else: ax.set_yticklabels(np.linspace(-1,-0.2,9))
        #else: plt.ylabel('subject %d'%(i+1))
        #if i==0: plt.title(str(m[j]*2+2))
    print figpath
    plt.savefig(figpath+FIG[3][0]+'%d'%pcaEv,
                dpi=DPI,bbox_inches='tight')
    est=np.array(est)
    print est.ndim, est
    if est.ndim>2: est=np.squeeze(est)
    print ndarray2latextable(est,decim=2)

#############################
#
#           TRACKING
#
#############################

def plotEvStats():
    plt.close()
    from Coord import initVP
    res=[]
    dat=np.zeros((3,30))
    plt.figure(figsize=(8,6))
    for vp in range(1,5):
        vp,ev,path=initVP(vp,0)
        si=np.load(path+'si.npy')
        out=[]
        for ev in range(6):
            out.append(np.sum(si[:,14]==ev))
        out.append(np.sum(si[:,14]>ev))
        out.append(np.sum(si[:,13]==1))
        out.extend([out[1]/float(out[0])*100, 100*out[-1]/float(out[1])])
        res.append(out)
        print np.max(si[:,14])
        for ev in range(dat.shape[1]):
            dat[0,ev]=np.logical_and(si[:,14]==ev,si[:,13]==1).sum()
            if ev: 
                dat[2,ev]=np.logical_and(si[:,14]==ev,si[:,13]==-1).sum()
                dat[1,ev]=(si[:,14]==ev).sum()-dat[0,ev]-dat[2,ev]
            else: 
                dat[1,ev]=(si[:,14]==1).sum()
                dat[2,ev]=(si[:,14]==ev).sum()-dat[1,ev]
                
            dat[:,ev]/=dat[:,ev].sum()
        plt.subplot(2,2,vp);plt.grid(False)
        for ev in range(20):
            a=dat[0,ev];b=dat[0,ev]+dat[1,ev]
            plt.bar(ev,a,color='#85dd7c',bottom=0,width=1)
            plt.bar(ev,b,color='#bcc4c4',bottom=a,width=1)
            plt.bar(ev,1,color='#FF9797',bottom=b,width=1)
        ax=plt.gca()
        if vp<3:
            ax.set_xticks(np.arange(20)+0.5)
            ax.set_xticklabels(np.arange(20))
        else: 
            ax.set_xticks([])
        if vp%2: ax.set_yticks([])
        else: ax.set_yticks([0,0.25,0.5,0.75,1])
        plt.ylim([0,1])
        for hh in [0.25,0.5,0.75]:plt.plot([0,20],[hh,hh],'k:')
    plt.savefig(figpath+FIG[4][0],dpi=DPI,bbox_inches='tight')
def si2tex():
    from matustools.matusplotlib import ndarray2latextable
    res=[]
    for vp in range(1,5):
        vp,ev,path=initVP(vp,0)
        si=np.load(path+'si.npy')
        out=[]
        for ev in range(6):
            out.append(np.sum(si[:,14]==ev))
        out.append(np.sum(si[:,14]>ev))
        out.append(np.sum(si[:,13]==1))
        out.extend([out[1]/float(out[0])*100, 100*out[-1]/float(out[1])])
        res.append(out)
    ndarray2latextable(np.array(res),8*[0]+[1,1])

def plotVel():
    plt.close()
    plt.figure(figsize=(4,3))
    vp,ev,path=initVP(4,1)
    tv=np.load(path+'trackVel.npy')
    T=tv.shape[2]
    for hh in range(1):
        t=[np.linspace(0,T*1000/85,T), 
           np.linspace(-T/2*1000/85,T/2*1000/85,T),
           np.linspace(-T*1000/85,0,T)][hh]
        for vp in range(4):
            #plt.subplot(3,1,hh+1)
            plt.plot(t,85*tv[vp,[0,2,1][hh],:])
            plt.ylim([10,13])
    plt.xlabel(FIG[5][1][0])
    plt.ylabel(FIG[5][1][1])
    plt.legend(FIG[5][1][2],loc=0,ncol=4)
    print FIG[5][0]
    plt.savefig(figpath+FIG[5][0],dpi=DPI,bbox_inches='tight')

def plotMeanPF():
    plt.close()
    vp,ev,path=initVP(4,1)
    D=np.load(path+'trackPF.npy')
    FFs=[]
    for vp in range(D.shape[0]): FFs.append([[],[],[]])
    Fs=[]
    for g in range(D.shape[2]):
        Fs=[];
        for vp in range(D.shape[0]):
            Fs.append([])
            
            for h in range(1,D.shape[1]):
                
                denom=[0.003,0.15,0.05,0.05][h]
                temp=D[vp,h,g,:,:,:]/denom
                Fs[-1].append(temp)
                temp[temp>1]=1
                if h==2:FFs[vp][[0,2,1][g]]=temp       
        from matustools.matusplotlib import plotGifGrid
        lbls=[['A',20,-10,95],['B',20,-10,235],['C',20,-10,370],#['D',20,-10,505],
              [FIG[6][1][3]+'1',20,65,-15],[FIG[6][1][3]+'2',20,200,-15],
              [FIG[6][1][3]+'3',20,340,-15],[FIG[6][1][3]+'4',20,475,-15]]
        plotGifGrid(Fs,fn=figpath+'Coord/'+['trackPFforw','trackPFback','trackPFmid'][g],
                    bcgclr=1,plottime=True,text=lbls,P=129,F=85)
    
    plotGifGrid(FFs,fn=figpath+'Coord/trackPF',bcgclr=1,forclr=0,plottime=True,P=129,F=85)
    plt.figure(figsize=(6,10))
    for g in range(D.shape[2]):
        for vp in range(4):
            d=D[vp,2,[0,2,1][g],60:68,:,:].mean(0)
            T=d.shape[1];P=d.shape[0]
            plt.subplot(4,3,g+3*vp+1)
            t=[np.linspace(-T*1000/85,0,T),
               np.linspace(-T/2*1000/85,T/2*1000/85,T),
               np.linspace(0,T*1000/85,T)][g]
            if not g:plt.ylabel(FIG[6][1][3]+str(vp+1),size=18)
            if False:  plt.gca().set_xticklabels([])
            if not vp:plt.title(FIG[6][1][2][g],size=18)
            p=np.linspace(-5,5,P)
            plt.pcolor(p,t,d.T,cmap='gray',vmax=0.05)
            plt.xlim([p[0],p[-1]])
            if g: plt.gca().set_yticklabels([])
            if g==1: plt.ylim([-500,500])
    plt.savefig(figpath+FIG[6][0],dpi=DPI,bbox_inches='tight')

#############################
#
#           SVM
#
#############################

def svmPlotExtrRep(event=0,plot=True,suf=''):
    from PercFields import initPath
    plt.close()
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
    if plot: plotGifGrid(dat,fn=figpath+'svm%sExtremaE%d'%(suf,event),
                         F=34,P=32,bcgclr=0.5)
    return dat

def svmPlotExtrema():
    from matustools.matusplotlib import plotGifGrid
    from PercFields import initPath
    plt.close()
    out=[[],[],[],[]]
    for nf in [[0,''],[1,''],[1,'3'],[1,'2']]:
        dat=svmPlotExtrRep(nf[0],suf=nf[1],plot=True)
        for vp in range(4):
            out[vp].extend([dat[vp][1],dat[vp][5]])
    path,inpath,figpath=initPath(1,0)
    txt=[['VP1',16,20,-16],['VP2',16,60,-16],['VP3',16,100,-16],['VP4',16,140,-16],
         ['A',16,-8,60],['B',16,-8,140],['C',16,-8,220],['D',16,-8,300]]
    plotGifGrid(out,fn=figpath+'svmExtrema',bcgclr=0.5,F=34,P=32,
                text=txt,duration=0.2,plottime=True,snapshot=True)  
        


if __name__=='__main__':
    #figures
    plotBehData()
    plotEvStats()
    plotBTpt()
    plotAnalysis(event=0)
    plotAnalysis(event=1)
    plotVel()
    plotMeanPF()
    from ReplayData import compareCoding
    compareCoding(vp=2,block=18,cids=[0,1,2,4])

    #movies
    pcAddition()
    plotBTmean()
    plotCoeff(97)
    plotCoeff(0)
    plotCoeff(1)
    svmPlotExtrema()

    from ReplayData import replayTrial
    #note the following trials will be displayed but not saved as movies
    replayTrial(vp =1,block=11,trial=7,tlag=0.0,coderid=4)#mov01
    replayTrial(vp =2,block=2,trial=19,tlag=0.0,coderid=4)#mov02
    replayTrial(vp =3,block=16,trial=12,tlag=0.0,coderid=4)#mov07
    replayTrial(vp =1,block=10,trial=9,tlag=0.0,coderid=4)#mov10
    replayTrial(vp =3,block=1,trial=12,tlag=0.0,coderid=4)#mov11
    replayTrial(vp =3,block=1,trial=7,tlag=0.0,coderid=4)#mov12
    #tables 
    si2tex()
    tabLatent(0,pcs=5)
    tabLatent(1,pcs=5)
    tabLatent(97,pcs=5)
    

    

