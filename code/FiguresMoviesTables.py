# -*- coding: utf-8 -*-
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
from scipy import stats
from matustools.matusplotlib import *
from Coord import initVP
import os
from scipy.optimize import fmin

LINEWIDTH=5.74 # peerj line width in inches
DPI=600
FMT='.avi' # movie export format
FIG=(('behdata.png',('Detection Time','Probability' ),
                     ('Subject','Detection Time'),
                     ('Subject','Detection Time'),
                     ('Subject','Proportion Correct'),
                      ('Subject','Proportion Correct')),
     ('bdtime.png',('Trial','Proportion Correct',['S1','S2','S3','S4'])),
     ('Coord/agdens',('Distance to Saccade Target in Degrees','Agent Density',['S1','S2','S3','S4','RT']),
      ('Time to Saccade Onset in Seconds','Agent Density'),
      ('Distance to Saccade Target in Degrees','Direction Changes'),
      ('Time to Saccade Onset in Seconds','Direction Changes'),
      ('Distance to Saccade Target in Degrees', 'Contrast Saliency'),
      ('Distance to Saccade Target in Degrees', 'Motion Saliency'),
      ('Time to Saccade Onset in Seconds', 'Contrast Saliency'),
      ('Time to Saccade Onset in Seconds', 'Motion Saliency')),
     ('Pixel/buttonPress','Time to button press in seconds','Distance in degrees','S',-15,30,72),
     ('Coord/trackEv','ES CS1           ...           CS9           ...           CS19','S'),
     ('Coord/trackVel',('Time in seconds','Velocity in deg/s',['S1','S2','S3','S4'])),
     ('Coord/trackPFrail',('Start','Middle','End'),'S',-15),
     ('S',-15,'CS1','ES')
     )

inpath = os.getcwd().rstrip('code')+'evaluation'+os.path.sep
figpath = os.getcwd().rstrip('code')+'figures'+os.path.sep

##############################################
# behavioral data
def plotBehData():
    plt.close()
    from matustools.matustats import lognorm
    rtmc=np.load(inpath+'rtmc.npy')
    rejs=np.load(inpath+'rejs.npy')
    rts=np.load(inpath+'rts.npy')
    acc=np.load(inpath+'acc.npy')
    figure(size=3,aspect=0.6)
    ax = plt.subplot2grid((2,3), (0, 0), colspan=2)
    #subplot(3,1,1)
    formatAxes(ax)
    #ax=plt.gca()
    vp=0
    d=rts[vp,rejs[vp,:]==1]
    d=d[d>0]
    x=np.linspace(1,30,30)
    hist(d,bins=x,normed=True)
    ax.plot(x-0.5,lognorm(mu=rtmc[vp,-2,:].mean(),
                sigma=rtmc[vp,-1,:].mean()).pdf(x-0.5),'k')
    plt.xlabel(FIG[0][1][0])
    plt.ylabel(FIG[0][1][1])
    subplotAnnotate(loc='ne',nr=0)
    #plt.subplot(2,2,i+1);i+=1
    for k in range(4):
        subplot(2,3,[4,5,3,6][k])
        
        
        plt.xlabel(FIG[0][2+k][0])
        #else:plt.gca().set_xticklabels([])
        #if k==0:
        #    print FIG[0][1+k]
        plt.ylabel(FIG[0][2+k][1])
        #elif k>1: 
        if k==1 or k==0: plt.ylim([6,20])
        else: plt.ylim([0.75,1])
        #if k==2:plt.ylabel(FIG[0][2+k][1])
        #if k%2==1: plt.gca().set_yticklabels([])
        
        errorbar(rtmc[:,k,:].T,x=range(1,5))
        subplotAnnotate(loc='ne',nr=k+1)
    plt.subplots_adjust(wspace=0.01,hspace=-0.2)
    plt.savefig(figpath+FIG[0][0],dpi=DPI,bbox_inches='tight')

    figure(size=2,aspect=0.4)
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

def tabSampleSize():
    res=[]
    for vp in range(1,5):
        res.append([])
        for ev in [97,0,1]:   
            path=inpath+'vp%03d/E%d/'%(vp,ev)
            sc=np.load(path+'X/score.npy')
            res[-1].append(sc.shape[0])
        ti=np.load(inpath+'vp%03d/'%vp+'ti.npy')
        res[-1].append(ti.shape[0])
    res=np.array(res)
    ndarray2latextable(res,decim=0)
    
#######################################
# saccades
def plotAnalysis(event=-1,clr='gray'):
    '''event - id of the catch-up saccade, 0=exploration saccade '''
    plt.close()
    #lim=[[0.25,2.5],[0.37,2.5]]
    if event==0: lim=[[0.25,0.022],[0.4,0.022,],[0.4,0.03]]
    else: lim=[[0.25,0.025],[0.4,0.025],[0.4,0.03]]
    figure(size=3,aspect=0.99)
    for vp in range(1,5):
        
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
                        alpha=0.2,fill=True,fc=clr,ec=clr))
            subplotAnnotate(loc='ne')
        #plt.xlabel(FIG[2][1][0])
        plt.ylabel(FIG[2][1][1])
        leg=plt.legend(FIG[2][1][2],loc=9,fontsize=7,frameon=True,ncol=2)
        box=leg.get_frame()
        box.set_linewidth(0.)
        box.set_facecolor([0.9]*3)
        plt.grid(True);plt.ylim([0,lim[event][0]]);
        plt.gca().set_yticks(np.linspace(0,lim[event][0],6))
        plt.xlim([0,14])

        subplot(4,2,2);plt.grid(True)
        x=np.linspace(sw,ew,I[0].shape[1])/1000.
        x=np.reshape(x,[x.size/2,2]).mean(1)
        hhh=0
        plt.plot(x,np.reshape(I[0,hhh,:,0],[x.size,2]).mean(1))
        if vp==4:
            xx=np.concatenate([x,x[::-1]])
            ci=np.concatenate([np.reshape(I[2,hhh,:,2],[x.size,2]).mean(1),
                np.reshape(I[2,hhh,:,1],[x.size,2]).mean(1)[::-1]])
            print xx.shape,ci.shape
            plt.gca().add_patch(plt.Polygon(np.array([xx,ci]).T,
                        alpha=0.2,fill=True,fc=clr,ec=clr))
            #plt.plot(xx,np.reshape(I[2,hhh,:,0],[xx.size,2]).mean(1))
            subplotAnnotate(loc='ne')
        plt.ylim([0,lim[event][0]]);
        plt.gca().set_yticks(np.linspace(0,lim[event][0],6))
        plt.xlim([-0.4,0.4])
        #plt.title('Radius=5 deg')
        #plt.xlabel(FIG[2][2][0])
        #plt.ylabel(FIG[2][2][1])#[a per deg^2]
        plt.gca().set_yticklabels([])
        

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
        #plt.figure(0)
        ################
        
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
                        alpha=0.2,fill=True,fc=clr,ec='w'))
            subplotAnnotate(loc='ne')
        #plt.xlabel(FIG[2][3][0])  
        plt.ylabel(FIG[2][3][1])
        
        plt.xlim([0,14]);plt.grid(True);plt.ylim([0,12])
        plt.yticks(range(0,13,3))
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
                        alpha=0.2,fill=True,fc=clr,ec='w'))
            subplotAnnotate(loc='ne')
        #plt.xlabel(FIG[2][4][0])  
        #plt.ylabel(FIG[2][4][1])
        plt.gca().set_yticklabels([])
        plt.xlim([-0.4,0.4]);plt.grid(True)
        plt.ylim([0,12]);plt.yticks(range(0,13,3))
        #plt.title('Radius = 10 deg')

        hh=-1
        for chan in ['SOintensity','COmotion']:
            hh+=1
            K=np.load(path+'E%d/grd%s.npy'%(event,chan))
            I=np.load(path+'E%d/rad%s.npy'%(event,chan)).T
            if vp==4:IR=np.load(path+'E%d/radT%s.npy'%(event,chan))
            subplot(4,2,5+2*hh)
            plt.plot(np.arange(1,15),I[:,I.shape[1]/2])
            if vp==4:
                x=np.concatenate([np.arange(1,15),np.arange(14,0,-1)])
                IR=np.sort(IR,axis=0)
                m=IR.shape[1]/2
                ci=np.concatenate([IR[2,m,:],IR[97,m,:][::-1]])
                plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                        alpha=0.2,fill=True,fc=clr,ec='w'))
                #plt.plot(np.arange(1,15),IR[[49,50],m,:].mean(0))
                subplotAnnotate(loc='ne')
            if chan=='COmotion': plt.xlabel(FIG[2][5+hh][0])  
            plt.ylabel(FIG[2][5+hh][1])
            plt.ylim([0.008,lim[event][1]]);plt.grid(True)
            plt.gca().set_yticks(np.linspace(0.008,lim[event][1],6))
            plt.xlim([0,14])
            subplot(4,2,6+2*hh)
            x=np.linspace(sw,ew,I.shape[1])/1000.
            hhh=3
            #plt.plot(x,np.nanmean(I[:hhh,:],0))
            plt.plot(x,I[0,:])
            if vp==4:
                xx=np.concatenate([x,x[::-1]])
                ci=np.concatenate([IR[2,:,0],IR[97,:,0][::-1]])
                plt.gca().add_patch(plt.Polygon(np.array([xx,ci]).T,
                    alpha=0.2,fill=True,fc=clr,ec='w'))
                subplotAnnotate(loc='ne')
            if chan=='COmotion':plt.xlabel(FIG[2][7+hh][0])
            #plt.ylabel(FIG[2][7+hh][1])
            plt.gca().set_yticklabels([])
            plt.ylim([0.008,lim[event][1]])
            plt.gca().set_yticks(np.linspace(0.008,lim[event][1],6))
            plt.xlim([-0.4,0.4]);plt.grid(True)
    plt.savefig(figpath+FIG[2][0]+'E%d.png'%event,dpi=DPI,bbox_inches='tight')
    plt.close('all')

# PF pca

def _getPC(pf,h):
    if pf.shape[0]!=64:pf=pf[:,h].reshape((64,64,68))
    pf-= np.min(pf)
    pf/= np.max(pf)
    return np.rollaxis(pf.squeeze(),1).T

def plotCoeff(event,rows=8,cols=5,pcs=10):
    '''event - id of the catch-up saccade, 0=exploration saccade
        rows - nr rows in the movie
        cols - nr columns in the movie
        pcs - number of principal components that will be displayed'''
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
    lbl=[]
    for i in range(4):lbl.append([FIG[6][1][3]+str(i+1),20,32+i*72,FIG[6][1][5]])
    for i in range(10):lbl.append([FIG[6][1][4]+str(i+1),20,-10,FIG[6][1][6]+i*FIG[6][1][7]])
    lbl[-1][3]-=10
    plotGifGrid(small, fn=figpath+'Pixel/pcE%ds'%(event)+FMT,tpL=True,
                duration=0.1,text=lbl,plottime=True,snapshot=1,bcgclr=1)   
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
    ndarray2gif(figpath+'Pixel/pcE%d'%(event)+FMT,
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
    '''pcs - number of principal components that will be displayed'''
    dat=[]
    if ev==1: vps=range(1,5)+[999]
    else: vps=range(1,5)
    for vp in vps:
        path=inpath+'vp%03d/E%d/X/'%(vp,ev)
        dat.append(np.load(path+'latent.npy')[:pcs]*100)
    return ndarray2latextable(np.array(dat),decim=1)

def plotPC1rail():
    T=68;P=64
    t=np.linspace(-0.8,0.8,T);p=np.linspace(-5,5,P)
    tm=np.repeat(t[np.newaxis,:],P,axis=0)
    pm=np.repeat(p[:,np.newaxis],T,axis=1)
    fig=figure(size=3,aspect=0.7)
    fig.tight_layout()
    bnds=[(1,None),(None,0),(None,None)];est=[]
    for ev in [0,1]:
        for vp in range(1,5):
            fn=inpath+'vp%03d/E%d/'%(vp,ev)+'X/coeff.npy'
            pc=_getPC(np.load(fn),0)
            if pc.mean()>=0.4: pc=1-pc
            inc=0
            D= pc.T[:,(31+inc):(33+inc),:].mean(1)
            D/=D.sum();
            subplot(2,4,ev*4+vp)
            plt.pcolor(p,t,D.T,cmap='gray')
            # below we set the initial guess 
            x0=np.array((5,-12,0.2))
            xopt=fmin(func=_fun,x0=x0,args=(D,t,p,False))
            est.append(xopt.tolist())
            plt.plot(xopt[0]-xopt[1]*t,t,'r',lw=1,alpha=0.4)
            plt.grid(True,'both');
            plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]]);
            ax=plt.gca();ax.set_axisbelow(False)
            ax.set_xticks([-4,-2,0,2,4])
            ax.set_yticks(np.linspace(-0.8,0.8,5))
            if not ev: ax.set_xticklabels([])
            if vp>1: ax.set_yticklabels([])
            else:
                ax.set_yticklabels(np.linspace(-0.8,0.8,5))
                plt.ylabel(['ES','CS1'][ev]+'\nTime to saccade in sec.')
            #if i==1: plt.text(2,-1,FIG[3][2],size=8)
            #if i==1: plt.xlabel(FIG[3][2])
            #else: plt.ylabel('subject %d'%(i+1))
            if not ev: plt.title(FIG[3][3]+str(vp))
    print FIG[2][1][0]
    plt.subplot(2,4,6);plt.text(-9,-1.15,FIG[2][1][0])
    plt.subplots_adjust(wspace=-1)
    plt.savefig(figpath+'Pixel'+os.path.sep+'pcSacRail',
                dpi=DPI,bbox_inches='tight')
    est=np.array(est)
    print est.ndim, est
    if est.ndim>2: est=np.squeeze(est)
    est[:,1]*=(t[-1]-t[0])/0.8
    print ndarray2latextable(est,decim=2)
    print np.corrcoef(est[:,1],est[:,2])[0,1]
def pcAddition(MOVIE=True):
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
        #out[-1].append((pc1+pc2)/2.)
        if False:
            out.append([])
            out[-1].append(pc1)
            out[-1].append(1-pc2)
            out[-1].append((pc1+pc2)/2.)
    if MOVIE:
        plotGifGrid(out,fn=figpath+'Pixel/pcAddition'+FMT,bcgclr=1,snapshot=2,
                plottime=True,text=[['A',20,12,-10],['B',20,84,-10]])
    bla
    print out[0][0].shape
    cols=5;fs=np.linspace(0,out[0][0].shape[0]-1,cols)
    ps=np.arange(out[0][0].shape[1])
    for i in range(len(out)):
        for j in range(len(out[0])):
            for fi in range(cols):
                plt.subplot(3,cols,j*cols+fi+1)
                plt.pcolor(ps,ps,out[i][j][fs[fi],:,:],cmap='gray')
                #plt.grid()
    plt.savefig(figpath+'Pixel'+os.path.sep+'pcAddition',
                dpi=DPI,bbox_inches='tight')
                
            
             

def plotScore(vp,event,pcs=5,scs=3):
    ''' vp - subject id
        event - id of the catch-up saccade, 0=exploration saccade
        pcs - number of principal components that will be displayed
        scs - number of score samples, total nr of rows is scs*2+1'''
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
    ndarray2gif(figpath+'Pixel/scoreVp%de%d'%(vp,event)+FMT,
                np.uint8(R*255),duration=0.1,plottime=True,snapshot=True)
def plotPCvp(vp,ev,t=1,pcs=5):
    '''t - 0=start,1=middle,2=end '''
    plt.close()
    path=inpath+'vp%03d/E%d/'%(vp,ev)
    coeff=np.load(path+'X/coeff.npy')
    for h in range(pcs):
        ax=plt.subplot(1,pcs,h+1)
        ax.set_xticks([]);ax.set_yticks([]);
        for sp in ['left','right','bottom','top']:
            ax.spines[sp].set_visible(False)
        pc=_getPC(coeff,h)
        if pc.mean()>=0.4: pc=1-pc
        pc=pc[[0,pc.shape[0]/2,pc.shape[0]][t],:,:]
        plt.imshow(pc,cmap='gray')
        plt.grid(False);
        plt.title('PC'+str(h+1))
    plt.savefig(figpath+'Pixel/PCvp%dev%d'%(vp,ev),dpi=DPI,bbox_inches='tight')  

def _fun(x,D=None,t=None,p=None,verbose=False):
    ''' objective function for fitting lines to rail diagrams'''
    T=t.size;P=p.size
    tm=np.repeat(t[np.newaxis,:],P,axis=0)
    pm=np.repeat(p[:,np.newaxis],T,axis=1)
    nrlines=len(x)/3; p1=np.nan;
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
    if verbose: print 'p0=%.2f, v=%.2f, p1=%.2f, s=%.2f, f=%f'%(p0,v,p1,s1,s2,fout)
    return fout

#########################################################
# button press
def plotBTavg(MAX=16,MOVIE=True):
    from matustools.matusplotlib import plotGifGrid
    dat=[];T=68;P=64;est=[]
    t=np.linspace(-0.8,0,T);p=np.linspace(-5,5,P)
    figure(size=3,aspect=1)
    for vp in range(1,5):
        dat.append([])
        for event in range(-6,0):
            fn=inpath+'vp%03d/E%d/'%(vp,100+event)+'PF/PF000.npy'
            d=np.squeeze(np.load(fn))
            #print np.max(d.mean(axis=0)),np.min(d.mean(axis=0))
            dat[-1].append(d.mean(axis=0)/float(MAX))

            inc=0#[-2,2,0,0,0,0][i+1]
            D= (d.mean(axis=0)/float(MAX))
            D=np.rollaxis(D,1)
            D=D[:,(31+inc):(33+inc),:].mean(1);D/=D.sum();
            j=vp-1; i=event+6
            subplot(6,4,i*4+j+1)
            plt.pcolor(p,t,D.T,cmap='gray')
            # below we set the initial guess
            if i==3:
                if vp==1: x0=np.array((0.5,-12,0.3))
                else: x0=np.array((3,-12,0.2,-0.5,-12,0.2))
                xopt=fmin(func=_fun,x0=x0,args=(D,t,p,False))
                est.append(xopt.tolist())
                plt.plot(xopt[0]-xopt[1]*t,t,'r',lw=1,alpha=0.4)
                if vp!=1: plt.plot(xopt[3]-xopt[4]*t,t,'r',lw=1,alpha=0.4)
                else:est[-1].extend([np.nan]*3)
            plt.grid(True,'both');
            plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]]);
            ax=plt.gca();ax.set_axisbelow(False)
            ax.set_xticks([-4,-2,0,2,4])
            ytck=np.linspace(-0.8,0,5)[1:]
            ax.set_yticks(ytck)
            if j>0: ax.set_yticklabels([])
            else: ax.set_yticklabels(ytck-[0.5,0.4,0.3,0.2,0.1,0.05][i])
            #if i==1: plt.text(2,-1,FIG[3][2],size=8)
            if i<5: ax.set_xticklabels([])
            #else: plt.ylabel('subject %d'%(i+1))
            #if i==0: plt.title(str(m[j]*2+2))
            #
            if i==0: plt.title(FIG[3][3]+str(j+1))
    plt.subplot(6,4,5)
    plt.text(-9,-0.5,FIG[3][1],rotation='vertical')
    plt.subplot(6,4,22)
    plt.text(-2,-1.2,FIG[3][2])
    plt.subplots_adjust(wspace=-1)
    plt.savefig(figpath+FIG[3][0]+'avg',dpi=DPI,bbox_inches='tight')
    est=np.array(est)
    print est.ndim, est
    if est.ndim>2: est=np.squeeze(est)
    print ndarray2latextable(est,decim=2)
    if not MOVIE: return dat
    lbl=[]    
    #for i in range(4):lbl.append([FIG[6][1][3]+str(i+1),20,32+i*72,-15])
    for i in range(4):lbl.append([FIG[3][3]+str(i+1),20,32+i*72,FIG[3][4]])
    #for i in range(6):lbl.append([str([500,400,300,200,100,50][i]),20,-10,30+i*72])
    for i in range(6):lbl.append([str([500,400,300,200,100,50][i]),20,-10,FIG[3][5]+i*FIG[3][6]])
    plotGifGrid(dat,fn=figpath+'buttonPressMean'+FMT,bcgclr=1,
                text=lbl,plottime=True)
    return dat

def railIdeal():
    T=68; P=64
    t=np.linspace(-0.8,0,T);p=np.linspace(-5,5,P)
    fn=inpath+'vp%03d/E%d/X/coeff.npy'%(999,1)
    pc=_getPC(np.load(fn),0)
    if pc.mean()>=0.4: pc=1-pc
    D= pc.T[:,31:33,:].mean(1)
    D/=D.sum();
    x0=np.array((3,-12,0.1,7,-12,0.1))
    xopt=fmin(func=_fun,x0=x0,args=(D,t,p,False)).tolist()
    xopt.append(abs(xopt[0]-xopt[3]))
    ndarray2latextable(np.array(xopt,ndmin=2),decim=2)

def plotBTpc(vpn=range(1,5),pcaEv=97):
    ''' vpn - list with subject ids
        pcaEv - id of the catch-up saccade, 0=exploration saccade,
            97 - 200 ms before button press'''
    #dat=plotBTavg(MOVIE=False)
    T=68#dat[0][0].shape[-1]
    P=64#dat[0][0].shape[0]
    t=np.linspace(-0.8,0,T);p=np.linspace(-5,5,P)
    tm=np.repeat(t[np.newaxis,:],P,axis=0)
    pm=np.repeat(p[:,np.newaxis],T,axis=1)
    rows=len(vpn)
    #cols=len(dat[0])
    fig=figure(size=2,aspect=0.75)
    fig.tight_layout()
    #m=[-251,-201,-151,-101,-51,-1]
    bnds=[(1,None),(None,0),(None,None)]
    est=[]
    for i in range(rows+2):
        #for k in [1]:
        #est.append([])
        #j=4# 200 ms 
        fn=inpath+'vp%03d/E%d/'%(vpn[max(0,i-2)],pcaEv)+'X/coeff.npy'
        if i==1: pc=(_getPC(np.load(fn),0)+_getPC(np.load(fn),1))/2.
        elif i==0: pc=(_getPC(np.load(fn),0)-_getPC(np.load(fn),1)+1)/2.
        else: pc=_getPC(np.load(fn),0)
        if pc.mean()>=0.4: pc=1-pc
        inc=[-2,2,0,0,0,0][i]
        D= pc.T[:,(31+inc):(33+inc),:].mean(1)
        D/=D.sum();
        subplot(2,3,i+1)
        plt.pcolor(p,t,D.T,cmap='gray')
        # below we set the initial guess 
        if i==0: x0=np.array((1,-12,0.3,-2,-12,0.1))
        elif i==1:x0=np.array((3,-12,0.1,0,-12,0.1))
        elif i==2: x0=np.array((0,-12,0.1))
        else: x0=np.array((3,-12,0.1,-2,-12,0.1))
        xopt=fmin(func=_fun,x0=x0,args=(D,t,p,False))
        est.append(xopt.tolist())
        plt.plot(xopt[0]-xopt[1]*t,t,'r',lw=1,alpha=0.4)
        if x0.size==6: plt.plot(xopt[3]-xopt[4]*t,t,'r',lw=1,alpha=0.4)
        elif x0.size==3:est[-1].extend([np.nan]*3)
        else: raise ValueError
        est[-1].append(abs(est[-1][0]-est[-1][3]))
        plt.grid(True,'both');
        plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]]);
        ax=plt.gca();ax.set_axisbelow(False)
        ax.set_xticks([-4,-2,0,2,4])
        ax.set_yticks(np.linspace(-0.8,0,5))
        if i%3!=0: ax.set_yticklabels([])
        else:
            ax.set_yticklabels(np.linspace(-1,-0.2,5))
        if i<3: ax.set_xticklabels([])
        if i==0: plt.text(-9,-0.3,FIG[3][1],size=8,rotation='vertical')
        if i==4: plt.xlabel(FIG[3][2])
        #else: plt.ylabel('subject %d'%(i+1))
        #if i==0: plt.title(str(m[j]*2+2))
        plt.text(2.1,-0.75,FIG[3][3]+str(vpn[max(i-2,0)])+['a','b',''][min(i,2)],color='w')
    plt.subplots_adjust(wspace=-1)
    plt.savefig(figpath+FIG[3][0]+'%d'%pcaEv,
                dpi=DPI,bbox_inches='tight')
    est=np.array(est)
    #for k in [2,5]: est[:,k]=est[:,k]/np.sin(np.arctan(1/-est[:,k-1]))
    print est.shape
    if est.ndim>2: est=np.squeeze(est)
    print ndarray2latextable(est,decim=2)


#############################
# smooth eye movements
def plotEvStats():
    clrs=['#85dd7c','#bcc4c4','#FF9797']
    clrs=['#333333','#999999','#CCCCCC']
    plt.close()
    from Coord import initVP
    res=[]
    dat=np.zeros((3,30))
    figure(size=3,aspect=0.6)
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
        subplot(2,2,vp);plt.grid(False)
        for ev in range(20):
            a=dat[0,ev];b=dat[0,ev]+dat[1,ev]
            plt.bar(ev,a,color=clrs[0],bottom=0,width=1)
            plt.bar(ev,b,color=clrs[1],bottom=a,width=1)
            plt.bar(ev,1,color=clrs[2],bottom=b,width=1)
        ax=plt.gca()
        #if vp<3:
        #ax.set_xticks(np.arange(0,20,1)+0.5)
        plt.xlabel(FIG[4][1])
        #ax.set_xticklabels(np.arange(0,20,1))
        #else: 
        ax.set_xticks([])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        plt.ylim([0,1])
        for hh in [0.25,0.5,0.75]:plt.plot([0,20],[hh,hh],'gray',lw=0.5)
        plt.text(0.15,0.8,FIG[4][2]+str(vp),color='k')
    plt.subplots_adjust(wspace=0.01,hspace=-0.2)
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
    figure(size=2,aspect=0.6)
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

def plotTrack(MOVIE=True):
    '''MOVIE - if true creates movie output'''
    plt.close()
    vp,ev,path=initVP(4,1)
    D=np.load(path+'trackPF.npy')
    count=np.load(path+'trackPFcount.npy')
    
    FFs=[]
    for vp in range(D.shape[0]): FFs.append([[],[],[]])
    Fs=[]
    for g in range(D.shape[2]):
        Fs=[];
        for vp in range(D.shape[0]):
            Fs.append([])
            temp=np.float32(np.max(count[vp,:,g,:],1))
            if not g:print 'vp=%d, nrags=%d, prop=%.3f'%(vp+1,2,temp[2]/temp[1:].sum())
            for h in range(1,D.shape[1]):
                denom=[0.003,0.15,0.05,0.05][h]
                temp=D[vp,h,g,:,:,:]/denom
                Fs[-1].append(temp)
                temp[temp>1]=1
                if h==2:FFs[vp][[0,2,1][g]]=temp       
        lbl=[]
        for i in range(4):lbl.append([FIG[6][2]+str(i+1),20,65+i*137,FIG[6][3]])
        for i in range(3):lbl.append([['1','2','>2'][i],20,-10,85+i*135])      
        if MOVIE: plotGifGrid(Fs,fn=figpath+'Coord/'+['trackPFforw',
                    'trackPFback','trackPFmid'][g]+FMT,
                    bcgclr=1,plottime=True,text=lbl,P=129,F=85)
    lbl=[]
    for i in range(4):lbl.append([FIG[6][2]+str(i+1),20,65+i*137,FIG[6][3]])
    lbl.extend([[FIG[6][1][0],20,-10,40],[FIG[6][1][1],20,-10,190],[FIG[6][1][2],20,-10,330]])
    if MOVIE: plotGifGrid(FFs,fn=figpath+'Coord/trackPF'+FMT,bcgclr=1,
                forclr=0,plottime=True,P=129,F=85,text=lbl)
    figure(size=3,aspect=1)
    for g in range(D.shape[2]):
        for vp in range(4):
            d=D[vp,2,[0,2,1][g],60:68,:,:].mean(0)
            T=d.shape[1];P=d.shape[0]
            ax=subplot(4,3,g+3*vp+1)
            t=[np.linspace(-T*1000/85,0,T),
               np.linspace(-T/2*1000/85,T/2*1000/85,T),
               np.linspace(0,T*1000/85,T)][g]
            if not g:plt.ylabel(FIG[6][2]+str(vp+1),size=18)
            if not vp:plt.title(FIG[6][1][g],size=18)
            p=np.linspace(-5,5,P)
            plt.pcolor(p,t,d.T,cmap='gray',vmax=0.05)
            plt.xlim([p[0],p[-1]])
            #if g: ax.set_yticklabels([])
            if vp<3: ax.set_xticklabels([])
            if g==1:
                plt.ylim([-500,500])
                ax.set_yticks([-500,-250,0,250,500])
            elif g==0: ax.set_yticks([-1000,-750,-500,-250,0])
            else: ax.set_yticks([0,250,500,750,1000])
            ax.set_axisbelow(False);plt.grid(True)
            ax.set_xticks([-5,-2.5,0,2.5,5])
    plt.savefig(figpath+FIG[6][0],dpi=DPI,bbox_inches='tight')

    for vp in range(D.shape[0]):
        for h in range(1,D.shape[1]):
            denom=[0.1,0.05,0.05][h-1]
            d=D[vp,h,0,60:68,:,:].mean(0)/denom
            d[d>1]=1
            d[np.isnan(d)]=0 # happens when no samples are available
            T=d.shape[1];P=d.shape[0]
            print vp,h ,np.min(d),np.max(d)
            ax=subplot(4,3,h+3*vp)
            t=np.linspace(0,T*1/85.,T)
            if not g:plt.ylabel(FIG[6][2]+str(vp+1),size=18)
            if not vp:plt.title(['1','2','>2'][h-1],size=18)
            p=np.linspace(-5,5,P)
            plt.pcolor(p,t,d.T,cmap='gray')
            plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]])
            if h>1: ax.set_yticklabels([])
            if vp<3: ax.set_xticklabels([])
            ax.set_yticks([0,0.250,0.500,0.750,1.])
            ax.set_axisbelow(False);plt.grid(True)
            ax.set_xticks([[-5],[]][h>1]+[-2.5,0,2.5,5])
    plt.savefig(figpath+'Coord/trackPFforw',dpi=DPI,bbox_inches='tight')

#############################
# SVM
def svmPlotExtrRep(event=0,plot=True,suf=''):
    from Pixel import initPath
    if plot: plt.close()
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
    lbl=[]
    for i in range(4):lbl.append([FIG[7][0]+str(i+1),20,18+i*40,FIG[7][1]])
    lbl.append([FIG[7][2],20,-10,70]);lbl.append([FIG[7][3],20,-10,245])
    if plot: plotGifGrid(dat,fn=figpath+'svm%sExtremaE%d'%(suf,event)+FMT,
                         F=34,P=32,text=lbl,bcgclr=0.5)
    return dat

def svmPlotExtrema():
    from matustools.matusplotlib import plotGifGrid
    from Pixel import initPath
    plt.close()
    out=[[],[],[],[]]
    for nf in [[0,''],[1,''],[1,'3'],[1,'2']]:
        dat=svmPlotExtrRep(nf[0],suf=nf[1],plot=True)
        for vp in range(4):
            out[vp].extend([dat[vp][1],dat[vp][5]])
    path,inpath,figpath=initPath(1,0)
    plotGifGrid(out,fn=figpath+'svmExtrema'+FMT,bcgclr=0.5,F=34,P=32,
                duration=0.2,plottime=True,snapshot=True)

def svmComparison():
    from Pixel import initPath
    figure(size=3,aspect=0.35)
    svm=svmPlotExtrRep(0,plot=False)
    for vp in range(1,5):
        est=np.zeros((2,3))
        ax=subplot(1,4,vp)
        for ev in [0,1]:
            path,inpath,figpath=initPath(vp,ev)
            coeff=np.load(inpath+'X/coeff.npy')
            pc=_getPC(coeff,0)
            if pc.mean()>=0.4: pc=1-pc
            t=np.linspace(-0.4,0.4,pc.shape[0])
            p=np.linspace(-5,5,pc.shape[1])
            D= pc.T[:,31:33,:].mean(1)
            D/=D.sum();
            # below we set the initial guess 
            x0=np.array((0,-12,0.1))
            xopt=fmin(func=_fun,x0=x0,args=(D,t,p,False))
            est[ev,:]=xopt
            plt.plot(xopt[0]-xopt[1]*t,t,['g','m'][ev],lw=1,alpha=1)

        #plt.plot(est[:,0].mean()-est[:,1].mean()*t,t,'r',lw=1,alpha=0.4)
        dat=svm[vp-1][1][15:17,:,:].mean(0)
        t=np.linspace(-0.4,0.4,dat.shape[1])
        p=np.linspace(-5,5,dat.shape[0])
        print dat.shape
        plt.xlim([p[0],p[-1]]);plt.ylim([t[0],t[-1]]);
        plt.pcolor(p,t,dat.T,cmap='gray');
        ax.set_axisbelow(False);plt.grid(True);
        ax.set_yticks([-0.4,-0.2,0,0.2,0.4])
        if vp>1: ax.set_yticklabels([])
        plt.text(-3.9,0.21,'S'+str(vp),color='w',fontsize=16)
    leg=plt.legend(['ES','CS1'],loc=4,frameon=True)
    box=leg.get_frame()
    box.set_linewidth(0.)
    box.set_facecolor([0.9]*3)
    plt.subplot(1,4,1);plt.text(8,-0.57,FIG[2][1][0],fontsize=8)
    plt.ylabel('Time to Saccade in Sec.')
    plt.savefig(figpath+'svmComparison',dpi=DPI,bbox_inches='tight')
    


if __name__=='__main__':
    #plotScore(999,1,pcs=4,scs=0)
    #plotTrack(MOVIE=False)
    #plotPCvp(999,1)
    #plotBTpc()
    #tabLatent(97,pcs=5)
    #plotPC1rail()
    plotPCvp(999,1)
    bla
    
    
    # to create figures run
    plotPC1rail()
    plotBehData()
    plotPC1rail()
    svmComparison()
    plotBTpc()
    plotTrack(MOVIE=False)
    plotVel()
    plotPCvp(999,1)
    
    from ReplayData import compareCoding
    compareCoding(vp=2,block=18,cids=[0,1,2,4])

    # to create movies run
    pcAddition()
    plotBTmean()
    plotCoeff(97)
    plotCoeff(0)
    plotCoeff(1)
    plotTrack()
    svmPlotExtrema()
    plotScore(999,1,pcs=4,scs=0)
    
    from ReplayData import replayTrial
    #note the following trials will be displayed but not saved as movies
    replayTrial(vp =1,block=11,trial=7,tlag=0.0,coderid=4)#mov01
    replayTrial(vp =2,block=2,trial=19,tlag=0.0,coderid=4)#mov02
    replayTrial(vp =1,block=10,trial=9,tlag=0.0,coderid=4)#mov10
    replayTrial(vp =3,block=1,trial=12,tlag=0.0,coderid=4)#mov11
    replayTrial(vp =3,block=1,trial=7,tlag=0.0,coderid=4)#mov12
    # to create tables run
    si2tex()
    tabLatent(0,pcs=5)
    tabLatent(1,pcs=5)
    tabLatent(97,pcs=5)

    
    

    

