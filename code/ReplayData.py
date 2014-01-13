from Settings import Q
from Constants import *
from psychopy import visual, core, event,gui
from psychopy.misc import pix2deg
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
from evalETdata import tseries2eventlist, t2f, selectAgentTRACKING, manualDC
from copy import copy
import os
import pylab as plt
hclrs=[]
cm = plt.get_cmap('Paired')
for i in range(14):
    hclrs.append(np.array(cm(((i+7)%14)/float(14))[:-1])*2-1) 
hclrs.append([-1,-1,-1])
sclrs=['red','green','black']
#hclrs=[[-1,1,-1],[1,-1,1],[-1,1,1],[1,1,-1],[-1,-1,-1], [-1,1,-1],[1,-1,1],[-1,1,1],[1,1,-1],[-1,-1,-1],[-1,1,-1],[1,-1,1],[-1,1,1],[1,1,-1],[-1,-1,-1]]
KL=[['j','k','l','semicolon'],['q','w','e','r']]
PATH = os.getcwd().rstrip('code')+'evaluation'+os.path.sep+'coding'+os.path.sep

class Trajectory():
    def __init__(self,gazeData,maze=None,wind=None,
            highlightChase=False,phase=1,eyes=1,coderid=0):
        self.wind=wind
        self.phase=phase
        self.cond=gazeData.oldtraj.shape[1]
        self.pos=[]
        self.eyes=eyes
        self.behsel=None
        # determine common time intervals
        g=gazeData.getGaze(phase)
        ts=max(g[0,0], gazeData.fs[0,1])
        te=min(g[-1,0],gazeData.fs[-1,1])
        self.t=np.linspace(ts,te,int(round((te-ts)*Q.refreshRate/1000.0)))
        # put data together
        g=gazeData.getGaze(phase,hz=self.t)
        tr=gazeData.getTraj(hz=self.t)
        if eyes==1: g=g[:,[7,8]];g=np.array(g,ndmin=3)
        else: g=np.array([g[:,[1,2]],g[:,[4,5]]])
        
        g=np.rollaxis(g,0,2)
        
        self.pos=np.concatenate([tr,g],axis=1)
        try:
            if type(self.wind)==type(None):
                self.wind=Q.initDisplay()
            #if gazeData!=None:
            self.cond+=1
            if eyes==2: self.cond+=1
            clrs=np.ones((self.cond,3))
            clrs[-1,[0,1,2]]=-1
            if eyes==2: clrs[-2,[0,1,2]]=-1
            if highlightChase: clrs[0,[0,2]]=0; clrs[1,[0,2]]=-1
            #print clrs
            self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
                nElements=self.cond,sizes=Q.agentSize,colors=clrs,interpolate=False,
                colorSpace='rgb',elementMask='circle',elementTex=None)
            if type(maze)!=type(None):
                self.maze=maze
                self.maze.draw(wind)
        except:
            self.wind.close()
            raise
    def showFrame(self,positions):
        try:
            try:
                self.maze.draw(wind)
            except AttributeError: pass
            self.elem.setXYs(positions)
            self.elem.draw()
            self.wind.flip()
        except: 
            self.wind.close()
            raise
    
    def play(self,tlag=0):
        """
            shows the trial as given by TRAJECTORIES
        """
        try:
            self.wind.flip()
            playing=False
            #t0=core.getTime()
            step=1000/float(Q.refreshRate)
            position=np.zeros((self.cond,2))
            sel=0; self.f=0
            while True:#self.f<self.pos.shape[0]:
                self.f=min(self.f,self.pos.shape[0]-1)
                self.f=max(0,self.f)
                position=self.pos[self.f,:,:]
                if self.phase==1:
                    clrs=np.copy(self.elem.colors)
                    ags,clrr=self.highlightedAgents()
                    #print ags
                    for a in range(self.cond-self.eyes): 
                        if a in ags: clrs[a,:]=clrr[ags.index(a)]
                        else: clrs[a,:]=[1,1,1]
                    if not self.behsel is None and self.f==(self.pos.shape[0]-1) and self.behsel[0]>=0:
                        clrs[self.behsel,:]=[-1,1,-1]
                    self.elem.setColors(clrs)
##                elif self.phase==2:
##                    if sel<1 and f>self.gazeData.behdata[8]*Q.refreshRate:
##                        clrs=np.copy(self.elem.colors)
##                        clrs[int(self.gazeData.behdata[7]),:]=np.array([1,1,0])
##                        self.elem.setColors(clrs);sel+=1
##                    if sel<2 and f>self.gazeData.behdata[10]*Q.refreshRate:
##                        clrs=np.copy(self.elem.colors)
##                        clrs[int(self.gazeData.behdata[9]),:]=np.array([1,1,0])
##                        self.elem.setColors(clrs);sel+=1
                self.showFrame(position)
                if playing and tlag>0: core.wait(tlag)
                for key in event.getKeys():
                    if key in ['escape']:
                        #try: self.saveSelection()
                        #except: print 'selection could not be saved'
                        self.wind.close()
                        return
                        #core.quit()
                    #print key
                    if key=='space': playing= not playing
                    if key=='i': self.f=1
                    if key==KL[RH][0]: self.f=self.f-15
                    if key==KL[RH][1]: self.f=self.f-1
                    if key==KL[RH][2]: self.f=self.f+1
                    if key==KL[RH][3]: self.f=self.f+15
                    if key=='s': self.save=True
                if playing and self.f>=self.pos.shape[0]-1:  playing=False
                if not playing: core.wait(0.01)
                if playing: self.f+=2
            self.wind.flip()
            #print core.getTime() - t0
            self.wind.close()
        except: 
            self.wind.close()
            raise
    def highlightedAgents(self): return [],[]
        
class GazePoint(Trajectory):
    def __init__(self, gazeData,wind=None):
        self.gazeData=gazeData
        self.wind=wind
        self.pos=[]
        g=self.gazeData.getGaze()
        self.trialDur=g.shape[0]/self.gazeData.hz*1000
        step=1000/float(Q.refreshRate)
        tcur=np.arange(0,self.trialDur-3*step,step)
        self.t=tcur
        step=1000/float(gazeData.hz)
        t=np.arange(0,g.shape[0]*step,step)
        self.gaze=np.array((interpRange(t,g[:,1],tcur),
            interpRange(t,g[:,2],tcur)))
        try:
            if type(self.wind)==type(None):
                self.wind=Q.initDisplay()
            self.cond=1
            self.gazeDataRefresh=gazeData.hz
            clrs=np.ones((self.cond,3))
            self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
                nElements=self.cond,sizes=Q.agentSize,rgbs=clrs,
                elementMask='circle',elementTex=None)
        except:
            self.wind.close()
            raise

class ETReplay(Trajectory):
    def __init__(self,gazeData,**kwargs):
        wind = kwargs.get('wind',None)
        if wind is None: wind = Q.initDisplay((1280,1100))
        Trajectory.__init__(self,gazeData,wind=wind,**kwargs)
        self.gazeData=gazeData
        self.mouse = event.Mouse(True,None,self.wind)
        self.coderid = kwargs.get('coderid',0)
        try:
            indic=['Velocity','Acceleration','Saccade','Fixation','OL Pursuit','CL Pursuit','HEV','Tracking']
            self.lim=([0,450],[-42000,42000],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1])# limit of y axis
            self.span=(0.9,0.9,0.6,0.6,0.6,0.6,0.6,0.6)# height of the window taken by graph
            self.offset=(0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.2)
            fhandles=[self.gazeData.getVelocity,self.gazeData.getAcceleration,
                      self.gazeData.getSaccades, self.gazeData.getFixations,
                      self.gazeData.getOLP,self.gazeData.getCLP,
                      self.gazeData.getHEV,self.gazeData.getTracking]
            self.ws=30; self.sws=150.0 # selection window size
            ga=[7.8337, 18.7095,-13.3941+5,13.3941+5] # graph area
            self.ga=ga
            mid=ga[0]+(ga[1]-ga[0])/2.0
            inc=(ga[3]-ga[2])/float(len(indic));self.inc=inc
            self.spar=(0,-9.4,2) #parameters for selection tool, posx, posy, height
            self.apar=(0,-12.7,4.5) # parameters for agent selection tool
            frame=[visual.Line(self.wind,(ga[0],ga[3]),(ga[0],ga[2]),lineWidth=4.0),
                visual.Line(self.wind,(-ga[1],ga[3]),(ga[1],ga[3]),lineWidth=4.0),
                visual.Line(self.wind,(mid,ga[3]),(mid,ga[2]),lineWidth=2.0),
                visual.Line(self.wind,(ga[1],ga[3]),(ga[1],ga[2]),lineWidth=4.0),
                visual.Line(self.wind, (-ga[1],ga[2]),(ga[1],ga[2]),lineWidth=4.0),
                visual.Line(self.wind, (-ga[1],ga[2]),(-ga[1],ga[3]),lineWidth=4.0),
                visual.Rect(self.wind, width=ga[1]*2,height=self.spar[2], pos=(self.spar[0],self.spar[1]),lineWidth=4.0),
                visual.Rect(self.wind, width=ga[1]*2,height=self.apar[2], pos=(self.apar[0],self.apar[1]),lineWidth=4.0),
                visual.Line(self.wind,(0,self.spar[1]+self.spar[2]/2.0),(0,self.apar[1]-self.apar[2]/2.0),lineWidth=2.0)
                ]
            self.seltoolrect=frame[6]
            self.atoolrect=frame[7]
            self.graphs=[]
            self.selrects=[];self.sacrects=[]; self.arects=[]
            for i in range(15): self.selrects.append(visual.Rect(self.wind,
                height=self.spar[2],width=1,fillColor='red',opacity=0.5,lineColor='red'))
            for i in range(15): self.sacrects.append(visual.Rect(self.wind,
                height=self.spar[2]+self.apar[2],width=2,fillColor='blue',opacity=0.5,lineColor='blue'))
            for i in range(30): self.arects.append(visual.Rect(self.wind,
                height=self.apar[2],width=2,fillColor='green',opacity=0.5,lineColor='blue'))
            
            for f in range(len(indic)):      
                frame.append(visual.Line(self.wind,(ga[0],ga[3]-(f+1)*inc),
                    (ga[1],ga[3]-(f+1)*inc),lineWidth=4.0))
                frame.append(visual.TextStim(self.wind,indic[f],
                    pos=(ga[0]+0.1,ga[3]-0.1-f*inc),
                    alignHoriz='left',alignVert='top',height=0.5))
                self.graphs.append(visual.ShapeStim(self.wind,
                                closeShape=False,lineWidth=2.0))
                self.graphs[f].setAutoDraw(True)
                
            self.frame=visual.BufferImageStim(self.wind,stim=frame)
            self.tmsg=visual.TextStim(self.wind,color=(0.5,0.5,0.5),pos=(-13,-7.8))
            self.msg= visual.TextStim(self.wind,color=(0.5,0.5,0.5),
                pos=(0,-7.8),text=' ',wrapWidth=20)
            self.msg.setAutoDraw(True)
            self.gData=[]
            for g in range(len(indic)):
                yOld=fhandles[g](self.phase,hz=self.t)
                self.gData.append(yOld)
            self.sev=[]
            scale=Q.refreshRate/self.gazeData.hz
            for gs in self.gazeData.sev:
                s=int(np.round(scale*gs[0]))
                e=min(len(self.t)-1,max(s+1, int(np.round(scale*gs[1]))))
                self.sev.append([s,e,gs[0],gs[1]])
            for gs in self.gazeData.bev:
                s=int(np.round(scale*gs[0]))
                e=min(len(self.t)-1,max(s+1, int(np.round(scale*gs[1]))))
                self.sev.append([s,e,gs[0],gs[1]])
            self.selected=[[]]
            try:
                for tr in self.gazeData.track:
                    s=int(np.round(scale*tr[0]))
                    e=min(len(self.t)-1,max(s+1, int(np.round(scale*tr[1]))))
                    self.selected[0].append([self.t[s],s,tr[0],self.t[e],e,tr[1],tr[2],[],False])
                    for a in tr[3]:
                        s=int(np.round(scale*a[0]))
                        e=min(len(self.t)-1,max(s+1, int(np.round(scale*a[1]))))
                        self.selected[0][-1][7].append([self.t[s],s,a[0],self.t[e],e,a[1],[-1,-1,-1]])
            except AttributeError: print 'Tracking events not available'
            #self.selected=[Coder.loadSelection(self.gazeData.vp,
            #        self.gazeData.block,self.gazeData.trial,prefix= 'track/coder1/')]
            self.pos[:,:,0]-=6 # shift agents locations on the screen
            self.pos[:,:,1]+=5
            self.wind.flip()
            self.released=False # mouse key flag
            self.save=False # flag for save selection tool data
        except:
            self.wind.close()
            raise
    def showFrame(self,positions):
        fs=max(0,self.f-self.ws)
        fe=min(self.f+self.ws,self.gData[0].shape[0]-1)
        #xveldata=np.array(self.t[fs:fe], ndmin=2)
        unit = (self.ga[1]-self.ga[0])/float(self.ws)/2.0
        step=(2*self.ws-(fe-fs))*unit
        if fs<self.ws: s=self.ga[0]+step
        else: s=self.ga[0]
        if fe>self.gData[0].size-self.ws: e=self.ga[1]-step
        else: e=self.ga[1]
        xveldata=np.array(np.linspace(s,e,fe-fs),ndmin=2)
        for g in range(len(self.graphs)):
            yveldata=self.gData[g][fs:fe]
            yveldata=(yveldata-self.lim[g][0])/float(self.lim[g][1]-self.lim[g][0])
            yveldata[yveldata>self.lim[g][1]]=self.lim[g][1]
            yveldata[yveldata<self.lim[g][0]]=self.lim[g][0]
            yveldata=np.array((self.ga[3]-(g+1)*self.inc)
                +(self.span[g]*yveldata+self.offset[g])*self.inc,ndmin=2)
            veldata=np.concatenate((xveldata,yveldata),axis=0).T.tolist()
            self.graphs[g].setVertices(veldata)
        rct=self.gazeData.recTime # update time message
        self.tmsg.setText('t%d Time %d:%02d:%06.3f' % (self.gazeData.trial,rct.hour,
                rct.minute+ (rct.second+int(self.t[self.f]/1000.0))/60,
                np.mod(rct.second+ self.t[self.f]/1000.0,60)))
#        for m in self.gazeData.msgs:
#            if m[0]>self.t[self.f] and m[0]<self.t[self.f]+100:
#                m.append(True)
#                self.msg.setText(m[2])
#                self.msg.draw()
        self.frame.draw()
        self.tmsg.draw()
        Trajectory.showFrame(self,positions)
    def highlightedAgents(self): return self.gazeData.getAgent(self.t[self.f]),[]
    
class Coder(ETReplay):
    def showFrame(self,positions):
        # draw saccades and blinks in the selection tool
        sws=np.float(self.sws); 
        s=max(self.f-sws/2.0,0);
        e= min(self.f+sws/2.0,self.gData[0].shape[0]-1)
        i=0; 
        for k in range(len(self.sev)):
            sac=self.sev[k]
            if (sac[1]<=e and sac[1]>=s) or (sac[0]<=e and sac[0]>=s):
                ss=(max(sac[0],s)-self.f)/sws*self.ga[1]*2;
                ee=(min(sac[1],e)-self.f)/sws*self.ga[1]*2;
                posy=self.spar[1]-self.apar[2]/2.0
                self.sacrects[i].setPos(( (ee-ss)/2.0+ss,posy))
                self.sacrects[i].setWidth(max(ee-ss,0.5))
                self.sacrects[i].setAutoDraw(True)
                self.sacrects[i].ad=k
                i+=1;
        while i<len(self.sacrects):
            self.sacrects[i].setAutoDraw(False);
            self.sacrects[i].ad=-1;i+=1;
        # draw selected blocks
        i=0;h=-1;
        tot=float(len(self.selected)); offset=np.linspace(-0.5,0.5,tot+1)[:-1]+0.5/tot
        for selection in self.selected:
            h+=1
            for k in range(len(selection)):
                sel= selection[k]
                trigger=False
                if len(sel)==3 and self.t[int(s)]<=sel[0] and self.t[int(e)]>=sel[0] :
                    self.selrects[i].setPos(((sel[1]-self.f)/sws*self.ga[1]*2,self.spar[1]+offset[h]*self.spar[2]))
                    self.selrects[i].setWidth(1);trigger=True
                elif (len(sel)>=6 and( self.t[int(s)]<=sel[3] and self.t[int(e)]>=sel[3]
                    or self.t[int(s)]<=sel[0] and self.t[int(e)]>=sel[0]
                    or self.t[int(s)]>=sel[0] and self.t[int(e)]<=sel[3])):
                    ss=(max(sel[1],s)-self.f)/sws*self.ga[1]*2
                    ee= (min(sel[4],e)-self.f)/sws*self.ga[1]*2
                    self.selrects[i].setPos(((ee-ss)/2.0 + ss,self.spar[1]+offset[h]*self.spar[2]))
                    self.selrects[i].setWidth(ee-ss);trigger=True
                if trigger:
                    self.selrects[i].ad=k;self.selrects[i].h=h
                    self.selrects[i].setFillColor(sclrs[h])
                    self.selrects[i].setAutoDraw(True);i+=1
        while i<len(self.selrects):
            self.selrects[i].setAutoDraw(False);
            self.selrects[i].ad=-1;i+=1;
        i=0;k=0;
        for sell in self.selected[0]:
            if len(sell)>6:
                h=0
                for sel in sell[7]:
                    if ( self.t[int(s)]<=sel[3] and self.t[int(e)]>=sel[3]
                    or self.t[int(s)]<=sel[0] and self.t[int(e)]>=sel[0]
                    or self.t[int(s)]>=sel[0] and self.t[int(e)]<=sel[3]):
                        if h==len(hclrs):continue
                        ss=(max(sel[1],s)-self.f)/sws*self.ga[1]*2
                        ee= (min(sel[4],e)-self.f)/sws*self.ga[1]*2
                        w=self.apar[2]/float(len(sell[7]))
                        self.arects[i].setPos(((ee-ss)/2.0 + ss,
                                self.apar[1]+self.apar[2]/2.0-(h+0.5)*w))
                        self.arects[i].setWidth(ee-ss);
                        self.arects[i].setHeight(w);
                        try:
                            self.arects[i].setFillColor(hclrs[sell[6][h]],colorSpace='rgb')
                        except:
                            print h, sell[6][h], len(hclrs)
                            raise
                        self.arects[i].ad=k;self.arects[i].ad2=h
                        self.arects[i].setAutoDraw(True);i+=1
                        sel[-1]=hclrs[sell[6][h]]
                    h+=1
            k+=1
        while i<len(self.arects):
            self.arects[i].setAutoDraw(False);
            self.arects[i].ad=-1;self.arects[i].ad2=-1;i+=1;
                        
                    
        ETReplay.showFrame(self,positions)
        # query mouse
        mkey=self.mouse.getPressed();select=False; selectA=False
        if 0<sum(mkey) and self.released:
            mpos=self.mouse.getPos();issac=False
            mkey=self.mouse.getPressed()
            for sr in self.sacrects:
                if sr.ad>-1 and sr.contains(mpos):# saccade is drawn and contains mouse
                    g=self.gazeData.getGaze()
                    if ((len(self.selected[0])>0 and len(self.selected[0][-1])==3)
                        and self.seltoolrect.contains(mpos) or
                        (self.atoolrect.contains(mpos) and  mkey[RH*2]>0)):
                        ff= self.sev[sr.ad][0]
                        gf=self.sev[sr.ad][2]
                        tt=g[gf,0]
                    else:
                        ff= self.sev[sr.ad][1]
                        gf=self.sev[sr.ad][3]
                        tt=g[gf,0]
                    if self.seltoolrect.contains(mpos) and mkey[0]>0: select=True
                    else: selectA=True
            if (not select and self.seltoolrect.contains(mpos) and mkey[0]>0
                or not selectA and self.atoolrect.contains(mpos)):
                ff=np.round(mpos[0]/self.ga[1]/2.0*sws)+self.f;
                tt=self.t[min(ff,self.t.size-1)]
                gf=np.round(ff/Q.refreshRate*self.gazeData.hz)
            if mkey[0]>0:
                if self.seltoolrect.contains(mpos):
                    if  (len(self.selected[0])==0 or len(self.selected[0][-1])>=6):
                        self.selected[0].append([tt,ff,gf])
                        self.msg.setText('Selection Open: %d'%self.selected[0][-1][0])
                    elif tt>self.selected[0][-1][0]:
                        self.selected[0][-1].extend([tt,ff,gf])
                        ags,tms=selectAgentTRACKING(self.selected[0][-1][2], self.selected[0][-1][5],self.gazeData.events )
                        tmsnew=[];scale=Q.refreshRate/self.gazeData.hz
                        for a in tms:
                            s=int(np.round(scale*a[0]))
                            e=min(len(self.t)-1,max(s+1, int(np.round(scale*a[1]))))
                            tmsnew.append([self.t[s],s,a[0],self.t[e],e,a[1],1])
                        self.selected[0][-1].extend([ags,tmsnew,True])
                        self.msg.setText('Selection Closed: %d,%d'%(self.selected[0][-1][0], self.selected[0][-1][3]))
            else:
                for sr in self.selrects:
                    if sr.ad>-1 and  sr.contains(mpos):
                        if  sr.h==0:
                            self.selected[0].pop(sr.ad)
                            self.msg.setText('Selection Deleted')
                        elif sr.h==1:  self.selected[0].append(self.selected[1][sr.ad])
                        elif sr.h==2:  self.selected[0].append(self.selected[2][sr.ad])
            if self.atoolrect.contains(mpos):
                pos=[mpos[0],self.spar[1]]
                tar=None
                for sr in self.selrects:
                    if sr.contains(pos):
                        for ar in self.arects:
                            if ar.ad==sr.ad and sr.ad>-1:
                                if abs(ar.pos[1]-mpos[1])<ar.height/2.0:
                                    assert tar==None
                                    tar=ar
                if tar!=None:
                    if mkey[RH*2]>0: self.selected[0][tar.ad][7][tar.ad2][3:6]=[tt,ff,gf]
                    else: self.selected[0][tar.ad][7][tar.ad2][:3]=[tt,ff,gf]
                    self.msg.setText('New Agent Timing: %d'%tt)          
            # agent selection
            for a in range(positions.shape[0]):
                dist=((positions[a,0]-mpos[0])**2+(positions[a,1]-mpos[1])**2)**0.5
                if dist<Q.agentSize/2.0: self.highlightAgent(a)
            self.released=False
        if 0==sum(mkey) and not self.released: self.released=True
        if self.save: self.saveSelection()
    def highlightedAgents(self):
        res=[];clr=[]
        for sel in self.selected[0]:
            if len(sel)>6:
                for k in range(len(sel[7])):
                    if sel[7][k][1]<=self.f and sel[7][k][4]>=self.f:
                        res.append(sel[6][k])
                        clr.append(sel[7][k][-1])
        return res,clr
    def highlightAgent(self,a):
        for sel in self.selected[0]:
            #print 'a', sel[6],sel[7]
            if len(sel)>6 and a in sel[6]:
                sell=sel[7][sel[6].index(a)]
                if sell[1]<=self.f and sell[4]>=self.f:
                    sel[7].pop(sel[6].index(a))
                    sel[6].remove(a)
            elif len(sel)>=6 and sel[1]<=self.f and sel[4]>=self.f:
                sel[6].append(a)
                sel[7].append(sel[:6])
                sel[7][-1].append([-1,-1,-1])
            #print sel[6],sel[7]
    @staticmethod
    def loadSelectionOld(vp,block,trial,prefix='track/'):
        #print prefix+'vp%03db%dtr%02d.trc'%(vp,block,trial)
        fin = open(prefix+'vp%03db%dtr%02d.trc'%(vp,block,trial),'r')
        out=[]
        for line in fin:
            line=line.rstrip('\n')
            els= line.rsplit(' ')
            els=np.int32(els).tolist()
            out.append(els[:6])
            out[-1].append(els[6:-1])
            out[-1].append([els[:6]]*len(out[-1][-1]))
            out[-1].append(els[-1])
        return out
    @staticmethod
    def loadSelection(vp,block,trial,coder=1,prefix=''):
        fname = PATH+prefix+os.path.sep+'coder%d'%coder+os.path.sep+'vp%03db%dtr%02d.trc'%(vp,block,trial)
        fin = open(fname,'r')
        out=[]
        for line in fin:
            line=line.rstrip('\n')
            els= line.rsplit(' ')
            els=np.int32(els).tolist()
            out.append(els[:6])
            out[-1].extend([[],[]])
            for a in range(len(els[6:-1])/7):
                out[-1][-2].append(els[6+a])
            for a in range(len(els[6:-1])/7):
                i=5+len(els[6:-1])/7+a*6
                out[-1][-1].append(els[(i+1):(i+7)])
            out[-1].append(els[-1])
        return out
        
        
    def saveSelection(self,coderid=None):
        """ take care that refreshrate settings remain the
            same between saving and loading """
        if coderid is None: coderid=self.coderid
        fout = open(PATH+'coder%d'% coderid +os.path.sep+'vp%03db%dtr%02d.trc'%(self.gazeData.vp,
                self.gazeData.block,self.gazeData.trial),'w')
        for sel in self.selected[0]:
            fout.write('%d %d %d %d %d %d'%tuple(sel[:6]))
            assert len(sel[6])==len(sel[7])
            for a in range(len(sel[6])):
                fout.write(' %d'%sel[6][a])
            for a in range(len(sel[6])):
                for k in sel[7][a][:-1]:
                    fout.write(' %d'%int(k))
            fout.write(' %d'%sel[8])
            fout.write('\n')
        
        if coderid!=0: self.msg.setText('Selection Saved')
        self.save=False

def replayTrial(vp,block,trial,tlag=0,coderid=0):
    from readETData import readEyelink
    data=readEyelink(vp,block)
    trl=data[trial]
    trl.extractBasicEvents()
    trl.driftCorrection(jump=manualDC(vp,block,trial))
    trl.extractComplexEvents()
    trl.importComplexEvents(coderid=coderid)
    R=Coder(gazeData=trl,phase=1,eyes=1,coderid=coderid)
    #R.saveSelection(coderid=0)
    R.play(tlag=tlag)
    
def replayBlock(vp,block,trial,tlag=0,coderid=0):
    behdata=np.loadtxt(os.getcwd().rstrip('code')+'behavioralOutput/vp%03d.res'%vp)
    trialStart=trial
    from readETData import readEyelink
    data=readEyelink(vp,block)
    #PATH+='coder%d'% coderid +os.path.sep
    for trial in range(trialStart,len(data)):
        print vp,block,trial
        trl=data[trial]
        trl.extractBasicEvents()
        trl.driftCorrection(jump=manualDC(vp,block,trial))
        trl.extractComplexEvents()
        trl.importComplexEvents(coderid=coderid)
    for trial in range(trialStart,len(data)):
        R=Coder(gazeData=data[trial],phase=1,eyes=1,coderid=coderid)
        bi=np.logical_and(block==np.int32(behdata[:,1]),trial==np.int32(behdata[:,2])).nonzero()[0][0]
        R.behsel= np.int32(behdata[bi,[-3,-5]])
        #R.saveSelection(coderid=8)
        #R.wind.close()
        R.play(tlag=tlag)
# coding verification routines       
def compareCoding(vp,block,cids=[0,1,2]):
    import pylab as plt
    import matplotlib as mpl
    plt.figure(figsize=(20,60))
    ax=plt.gca(); N=len(cids)
    for trial in range(40):
        for k in range(N):
            try: D=Coder.loadSelection(vp,block,trial,coder=cids[k])
            except IOError: 
                print trial, cids[k]
                continue
            for e in D:
                r=mpl.patches.Rectangle((e[0]/1000.0,trial+k/float(N)),
                    (e[3]-e[0])/1000.0,1/float(N),ec='w',fc='gray',alpha=0.1)
                ax.add_patch(r)
                ags=e[6];ts=e[7]
                for a in range(len(ags)):
                    r=mpl.patches.Rectangle((ts[a][0]/1000.0,trial+(k+a/float(len(ags)))/float(N)),
                        (ts[a][3]-ts[a][0])/1000.0,1/float(N)/float(len(ags)),color=(1+hclrs[ags[a]])/2.0,alpha=0.5)
                    
                    ax.add_patch(r)
    plt.xlim([0,30])
    ax.set_yticks(range(40))
    ax.set_xticks(range(30))
    #ax.set_yticklabels(['pc','anna','matus']*40)
    for x in range(0,30,5):plt.plot([x,x],[0,40],'gray',lw=1)
    for y in range(40):plt.plot([0,30],[y,y],'k',lw=2)
    for y in range(40*len(cids)):plt.plot([0,30],[y/float(len(cids)),y/float(len(cids))],'gray',lw=1)
    plt.ylim([0,40])
    plt.grid()
    plt.savefig(PATH+'comparison'+os.path.sep+'vp%db%d.jpg'%(vp,block),format='jpg',dpi=100,bbox_inches='tight')
    #plt.show()
    
def missingTEfiles():
    vp=1
    for b in range(1,22):
        for t in range(40):
            for c in range(2):
                try:
                    dat=Coder.loadSelection(vp,b,t,coder=c+1)
                except IOError:
                    print vp,b,t,c+1
            
if __name__ == '__main__':
    RH=0 # set right handed or left handed layout
    replayTrial(vp = 1,block = 12,trial=37,tlag=0.,coderid=2)
    #compareCoding(block=11,vp=1,cids=[0,8])
    #missingTEfiles()

