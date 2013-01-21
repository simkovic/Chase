from Settings import *
from Constants import *
from psychopy import visual, core, event,gui
from psychopy.misc import pix2deg
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

def interpRange(xold,yold,xnew):
    ynew=xnew.copy()
    f=interp1d(xold,yold)
    for i in range(xnew.size):
        ynew[i] =f(xnew[i])
    return ynew  

class Trajectory():

    
    def __init__(self,trajectories,maze=None,wind=None,trajRefresh=75.0,
                 gazeData=None,highlightChase=False,trialDur=30000.0,
                 phase=1,eyes=1):
        
        self.trajectories=trajectories
        self.trajRefresh=trajRefresh
        self.gazeData=gazeData
        g=self.gazeData.getGaze(phase)
        self.wind=wind
        self.phase=phase
        self.cond=trajectories.shape[1]
        self.pos=[]
        self.tstart=g[0,0]
        self.tend=g[-1,0]
        self.eyes=eyes
        # subsample trajectory data
        stepm=1000/float(Q.refreshRate)
        tm=np.arange(self.tstart,self.tend,stepm)
        stept=1000/float(trajRefresh)
        tt=np.arange(0,(trajectories.shape[0])*stept,stept)
        self.t=tm
        #print tcur.shape,t.shape,tcur[-1],t[-1], trajectories.shape 
        for a in range(self.cond):
            if self.phase==2:
                i=int(np.round((self.gazeData.getGaze()[-1,0]+self.tstart)/stept))
                self.pos.append( (trajectories[i,a,0]*np.ones(tm.shape), trajectories[i,a,1]*np.ones(tm.shape)))
            else:
                self.pos.append((interpRange(tt,trajectories[:,a,0],tm),
                    interpRange(tt,trajectories[:,a,1],tm)))
        self.pos=np.array(self.pos)
        #print self.pos.shape, self.tend
        #print 'pos ', self.pos.shape
        # subsample gaze data
        #stepg=1000/float(gazeData.hz)
        #tg=np.arange(0,self.trialDur+stepg,stepg)
        #print g[-1,0], tm[-1], g[0,0],tm[0]
        if eyes==1:
            self.gaze=np.array((interpRange(g[:,0],g[:,1],tm),
                interpRange(g[:,0],g[:,2],tm)))
        else:
            g=self.gazeData.gaze[phase][:,[0,1,2]]
            self.gaze=np.array((interpRange(g[:,0],g[:,1],tm),
                interpRange(g[:,0],g[:,2],tm)))
            g=self.gazeData.gaze[phase][:,[0,4,5]]
            self.gaze2=np.array((interpRange(g[:,0],g[:,1],tm),
                interpRange(g[:,0],g[:,2],tm)))
        try:
            if type(self.wind)==type(None):
                self.wind=Q.initDisplay()
            #if gazeData!=None:
            self.cond+=1
            if eyes==2: self.cond+=1
            clrs=np.ones((self.cond,3))
            clrs[-1,[0,1]]=0
            if eyes==2: clrs[-2,[0,1]]=0
            if highlightChase: clrs[0,[0,2]]=0; clrs[1,[1,2]]=0
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
    
    def play(self,tlag=0.02):
        """
            shows the trial as given by TRAJECTORIES
        """
        try:
            self.wind.flip()
            playing=False
            #t0=core.getTime()
            step=1000/float(Q.refreshRate)
            position=np.zeros((self.cond,2))
            sel=0; f=0
            while f<self.gaze.shape[1]:
                #print f, type(f),type(self.pos)
                if position.shape[0]>1:
                    if self.eyes==1:position[:-1,:]=self.pos[:,:,f]
                    else:position[:-2,:]=self.pos[:,:,f]
                if self.eyes==1:
                    position[-1,:]=self.gaze[:,f].transpose()
                else:
                    position[-2,:]=self.gaze[:,f].transpose()
                    position[-1,:]=self.gaze2[:,f].transpose()
                if self.phase==1:
                    clrs=np.copy(self.elem.colors)
                    ags=self.gazeData.getAgent(self.t[f])
                    #print self.t[f], ags
                    for a in range(self.pos.shape[0]): 
                        if a in ags: clrs[a,:]=[0,1,0]
                        else: clrs[a,:]=[1,1,1]
                    self.elem.setColors(clrs)
                elif self.phase==2:
                    if sel<1 and f>self.gazeData.behdata[8]*Q.refreshRate:
                        clrs=np.copy(self.elem.colors)
                        clrs[int(self.gazeData.behdata[7]),:]=np.array([1,1,0])
                        self.elem.setColors(clrs);sel+=1
                    if sel<2 and f>self.gazeData.behdata[10]*Q.refreshRate:
                        clrs=np.copy(self.elem.colors)
                        clrs[int(self.gazeData.behdata[9]),:]=np.array([1,1,0])
                        self.elem.setColors(clrs);sel+=1
                
                self.f=f
                #print f,position[:,0]
                self.showFrame(position)
                if playing and tlag>0: core.wait(tlag)
                for key in event.getKeys():
                    if key in ['escape']:
                        self.wind.close()
                        return
                        #core.quit()
                    if key=='q': playing= not playing
                    if key=='o': f+=1
                    if key=='i': f=max(0,f-1)
                    if key=='p': f+=10
                    if key=='u': f=max(0,f-10)
                if not playing: core.wait(0.01)
                if playing: f+=1
            self.wind.flip()
            #print core.getTime() - t0
            self.wind.close()
        except: 
            self.wind.close()
            raise
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
    
    
class BabyETData(Trajectory, GazePoint):
    def __init__(self,trajectories,gazeData,**kwargs):
        wind = kwargs.get('wind',None)
        if wind is None:
            wind = Q.initDisplay((1280,1000))
        if trajectories!=None:
            TrajectoryData.__init__(self,trajectories,
                wind=wind,gazeData=gazeData,**kwargs)
        else: GazePoint.__init__(self, gazeData,wind=wind)
    def showFrame(self,positions):
        
        TrajectoryData.showFrame(self,positions)
        
        
class ETReplay(Trajectory,GazePoint):
    def __init__(self,trajectories,gazeData,**kwargs):
        wind = kwargs.get('wind',None)
        if wind is None:
            wind = Q.initDisplay((1280,1000))
        if trajectories!=None:
            Trajectory.__init__(self,trajectories,
                wind=wind,gazeData=gazeData,**kwargs)
        else: GazePoint.__init__(self, gazeData,wind=wind)

        try:
            indic=['Velocity','Acceleration','Fixation','Saccade','Pursuit','Tracking','Searching']
            self.lim=([0,450],[-42000,42000],[0,1],[0,1],[0,1],[0,1],[0,1])# limit of y axis
            self.span=(0.9,0.9,0.6,0.6,0.6,0.6,0.6)# height of the window taken by graph
            self.offset=(0.1,0.1,0.2,0.2,0.2,0.2,0.2)
            fhandles=[self.gazeData.getVelocity,self.gazeData.getAcceleration,
                      self.gazeData.getFixations,self.gazeData.getSaccades,
                      self.gazeData.getPursuit, self.gazeData.getTracking,
                      self.gazeData.getSearch]
            self.ws=30 
            ga=[7.8337, 18.7095,-13.3941,13.3941] # graph area
            self.ga=ga
            mid=ga[0]+(ga[1]-ga[0])/2.0
            inc=2*ga[3]/float(len(indic));self.inc=inc
            
            frame=[visual.Line(self.wind,(ga[0],ga[3]),(ga[0],ga[2]),lineWidth=4.0),
                visual.Line(self.wind,(-ga[1],ga[3]),(ga[1],ga[3]),lineWidth=4.0),
                visual.Line(self.wind,(mid,ga[3]),(mid,ga[2]),lineWidth=2.0),
                visual.Line(self.wind,(ga[1],ga[3]),(ga[1],ga[2]),lineWidth=4.0)]
            
            self.graphs=[]
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
            self.tmsg=visual.TextStim(self.wind,color=(0.5,0.5,0.5),pos=(-15,15))
            
            self.gData=[]
            step=1000/float(Q.refreshRate)
            xNew=np.arange(self.tstart,self.tend,step)
            xOld=self.gazeData.getGaze(self.phase)[:,0]#[:-1,0]
            for g in range(len(fhandles)):
                yOld=fhandles[g](self.phase)
                #print 'graphdata ',g, xOld.shape,yOld.shape
                self.gData.append(interpRange(xOld,yOld,xNew))
            self.wind.flip()
        except:
            self.wind.close()
            raise
        
    def showFrame(self,positions):
        for g in range(len(self.graphs)):
            fs=max(0,self.f-self.ws)
            fe=min(self.f+self.ws,self.gazeData.getGaze().shape[0]-1)
            yveldata=self.gData[g][fs:fe]
            
            yveldata=(yveldata-self.lim[g][0])/float(self.lim[g][1]-self.lim[g][0])

            yveldata[yveldata>self.lim[g][1]]=self.lim[g][1]
            yveldata[yveldata<self.lim[g][0]]=self.lim[g][0]
            yveldata=np.matrix((self.ga[3]-(g+1)*self.inc)
                        +(self.span[g]*yveldata+self.offset[g])*self.inc)
            if g==0: # todo: move this bit into constructor?        
                unit = (self.ga[1]-self.ga[0])/float(self.ws)/2.0
                step=(2*self.ws-yveldata.size)*unit
                if fs<self.ws: s=self.ga[0]+step
                else: s=self.ga[0]
                if fe>self.gData[g].size-self.ws: e=self.ga[1]-step
                else: e=self.ga[1]
                xveldata=np.matrix(np.linspace(s,e,yveldata.size))
            veldata=np.concatenate((xveldata,yveldata),axis=0).T.tolist()
            self.graphs[g].setVertices(veldata)
        rct=self.gazeData.recTime
        self.tmsg.setText('Time %d:%02d:%06.3f' % (rct.hour,
                rct.minute+ (rct.second+int(self.t[self.f]/1000.0))/60,
                np.mod(rct.second+ self.t[self.f]/1000.0,60)) )
        self.frame.draw()
        self.tmsg.draw()
        positions[:,0]-=6.0 # shift
        Trajectory.showFrame(self,positions)

if __name__ == '__main__':
    #t=np.load('test.npy')
    from Settings import *
    from Constants import *
    from Maze import *
    from evalETdata import *
    vp=18
    block=3
    trial=7
    data=readEdf(vp,block)
    D = ETBlockData(data)
    D.driftCorrection()
    D.loadBehavioralData()
    trl=D.getTrial(trial)
    trl.extractAgentDistances()
    trl.extractTracking()
    R=ETReplay(trl.getTraj(),gazeData=trl,
               trajRefresh=100.0,phase=1,eyes=2)
    R.play(tlag=0.02)
    #print R.tend
##    data=readTobii(vp,block)
##    #maze=TestMaze(10,dispSize=24)
##    order=np.load('input%svp%03d%sordervp%03db%d.npy'%(Q.delim,vp,Q.delim,vp,block))
##    scale=0.8
##    traj=scale*np.load('input%svp%03d%svp%03db%dtrial%03d.npy'%(Q.delim,vp,Q.delim,vp,block,order[trial]))
##    data[trial].gaze*=scale
##    tr=ETData(traj,gazeData=data[trial])
##    tr.replay(tlag=0)
    

    #r=traj2image(t[113,:,:].squeeze())
