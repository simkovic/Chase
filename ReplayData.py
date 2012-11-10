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

class TrajectoryData():

    
    def __init__(self,trajectories,maze=None,wind=None,trajRefresh=75.0,
                 gazeData=None,highlightChase=False):
        
        self.trajectories=trajectories
        self.trajRefresh=trajRefresh
        self.gazeData=gazeData
        self.wind=wind
        self.pos=[]
        self.trialDur=min(30000.0,self.gazeData.gaze.shape[0]/float(self.gazeData.hz)*1000)
        step=1000/float(self.gazeData.hz)
        tcur=np.arange(0,self.trialDur-3*step,step)
        step=1000/float(trajRefresh)
        t=np.arange(0,(trajectories.shape[0])*step,step)
        self.t=tcur
        #print tcur.shape,t.shape,tcur[-1],t[-1], trajectories.shape 
        for a in range(trajectories.shape[1]):
            self.pos.append((interpRange(t,trajectories[:,a,0],tcur),
                interpRange(t,trajectories[:,a,1],tcur)))
        self.pos=np.array(self.pos)

        step=1000/float(gazeData.hz)
        t=np.arange(0,gazeData.gaze.shape[0]*step,step)
        self.gaze=np.array((interpRange(t,gazeData.gaze[:,1],tcur),
            interpRange(t,gazeData.gaze[:,2],tcur)))
        try:
            if type(self.wind)==type(None):
                self.wind=initDisplay()
            self.cond=trajectories.shape[1]
            if gazeData!=None:
                self.cond+=1
                self.gazeDataRefresh=gazeData.hz
            clrs=np.ones((self.cond,3))
            if self.gazeData!=None: clrs[-1,[0,1]]=0
            if highlightChase:
                clrs[0,[0,2]]=0
                clrs[1,[1,2]]=0
            self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
                nElements=self.cond,sizes=Q.agentSize,rgbs=clrs,
                elementMask='circle',elementTex=None)
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
    
    def replay(self,tlag=0.02):
        """
            shows the trial as given by TRAJECTORIES
        """
        try:
            self.wind.flip()

            #t0=core.getTime()
            step=1000/float(Q.refreshRate)
            position=np.zeros((self.cond,2))
            for f in range(self.gaze.shape[1]):
                #print f, type(f),type(self.pos)
                if position.shape[0]>1:
                    position[:-1,:]=self.pos[:,:,f]
                position[-1,:]=self.gaze[:,f].transpose()
                self.f=f
                
                self.showFrame(position)
                if tlag>0:
                    core.wait(tlag)
                for key in event.getKeys():
                    if key in ['escape']:
                        self.wind.close()
                        return
                        #core.quit()
            self.wind.flip()
            #print core.getTime() - t0
            #core.wait(1.0)
            self.wind.close()
        except: 
            self.wind.close()
            raise
class GazePoint(TrajectoryData):
    def __init__(self, gazeData,wind=None):

        self.gazeData=gazeData
        self.wind=wind
        self.pos=[]
        self.trialDur=self.gazeData.gaze.shape[0]/self.gazeData.hz*1000
        step=1000/float(Q.refreshRate)
        tcur=np.arange(0,self.trialDur-3*step,step)
        self.t=tcur
        step=1000/float(gazeData.hz)
        t=np.arange(0,gazeData.gaze.shape[0]*step,step)
        self.gaze=np.array((interpRange(t,gazeData.gaze[:,1],tcur),
            interpRange(t,gazeData.gaze[:,2],tcur)))
        try:
            if type(self.wind)==type(None):
                self.wind=initDisplay()
            self.cond=1
            self.gazeDataRefresh=gazeData.hz
            clrs=np.ones((self.cond,3))
            self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
                nElements=self.cond,sizes=Q.agentSize,rgbs=clrs,
                elementMask='circle',elementTex=None)
        except:
            self.wind.close()
            raise
    
    

class ETData(TrajectoryData,GazePoint):
    def __init__(self,trajectories,gazeData,**kwargs):
        wind = kwargs.get('wind',None)
        if wind is None:
            wind = Q.initDisplay((1280,1000))
        if trajectories!=None:
            TrajectoryData.__init__(self,trajectories,
                wind=wind,gazeData=gazeData,**kwargs)
        else: GazePoint.__init__(self, gazeData,wind=wind)

        try:
            indic=['Velocity','Acceleration','Saccade','Fixation','Pursuit']
            self.lim=([0,450],[-42000,42000],[0,1],[0,1],[0,1])# limit of y axis
            self.span=(0.9,0.9,0.6,0.6,0.6)# height of the window taken by graph
            self.offset=(0.1,0.1,0.2,0.2,0.2)
            fhandles=[self.gazeData.getVelocity,self.gazeData.getAcceleration,
                      self.gazeData.getSaccades,self.gazeData.getFixations,
                      self.gazeData.getPursuit]
            self.ws=30 # 
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
            xNew=np.arange(0,self.trialDur-3*step,step)
            step=1000/float(self.gazeData.hz)
            xOld=np.arange(0,(self.gazeData.gaze.shape[0]-1)*step,step)
            for g in range(len(fhandles)):
                yOld=fhandles[g]()
                #print 'graphdata ',g, xOld.shape,yOld.shape
                self.gData.append(interpRange(xOld,yOld,xNew))
            self.wind.flip()
        except:
            self.wind.close()
            raise
        
    def showFrame(self,positions):
        for g in range(len(self.graphs)):
            fs=max(0,self.f-self.ws)
            fe=min(self.f+self.ws,self.gazeData.gaze.shape[0]-1)
            yveldata=self.gData[g][fs:fe]
            
            yveldata=(yveldata-self.lim[g][0])/float(self.lim[g][1]-self.lim[g][0])

            yveldata[yveldata>self.lim[g][1]]=self.lim[g][1]
            yveldata[yveldata<self.lim[g][0]]=self.lim[g][0]
            yveldata=np.matrix((self.ga[3]-(g+1)*self.inc)
                        +(self.span[g]*yveldata+self.offset[g])*self.inc)
            if g==0: # todo: move this into constructor        
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
        TrajectoryData.showFrame(self,positions)

if __name__ == '__main__':
    #t=np.load('test.npy')
    from Settings import *
    from Constants import *
    from Maze import *
    from evalETdata import *
    vp=106
    block=0
    trial=0
    os.chdir('..')
    data=readTobii(vp,block)
    #maze=TestMaze(10,dispSize=24)
    order=np.load('input%svp%03d%sordervp%03db%d.npy'%(Q.delim,vp,Q.delim,vp,block))
    scale=0.8
    traj=scale*np.load('input%svp%03d%svp%03db%dtrial%03d.npy'%(Q.delim,vp,Q.delim,vp,block,order[trial]))
    data[trial].gaze*=scale
    tr=ETData(traj,gazeData=data[trial])
    tr.replay(tlag=0)
    

    #r=traj2image(t[113,:,:].squeeze())
