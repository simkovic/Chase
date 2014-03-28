from psychopy import visual, core, event,gui,sound
from psychopy.misc import pix2deg, deg2pix
import os,time, sys
import numpy as np
import random
from psychopy.core import Clock
from Settings import Q
ppClock=Clock()
try: from Tobii import TobiiController
except: pass

class Agent():
    PEEK,FREEZE,HIDE,FINISHED = range(4)
    def __init__(self,win,shift=0):
        self.constLag=1 # constant lag before motion starts
        self.varLag=1 # maximum variable lag before motion starts
        self.fDur=2 # duration of freeze
        self.pDist=7  # max distance from center during peek in degrees
        self.r=1 # circle radius
        self.pvel=0.05# deg per frame
        self.hvel=1
        self.shift=shift
        self.c=visual.Circle(win,radius=self.r,fillColor='blue',lineColor='blue',
            interpolate=False,depth=2,pos=(self.shift,0))
        #self.c.setAutoDraw(True)
        
    def initMotion(self,deg,r):
        
        self.dpos=np.array([np.cos(deg/180.0*np.pi),np.sin(deg/180.0*np.pi)])
        self.initpos=self.dpos*(r-self.r-0.1)
        #print self.initpos
        self.initpos[0]+=self.shift
        #print self.initpos
        self.c.setPos(self.initpos)
        self.dpos *=self.pvel
        self.status=Agent.PEEK
        tonset=self.constLag+self.varLag*np.random.rand()
        return tonset
        
    def peek(self):
        self.c.setPos(self.c.pos+self.dpos)
        self.c.draw()
        pos=np.copy(self.c.pos)
        pos[0] -= self.shift
        if np.linalg.norm(pos)>self.pDist:
            self.status=Agent.FREEZE
            self.t0=ppClock.getTime()
        
    def freeze(self):
        self.c.draw()
        if ppClock.getTime()-self.t0>self.fDur: self.status=Agent.HIDE
        
    def hide(self):
        #pos=np.copy(self.c.pos)
        #pos[0] -= self.shift
        if np.linalg.norm(self.c.pos+[-self.shift,0])<=np.linalg.norm(self.initpos+[-self.shift,0]):
            self.status=Agent.FINISHED
        self.c.setPos(self.c.pos-2*self.dpos)
        self.c.draw()
        #print 'end'
        #core.quit()
            
    def move(self):
        if self.status==Agent.PEEK: self.peek()
        elif self.status==Agent.FREEZE: self.freeze()
        elif self.status==Agent.HIDE: self.hide()
        return self.status!=Agent.FINISHED

class InteractiveAgent(Agent):
    def __init__(self,win,gazeData):
        Agent.__init__(self,win)
        self.gaze=gazeData
        self.fDur=5
        
    def initMotion(self,deg,r):
        f=False; t0 =ppClock.getTime()
        while not f and ppClock.getTime()-t0<5: 
            gc,fc,f=self.gaze(units='deg')
            time.sleep(0.01)
        #print fc
        theta=(np.arctan2(fc[1],fc[0]-self.shift)/np.pi*180+360+180)%360
        #print 'initMotion: theta= ',theta, ' fc= ',fc
        return Agent.initMotion(self,theta,r)
            
        

    
class Experiment():
    def __init__(self,outputPath='',win=None,showGui=True,shift=-0):
        self.outputPath=os.getcwd()+outputPath
        if showGui: # ask infos
            myDlg = gui.Dlg(title="Experiment zur Interaktion",pos=(-800,400))   
            myDlg.addText('VP Infos')   
            myDlg.addField('VP Nummer:',99)
            myDlg.addField('Alter in Tagen:', 21)
            myDlg.addField('Geschlecht (m/w):',choices=(u'weiblich',u'maennlich'))
            myDlg.addField('Bedingung:',choices=('interaktiv','zufall'))
            #myDlg.addField('Starte bei Trial:', 0)
            myDlg.show()#show dialog and wait for OK or Cancel
            vpInfo = myDlg.data
            self.cond=vpInfo[-1]=='interaktiv'
            self.id=vpInfo[0]
            #self.initTrial=vpInfo[-1]
            if myDlg.OK:#then the user pressed OK
                subinf = open(self.outputPath+'vpinfoCG.res','a')
                subinf.write('%d\t%d\t%s\t%s\n'% tuple(vpInfo))
                subinf.close()               
            else: print 'Experiment cancelled'
        #init stuff
        if win is None:
            self.win=visual.Window(monitor=Q.monname,size=(900,900),pos=(1280,0),
                  fullscr=True,units='deg',winType='pyglet')
        else: self.win=win
        self.occluder=visual.Circle(self.win,radius=5,fillColor=(0.2,0.2,-0.8),lineColor=(0.2,0.2,-0.8),
            interpolate=False,depth=1,pos=(shift,0))
        self.agent=Agent(self.win,shift=shift)
        self.tlag=0
        self.f=0
    def getWind(self): return self.win 
    def setFixationSource(self,fm):
        self.agent=InteractiveAgent(self.win, fm)
    def run(self):
        self.win.flip()
        self.occluder.setAutoDraw(True)
        self.win.flip()
        time.sleep(1)
        for i in range(10):
            self.theta=np.random.rand()*360
            tonset=self.agent.initMotion(self.theta,self.occluder.radius)
            time.sleep(tonset)
            self.runTrial()
        self.win.close()

            
    def runTrial(self):
        while self.agent.move():
            if 'escape' in event.getKeys(): self.win.close(); return
            self.showFrame()
            #self.win.flip()
        
     
    def replay(self,thetas,tlag=0):
        self.tlag=tlag
        self.win.flip()
        self.f=0
        self.occluder.setAutoDraw(True)
        self.gazeP=visual.Circle(self.win,radius=0.5,fillColor='white',lineColor='black',
            interpolate=True)
        self.gazeP.setAutoDraw(True)
        
        ppClock.reset()
        for i in range(10):
            self.theta=thetas[i,1]
            while self.f*1000/ Q.refreshRate<thetas[i,0]:
                self.showFrame()
            t0=ppClock.getTime()
            self.runTrial()
            self.tonset=0
        self.win.close()
        
    def showFrame(self):
        self.f+=1
        if self.tlag>0: time.sleep(self.tlag)
        if self.agent.__class__.__name__ is InteractiveAgent.__name__:
            gc,fc,f=self.agent.gaze(units='deg')
            if np.linalg.norm(fc-self.agent.c.pos)<2:
                self.agent.status= Agent.HIDE
        self.win.flip()
        
class BabyExperiment(Experiment):
    def __init__(self):
        Experiment.__init__(self)
        self.etController = TobiiController(self.getWind(),sid=self.id,block=9)
        if self.cond: self.setFixationSource(self.etController.getCurrentFixation)
        self.etController.doMain()
    def run(self):
        self.etController.preTrial(driftCorrection=False)
        self.etController.sendMessage('Trial\t0' )
        Experiment.run(self)
        self.etController.sendMessage('Omission')
        self.etController.postTrial()
        self.etController.closeConnection()
        
    def runTrial(self):
        self.etController.sendMessage('Theta\t%d'%self.theta)
        Experiment.runTrial(self)
        

if __name__ == '__main__':
    from evalETdata import readTobii
    #data=readTobii(0,9)
    #exp=Experiment()
    #exp.replay(data[1].theta)
    exp=BabyExperiment()
    exp.run()
