from psychopy import visual,monitors
from psychopy.misc import pix2deg
from Constants import *
from os import getcwd


class Settings():
    def __init__(self,monname,os,trialDur,refreshRate,agentSize,phiRange,
        pDirChange,initDistCC,bckgCLR,agentCLR,mouseoverCLR,selectedCLR,aSpeed,
        guiPos,winPos):
        self.refreshRate=float(refreshRate)
        self.monname=monname
        self.monitor=monitors.Monitor(self.monname)
        self.os=os
        self.trialDur=trialDur
        self.agentSize=agentSize
        self.phiRange=phiRange
        self.pDirChange=[pDirChange[CHASEE]/self.refreshRate,
                         pDirChange[CHASER]/self.refreshRate]
        self.initDistCC=initDistCC
        self.bckgCLR=bckgCLR
        self.agentCLR=agentCLR
        self.mouseoverCLR=mouseoverCLR
        self.selectedCLR=selectedCLR
        self.aSpeed=aSpeed/self.refreshRate
        self.guiPos=guiPos
        self.winPos=winPos
        if self.os==WINDOWS: self.delim='\\'
        else: self.delim='/'
        path = getcwd()
        path = path.rstrip('code')
        self.inputPath=path+"input"+self.delim
        self.outputPath=path+"output"+self.delim
        self.agentRadius=self.agentSize/2.0
        self.nrframes=self.trialDur*self.refreshRate+1
  
    def initDisplay(self,sz=1000,fullscr=False):
        if type(sz)==int: sz=(sz,sz)
        wind=visual.Window(monitor=self.monname,fullscr=fullscr,
            size=sz,units='deg',color=self.bckgCLR,pos=self.winPos,
            winType='pyglet')
        return wind
    def norm2pix(self,xy):
        return (np.array(xy)) * np.array(self.monitor.getSizePix())/2.0
    def norm2deg(self,xy):
        xy=self.norm2pix(xy)
        return pix2deg(xy,self.monitor)
        

laptop={'monname' :     'dell',
        'refreshRate':  75,                 # [hz]
        'os':           LINUX,              # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [0.08*60,0.09*60],  # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     30,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,0)}              # in pixels
eyelinklab ={'monname' :     'sony',
        'refreshRate':  100,                # [hz]
        'os':           WINDOWS,            # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [0.08*60,0.09*60],  # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     30,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,0)}              # in pixels
tobiilab ={'monname' :     'hyundai',
        'refreshRate':  75,                # [hz]
        'os':           WINDOWS,            # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [0.08*60,0.09*60],  # avg number of direction changes per second
        'bckgCLR':  [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     120,                 # in seconds
        'aSpeed':       9,                  # in degrees of visual angle per second
        'guiPos':       (-800,400),         # in pixels
        'winPos':       (1280,0)}           # in pixels

Q=Settings(**laptop)
