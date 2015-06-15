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

from psychopy import visual,monitors
from psychopy.misc import pix2deg,deg2pix
from Constants import *
from os import getcwd
import pickle

class Settings():
    def __init__(self,monitor,os,trialDur,refreshRate,agentSize,phiRange,
        pDirChange,initDistCC,bckgCLR,agentCLR,mouseoverCLR,selectedCLR,aSpeed,
        guiPos,winPos,fullscr):
        self.refreshRate=float(refreshRate)
        self.monitor=monitor
        self.os=os
        self.fullscr=fullscr
        self.setTrialDur(trialDur)
        self.agentSize=agentSize
        self.phiRange=phiRange
        self.setpDirChange(pDirChange)
        self.initDistCC=initDistCC
        self.bckgCLR=bckgCLR
        self.agentCLR=agentCLR
        self.mouseoverCLR=mouseoverCLR
        self.selectedCLR=selectedCLR
        self.setAspeed(aSpeed)
        self.guiPos=guiPos
        self.winPos=winPos
        if self.os==WINDOWS: self.delim='\\'
        else: self.delim='/'
        path = getcwd()
        path = path.rstrip('code')
        self.inputPath=path+"input"+self.delim
        self.outputPath=path+"output"+self.delim
        self.stimPath=path+"stimuli"+self.delim
        self.agentRadius=self.agentSize/2.0
        self.fullscr=fullscr
    def setTrialDur(self,td):
        self.trialDur=td
        self.nrframes=self.trialDur*self.refreshRate+1
    def setpDirChange(self,pDirChange):
        self.pDirChange=[pDirChange[CHASEE]/self.refreshRate,
             pDirChange[CHASER]/self.refreshRate,
             pDirChange[DISTRACTOR]/self.refreshRate]
    def setAspeed(self,aSpeed):  self.aSpeed=aSpeed/self.refreshRate
  
    def initDisplay(self,sz=None):
        if sz==None: sz=(1024,1024)
        elif type(sz)==int: sz=(sz,sz)
        wind=visual.Window(monitor=self.monitor,fullscr=self.fullscr,
            size=sz,units='deg',color=self.bckgCLR,pos=self.winPos,
            winType='pyglet',screen=0)
        return wind
    def norm2pix(self,xy):
        return (np.array(xy)) * np.array(self.monitor.getSizePix())/2.0
    def norm2deg(self,xy):
        xy=self.norm2pix(xy)
        return pix2deg(xy,self.monitor)
    def pix2deg(self,pix):
        return pix2deg(pix,self.monitor)
    def deg2pix(self,deg):
        return deg2pix(deg,self.monitor)
    def save(self,filepath):
        f=open(filepath,'wb')
        try: pickle.dump(self,f);f.close()
        except: f.close(); raise
    @staticmethod
    def load(filepath):
        f=open(filepath,'rb')
        try: out=pickle.load(f);f.close()
        except: f.close(); raise
        return out
# monitors
dell=monitors.Monitor('dell', width=37.8, distance=60); dell.setSizePix((1280,1024))
sonycrt=monitors.Monitor('sony', width=40, distance=60); sonycrt.setSizePix((1280,1024))
smidell=monitors.Monitor('smiDell', width=47.5, distance=60);smidell.setSizePix((1680,1024))
t60=monitors.Monitor('tobii', width=34, distance=50); t60.setSizePix((1280,1024))
eizo=monitors.Monitor('eizo', width=34, distance=40); eizo.setSizePix((1280,1024))

laptop={'monitor' :     dell,
        'refreshRate':  60,                 # [hz]
        'os':           LINUX,              # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [4.8,5.4,4.8],          # avg number of direction changes per second
        'bckgCLR':      [-1,-1,-1],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     30,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,1),              # in pixels
        'fullscr':      False}

eyelinklab ={'monitor' :sonycrt,
        'refreshRate':  85,                # [hz]
        'os':           WINDOWS,            # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [4.8,5.4,4.8],          # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     30,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,0),              # in pixels
        'fullscr':      True}

smilab ={'monitor' :     smidell,
        'refreshRate':  60,                # [hz]
        'os':           WINDOWS,            # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [4.8,5.4,4.8],  # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     30,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,0),              # in pixels
        'fullscr':      True}
tobiilab ={'monitor' :  t60,
        'refreshRate':  75,                # [hz]
        'os':           LINUX, #hotfix           # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [4.8,5.4,4],          # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     120,                # in seconds
        'aSpeed':       9,                  # in degrees of visual angle per second
        'guiPos':       (-800,400),         # in pixels
        'winPos':       (1280,0),           # in pixels
        'fullscr':      True}

gao10e3={'monitor' :     t60,
        'refreshRate':  75,                 # [hz]
        'os':           WINDOWS,              # Linux or Windows
        'phiRange':     [90,90],          # in degrees [0-360]
        'agentSize':    1.9,                  # in degrees of visial angle
        'initDistCC':   [4.0 ,4.0],       # in degrees of visial angle
        'pDirChange':   [3.0,3.0,3.0],          # avg number of direction changes per second
        'bckgCLR':      [-1,-1,-1],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     17,                 # in seconds
        'aSpeed':       7.8,               # in degrees of visual angle per second
        'guiPos':       (-800,400),          # in pixels
        'winPos':       (1280,0),              # in pixels
        'fullscr':      True}
        
gao10e4={'monitor' :     t60,
        'refreshRate':  75,                 # [hz]
        'os':           WINDOWS,              # Linux or Windows
        'phiRange':     [90,60.0],          # in degrees [0-360]
        'agentSize':    1.5,                  # in degrees of visial angle
        'initDistCC':   [4.0 ,4.0],       # in degrees of visial angle
        'pDirChange':   [3.0,3.0,3.0],          # avg number of direction changes per second
        'bckgCLR':      [-1,-1,-1],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     8,                 # in seconds
        'aSpeed':       5.5,               # in degrees of visual angle per second
        'guiPos':       (-800,400),          # in pixels
        'winPos':       (1280,0),              # in pixels
        'fullscr':      True}

matusdesktop ={'monitor' :     eizo,
        'refreshRate':  60,                 # [hz]
        'os':           LINUX,              # Linux or Windows
        'phiRange':     [120,0*2],          # in degrees [0-360]
        'agentSize':    1,                  # in degrees of visial angle
        'initDistCC':   [12.0 ,18.0],       # in degrees of visial angle
        'pDirChange':   [4.8,5.4,4.8],          # avg number of direction changes per second
        'bckgCLR':      [-0,-0,-0],
        'agentCLR':     1,                  # [1 -1]
        'mouseoverCLR': 0.5,                # [1 -1]
        'selectedCLR':  -0.5,               # [1 -1]
        'trialDur':     120,                 # in seconds
        'aSpeed':       14.5,               # in degrees of visual angle per second
        'guiPos':       (200,400),          # in pixels
        'winPos':       (0,0),              # in pixels
        'fullscr':      False}

Q=Settings(**tobiilab)

Q=Settings(**matusdesktop)
Qexp=Settings(**tobiilab)


