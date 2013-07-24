
# -*- coding: utf-8 -*-
from Settings import Q
from Constants import *
from psychopy import visual, core, event,gui,sound,parallel
from psychopy.misc import pix2deg, deg2pix
import time, sys
import numpy as np

import random
try: from Eyelink import TrackerEyeLink
except ImportError: print 'Warning >> Eyelink import failed'
try: from SMI import TrackerSMI
except ImportError: print 'Warning >> Eyelink import failed'
from Tobii import TobiiController,TobiiControllerFromOutput
class Experiment():
    def __init__(self):
        # ask infos
        myDlg = gui.Dlg(title="Experiment zur Bewegungswahrnehmung",pos=Q.guiPos)   
        myDlg.addText('VP Infos')   
        myDlg.addField('Subject ID:',0)
        myDlg.addField('Block:',0)
        #myDlg.addField('Scale (1 or 0.6):',1)
        myDlg.addField('Alter:', 21)
        myDlg.addField('Geschlecht (m/w):',choices=(u'weiblich',u'maennlich'))
        myDlg.addField(u'Händigkeit:',choices=('rechts','links'))
        myDlg.addField(u'Dominantes Auge:',choices=('rechts','links'))
        myDlg.addField(u'Sehschärfe: ',choices=('korrigiert','normal'))
        myDlg.addField(u'Wochenstunden vor dem Komputerbildschirm:', choices=('0','0-2','2-5','5-10','10-20','20-40','40+'))
        myDlg.addField(u'Wochenstunden Komputerspielen:', choices=('0','0-2','2-5','5-9','10-20','20+'))
        myDlg.addField('Starte bei Trial:', 0)
        myDlg.show()#show dialog and wait for OK or Cancel
        vpInfo = myDlg.data
        self.id=vpInfo[0]
        self.block=vpInfo[1]
        self.initTrial=vpInfo[-1]
        self.scale=1#vpInfo[2]
        if myDlg.OK:#then the user pressed OK
            subinf = open(Q.outputPath+'vpinfo.res','a')
            subinf.write('%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n'% tuple(vpInfo))
            subinf.close()               
        else: print 'Experiment cancelled'
        # save settings, which we will use
        Q.save(Q.inputPath+'vp%03d'%self.id+Q.delim+'SettingsExp.pkl')
        #init stuff
        self.wind=Q.initDisplay()
        self.mouse = event.Mouse(False,None,self.wind)
        self.mouse.setVisible(False)
        fcw=0.1; fch=0.8 #fixcross width and height
        fclist=[ visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fcw,fch),(-fcw,fch),(-fcw,-fch),(fcw,-fch))),
            visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fch,fcw),(-fch,fcw),(-fch,-fcw),(fch,-fcw))),
            visual.Circle(win=self.wind, pos=[0,0],fillColor='black',radius=0.1)]
        self.fixcross=visual.BufferImageStim(self.wind,stim=fclist)
        self.wind.flip(); self.wind.flip()
        self.score=0
        self.rt=0
        # init text
        fs=1 # font size
        self.text1=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,2])
        self.text2=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,0])
        self.text3=visual.TextStim(self.wind, text='Error',wrapWidth=30,pos=[0,-10])
        self.text1.setHeight(fs)
        self.text2.setHeight(fs)
        self.text3.setHeight(fs)
        self.f=0
        
    def getWind(self):
        try: return self.wind
        except AttributeError: 
            self.wind=Q.initDisplay()
            return self.wind

    def getJudgment(self,giveFeedback=False):
        position=np.transpose(self.pos)
        cond=position.shape[1]
        self.mouse.clickReset();self.mouse.setVisible(1)
        elem=self.elem
        t0=core.getTime();selected=[]
        mkey=self.mouse.getPressed()
        lastPress=t0
        while sum(mkey)>0:
            elem.draw()
            self.wind.flip()
            mkey=self.mouse.getPressed()
        released=True
        clrs=np.ones((cond,1))*Q.agentCLR
        while len(selected) <2:
            elem.draw()
            self.wind.flip()
            mpos=self.mouse.getPos()
            mkey=self.mouse.getPressed()
            mtime=core.getTime()
            for a in range(cond):
                if (event.xydist(mpos,np.squeeze(position[:,a]))
                    < Q.agentRadius*self.scale):
                    if 0<sum(mkey) and released: # button pressed
                        if selected.count(a)==0: # avoid selecting twice
                            clrs[a]=Q.selectedCLR
                            elem.setColors(clrs,'rgb')
                            selected.append(a)
                            self.output.write('\t%d\t%2.4f' % (a,mtime-t0))
                        released=False
                    elif a in selected: # no button pressed but selected already
                        clrs[a]=Q.selectedCLR
                        elem.setColors(clrs,'rgb')
                    else: # no button pressed but mouse cursor over agent
                        clrs[a]=Q.mouseoverCLR
                        elem.setColors(clrs,'rgb')
                elif a in selected: # no button pressed, no cursor over agent, but already selected
                    clrs[a]=Q.selectedCLR
                    elem.setColors(clrs,'rgb')
                else: # no button press, no cursor over agent, not selected
                    clrs[a]=Q.agentCLR
                    elem.setColors(clrs,'rgb')
            if 0==sum(mkey) and not released:
                released=True       
        t0=core.getTime()
        while core.getTime()-t0<1:
            elem.draw()
            self.wind.flip()
        self.mouse.setVisible(0)
        if (selected[0]==0 and selected[1]==1
            or selected[0]==1 and selected[1]==0):
            return 1
            
        else: return 0
    def trialIsFinished(self): return False
    def omission(self): pass
    def getf(self): return self.f
    def flip(self): self.wind.flip()
    def runTrial(self,trajectories,fixCross=True):
        self.nrframes=trajectories.shape[0]
        self.cond=trajectories.shape[1]
        self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
            nElements=self.cond, sizes=Q.agentSize*self.scale,
            elementMask=RING,elementTex=None,colors=Q.agentCLR)
        # display fixation cross
        if fixCross:
            self.fixcross.draw()
            self.wind.flip()
            core.wait(1+random.random()/2)
            #core.wait(0.5+random.random()/2)
            #self.eyeTracker.sendMessage('Movement Start')
        self.mouse.clickReset()
        event.clearEvents() #agents[a].setLineColor(agentCLR)
        self.noResponse=True
        self.t0=core.getTime()
        #t0=core.getTime()
        #times=[]
        self.f=0
        while self.f<self.nrframes:
            self.pos=trajectories[self.f,:,[X,Y]].transpose()*self.scale
            self.elem.setXYs(self.pos)
            self.elem.draw()
            self.flip()

            # check for termination signal
            for key in event.getKeys():
                if key in ['escape']:
                    self.wind.close()
                    core.quit()
                    sys.exit()
            if self.trialIsFinished(): break
            self.f+=1
            #times.append(core.getTime()-t0)
            #t0=core.getTime()
        #np.save('times',np.array(times))
        if self.noResponse:
            self.omission() 
        self.wind.flip()
        core.wait(1.0)
        self.output.write('\n')
        
    def run(self,mouse=None,prefix=''):        
        self.output = open(Q.outputPath+prefix+'vp%03d.res'%self.id,'a')
        
        permut=np.load(Q.inputPath+'vp%03d'%self.id+Q.delim
            +prefix+'ordervp%03db%d.npy'%(self.id,self.block))
        if len(permut.shape)>1 and permut.shape[1]>1:
            self.data=permut[:,1:]
            permut=permut[:,0]
        self.nrtrials=permut.size
        # show initial screen until key pressed
#        self.text2.setText(u'Bereit ?')
#        self.text2.draw()
#        self.wind.flip()
#        self.mouse.clickReset()
#        mkey = self.mouse.getPressed(False)
#        while not (mkey[0]>0 or mkey[1]>0 or mkey[2]>0):
#            mkey = self.mouse.getPressed(False)
        # loop trials
        for trial in range(self.initTrial,self.nrtrials):   
            self.t=trial; self.pt=permut[trial]
            #print self.t
            self.output.write('%d\t%d\t%d\t%s'% (self.id,self.block,trial,int(permut[trial])))
            fname=prefix+'vp%03db%dtrial%03d.npy' % (self.id,self.block,permut[trial])
            # print 'showing',fname
            self.trajectories= np.load(Q.inputPath+'vp%03d'%self.id+Q.delim+fname)
            self.runTrial(self.trajectories)
            
        
class Gao09Experiment(Experiment):
    def trialIsFinished(self):
        return False
    def omission(self):
        self.text3.setText('Wurde Verfolgung gezeigt?')
        t0=core.getTime();selected=[]
        self.mouse.clickReset()
        mkey=self.mouse.getPressed()
        while sum(mkey)==0:
            self.elem.draw()
            self.text3.draw()
            self.wind.flip()
            mkey=self.mouse.getPressed()
        if mkey[0]>0:
            self.output.write('\t%d\t%2.4f\t1'%(self.data[self.pt,0],core.getTime()-t0))
            if self.data[self.pt,0]>=6: self.sFalse.play()
            resp=self.getJudgment()
            res= self.data[self.pt,0]<6 and resp==1
        else: 
            self.output.write('\t%d\t%2.4f\t0\t-1\t-1\t-1\t-1'%(self.data[self.pt,0],core.getTime()-t0))
            res= self.data[self.pt,0]>=6
            if self.data[self.pt,0]<6: self.sFalse.play()
        if not res: self.output.write('\t0')
        else: self.output.write('\t1')
        
    def run(self):
        sFalse=sound.SoundPygame()
        Experiment.run(self,prefix='gao09e1')
        
class BabyExperiment(Experiment):
    criterion=6*Q.refreshRate # abort trial after infant is continuously not looking for more than criterion nr of frames 
    fixRadius=3 # threshold in deg
    sacToReward=[3,5] # reward is presented after enough saccades are made
    maxSacPerPursuit=12 # trial will be terminated if the pursuit is upheld
    blinkTolerance=0.2*Q.refreshRate # iterations
    rewardIterations=2.5*Q.refreshRate # 3 nr of frames the reward is shown since last saccade to ChCh
    maxNrRewards=1 # maximum rewards shown per trial
    initBlockDur = 1*Q.refreshRate# nr of frames, duration when reward is blocked at the trial start
    dataLag = 14 # nr of frames between the presentation and the availibility of fixation data
    maxFixInterval=2*Q.refreshRate # nr of frames, maximum allowed duration between two consecutive fixations during pursuit
    finished=2 # abort after consecutive nr of attention catchers
    expDur=6 # total experiment duration in minutes
    doReward=True # show colored reward?
    
    def __init__(self):
        Experiment.__init__(self)
        #self.etController=TobiiControllerFromOutput(self.getWind(),sid=self.id,playMode=True,block=self.block,initTrial=self.initTrial)
        self.etController = TobiiController(self.getWind(),getfhandle=self.getf,sid=self.id,block=self.block)
        self.etController.doMain()
        #self.clrOscil=0.05
        self.rewardColor1=np.array((0,-1,1))
        self.rewardColor2=np.array((1,1,1))
        self.clrs=[]; self.noscils=40; # gives 75/40=1.875 hz
        a=np.linspace(1,-1,self.noscils/2,endpoint=False)
        b=np.linspace(-1,1,self.noscils/2,endpoint=False)
        pattern=np.concatenate([a,b])
        #pattern=np.cos(np.linspace(0,2*np.pi,self.noscils,endpoint=False))
        for i in range(self.rewardColor1.size):
            self.clrs.append((pattern+1)/2.0
                *(self.rewardColor1[i]-self.rewardColor2[i])
                +self.rewardColor2[i])
        self.clrs=np.array(self.clrs).T.tolist()
        #print 'clrs',self.clrs
        self.showAttentionCatcher=False
        self.nrframes=-1
        self.phases=np.load(Q.inputPath+'vp%d'%self.id+Q.delim+'phasevp%sb%d.npy'% (self.id,self.block)) # 0 - show easy reward, 1 - show difficult reward, 2 - no reward (test)
        self.account=0 # count consecutive attention catchers
        self.pi=0
        self.eeg=parallel.PParallelInpOut32()
        
    def run(self):
        self.tStart=core.getTime()
        Experiment.run(self)
        self.etController.closeConnection()
        
    def runTrial(self,*args):
        self.babyStatus=0 # -1 no signal, 0 saccade, 1 fixation,
        self.sacPerPursuit=0
        self.pursuedAgents=False
        self.rewardIter=0
        self.nrRewards=0
        self.blinkCount=0
        self.tFix=0
        self.isFixLast=False
        self.babySawReward=False
        ende=False
        if core.getTime()> BabyExperiment.expDur*60+self.tStart: ende=True
        if ende:
            print core.getTime()-self.tStart
            self.etController.sendMessage('Finished')
            self.etController.closeConnection()
            self.wind.close(); core.quit()
        self.timeNotLooking=0
        self.etController.preTrial(driftCorrection=self.showAttentionCatcher>0)
        self.etController.sendMessage('Trial\t%d'%self.t)        
        self.etController.sendMessage('Phase\t%d'%self.phases[self.pi])
        if self.eeg!=None: 
            self.eeg.setData(int(self.t+1))
        Experiment.runTrial(self,*args,fixCross=False)
        self.etController.postTrial()
        
    def trialIsFinished(self):
        #if self.eeg!=None: self.eeg.setData(1)
        gc,fc,isFix,incf=self.etController.getCurrentFixation(units='deg')
        self.f+=incf 
        #if self.f>750: return True
        if np.isnan(gc[0]): self.babyStatus=-1; self.blinkCount+=1
        elif self.babyStatus==-1: 
            self.babyStatus=0
            self.blinkCount=0
        if self.blinkCount>=BabyExperiment.blinkTolerance: # reset if gaze is lost
            self.sacPerPursuit=0
            if self.rewardIter>0: self.turnOffReward()
                
        temp=self.trajectories[max(self.f-BabyExperiment.dataLag,0),:,:]
        #print np.array([fc.tolist()]*self.cond).shape,temp[:,:2].shape
        dxy=np.array([fc.tolist()]*self.cond)-temp[:,:2]#self.elem.xys
        distance= np.sqrt(np.power(dxy[:,0],2)+np.power(dxy[:,1],2))
        agentsInView=BabyExperiment.fixRadius>distance
        if self.f-BabyExperiment.dataLag <0: agentsInView[0]=False; agentsInView[1]=False
        #print 'show ',f, self.babyStatus
        if not isFix and self.babyStatus == 1: # saccade initiated
            self.babyStatus=0
            #self.etController.sendMessage('Saccade '+str(self.f))
        elif not self.isFixLast and isFix: # fixation started
            self.babyStatus=1
            #self.etController.sendMessage('Fixation '+str(fc[0])+' '+str(fc[1])+' '+str(self.f))
            firstFix=(self.rewardIter==0 and (not self.pursuedAgents 
                or self.f-self.tFix > BabyExperiment.maxFixInterval))
            if firstFix or self.rewardIter>0: self.pursuedAgents= (agentsInView[0] or agentsInView[1])
            else: self.pursuedAgents= (agentsInView[0] or agentsInView[1]) and self.pursuedAgents
            
            if self.pursuedAgents:
                if firstFix: self.sacPerPursuit=1
                else: self.sacPerPursuit+=1
                self.etController.sendMessage('%dth Saccade %.4f %.4f' % (self.sacPerPursuit,fc[0],fc[1]))
                if self.rewardIter>1: self.rewardIter=1
            #elif self.rewardIter==0: self.sacPerPursuit=0 
            self.tFix=self.f
        ind= (1 if self.phases[self.pi]>0 else 0)
        if (self.f < BabyExperiment.initBlockDur and 
            self.sacPerPursuit>= BabyExperiment.sacToReward[ind]):
            self.sacPerPursuit= BabyExperiment.sacToReward[ind]-1
        if (self.sacPerPursuit>= BabyExperiment.sacToReward[ind] and self.rewardIter==0): 
            self.turnOnReward()
        if self.sacPerPursuit> BabyExperiment.sacToReward[ind]: self.babySawReward=True
                
                
        #print 'f ',self.f
        if  self.rewardIter>0 and self.rewardIter<BabyExperiment.rewardIterations:
            if self.phases[self.pi]< 2: self.reward()#self.pursuedAgents.nonzero()[0])
            self.rewardIter+=1
        elif self.rewardIter>=BabyExperiment.rewardIterations: self.turnOffReward()
        
        if self.babyStatus is -1: self.timeNotLooking+=1
        else: self.timeNotLooking=0
        self.showAttentionCatcher=self.timeNotLooking>BabyExperiment.criterion
        out=(self.showAttentionCatcher or self.sacPerPursuit>=BabyExperiment.maxSacPerPursuit
            or self.nrRewards>=BabyExperiment.maxNrRewards)
        if (not self.showAttentionCatcher) and out and self.babySawReward: self.pi+=1
        if self.showAttentionCatcher and self.rewardIter>0: self.turnOffReward()
        #print gc,f,self.babyStatus,self.timeNotLooking,out
        self.isFixLast=isFix
        return out
    def turnOnReward(self):
        self.etController.sendMessage('Reward On')
        if self.eeg!=None: self.eeg.setData(70)
        self.rewardIter=1
        self.kk=0
        #clrs=np.ones((self.cond,3))
        #clrs[0,:]=self.rewardColor1
        #clrs[1,:]=self.rewardColor1
        #if BabyExperiment.reward and self.phases[self.pi]< 2: self.elem.setColors(clrs)
        
    def turnOffReward(self):
        self.etController.sendMessage('Reward Off')
        if self.eeg!=None: self.eeg.setData(80)
        self.rewardIter=0
        self.kk= -1
        self.elem.setColors(np.ones((self.cond,3)))
        self.nrRewards+=1
        
    def reward(self):
        ''' Present reward '''
        if not BabyExperiment.doReward: return
        clrs=np.ones((self.cond,3))
        if self.elem.colors.shape[0]!=self.cond: self.elem.setColors(clrs)
        nc=self.clrs[self.kk%len(self.clrs)]
        self.kk+=1
        clrs[0,:]=nc
        clrs[1,:]=nc
        self.elem.setColors(clrs)
        
    def omission(self):
        self.etController.sendMessage('Omission')
        if self.eeg!=None: self.eeg.setData(90)
    
class BehavioralExperiment(Experiment):
            
    def omission(self):
        self.output.write('\t0\t%d\t30.0\t-1\t-1\t-1\t-1\t0' %self.cond)
        self.rt+=30.0

    def trialIsFinished(self):
        mkey = self.mouse.getPressed(False)
        if mkey[0]>0 or mkey[1]>0 or mkey[2]>0:
            mtime=core.getTime()-self.t0
            self.output.write('\t1\t%d\t%2.4f' %(self.cond,mtime))
            self.rt+=mtime
            resp=self.getJudgment()
            if resp==1: self.score+=1
            self.output.write('\t%d'%resp)
            self.noResponse=False
            return True
        else: return False

    def runTrial(self,trajectories,fixCross=True):
#        # display query
#        self.mouse.clickReset()
#        self.fixcross.draw()
#        #self.text1.setText(u'Bereit ?')
#        #self.text1.draw()
#        self.wind.flip()
#        while True:
#            mkey=self.mouse.getPressed()
#            if sum(mkey)>0: break
#            core.wait(0.01)
        #self.eyeTracker.sendMessage('Drift Correction')
        Experiment.runTrial(self,trajectories,fixCross)
        # display progress info
        if self.t % 10==9:
            self.text1.setText(u'Sie hatten %d Prozent richtig.'% (self.score/10.0*100))
            self.text1.draw()
            self.score=0
            self.text2.setText('(Es geht gleich automatisch weiter.)')
            self.text2.draw()
            #text3.setText('Durchschnittliche Reaktionszeit: %d Sekunden.' % (self.rt/10.0))
            #text3.draw()
            self.rt=0
            self.wind.flip()
            core.wait(4)
        elif False:#permut.size==11+selt.t:
            self.text1.setText('Noch zehn Trials!')
            text1.draw()
            text2.setText('(Es geht gleich automatisch weiter.)')
            text2.draw()
            self.wind.flip()
            core.wait(5)
        if self.t==39: # say goodbye
            self.text1.setText(u'Der Block ist nun zu Ende.')
            self.text1.draw()
            self.text2.setText(u'Bitte, verständigen Sie den Versuchsleiter.')
            self.text2.draw()
            self.wind.flip()
            core.wait(10)
            self.output.close()
            self.wind.close()

    
class AdultExperiment(BehavioralExperiment):
    def __init__(self,doSetup=True):
        BehavioralExperiment.__init__(self)
        #self.eyeTracker = TrackerEyeLink(self.getWind(), core.Clock(),sj=self.id,block=self.block,doSetup=doSetup,target=self.fixcross)
        self.eyeTracker = TrackerSMI(self.getWind(), sj=self.id,block=self.block,target=self.fixcross)
        self.eyeTracker.sendMessage('MONITORDISTANCE %f'% self.wind.monitor.getDistance())
    def run(self):
        BehavioralExperiment.run(self)
        self.eyeTracker.closeConnection()
    def runTrial(self,*args):
        self.eyeTracker.preTrial(self.t,False,self.getWind(),autoDrift=True)
        self.eyeTracker.sendMessage('START')
        BehavioralExperiment.runTrial(self,*args,fixCross=False)
        self.eyeTracker.postTrial()
    def getJudgment(self,*args):
        self.eyeTracker.sendMessage('DETECTION')
        resp=BehavioralExperiment.getJudgment(self,*args)
        return resp
    def omission(self):
        self.eyeTracker.sendMessage('OMISSION')
        BehavioralExperiment.omission(self)
    def flip(self):
        self.eyeTracker.sendMessage('FRAME %d %f'%(self.f, core.getTime()))
        return BehavioralExperiment.flip(self)
        
class TobiiExperiment(BehavioralExperiment):
    def __init__(self,doSetup=True):
        BehavioralExperiment.__init__(self)
        self.eyeTracker = TobiiController(self.getWind(),sj=self.id,block=self.block,doSetup=doSetup)
        self.eyeTracker.sendMessage('Monitor Distance\t%f'% self.wind.monitor.getDistance())
    def run(self):
        BehavioralExperiment.run(self)
        self.eyeTracker.closeConnection()
    def runTrial(self,*args):
        self.eyeTracker.preTrial(False)#self.t,False,self.getWind(),autoDrift=True)
        self.eyeTracker.sendMessage('Trial\t%d'%self.t)
        BehavioralExperiment.runTrial(self,*args,fixCross=True)
        self.eyeTracker.postTrial()
    def getJudgment(self,*args):
        self.eyeTracker.sendMessage('Detection')
        resp=BehavioralExperiment.getJudgment(self,*args)
        return resp 
    def omission(self):
        self.eyeTracker.sendMessage('Omission')
        pass
    

if __name__ == '__main__':
    from Settings import Q
    #E=BehavioralExperiment()
    #E=TobiiExperiment()
    #E=Gao09Experiment()
    E=BabyExperiment()
    E.run()


