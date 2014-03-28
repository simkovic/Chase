import numpy as np
import pylab as plt
from Constants import *
from Settings import Q
##plt.ion()
##vpn=range(143,161)
##vpn=range(170,185)
##stats=np.zeros((len(vpn),6))
##sacs=np.zeros((len(vpn),12))
##for vp in vpn:
##    print vp
##    data=readTobii(vp,0)
##    for d in data:
##        stats[vp-vpn[0],4]+= min(1,len(d.reward))
##        stats[vp-vpn[0],5]+=1
##        for m in d.msgs:
##            for k in range(12):
##                sacs[vp-vpn[0],k]+= '%dth'%(k+1) in m[2]
##        for fix in d.fev:
##            stats[vp-vpn[0],3]+=1
##            ags=fix[2:]
##            if 0 in ags or 1 in ags:
##                if len(d.reward) and d.gaze[fix[0],0]>d.reward[0]:
##                    stats[vp-vpn[0],1]+=1
##                else: stats[vp-vpn[0],0]+=1
##            elif len(ags):
##                stats[vp-vpn[0],2]+=1
##stats2=np.copy(stats)
##sacs2=np.copy(sacs)
##np.save('stats2',stats2)

#gesamtzeit

plt.cla()  
stats1=np.load('stats1.npy')
stats2=np.load('stats2.npy')
tot=stats1[:,0]+stats1[:,2]
plt.plot(range(143,161),stats1[:,0]/tot,'or')
tot=stats2[:,0]+stats2[:,2]
plt.plot(range(170,185),stats2[:,0]/tot,'ok')

# reinforcement learning
##vp=150
##i=0
##search=True
##maxags=8
##
##phase=np.load(Q.inputPath+'vp%d/phasevp%db0.npy'%(vp,vp))
##pi=0
##Qtable=np.zeros((2,2,2))
##
##for t in range(2):
##    trackedChase=False
##    sawPurple=False
##    sac2chase=0;sac2other=0
##    lasta=-1
##    print t,phase[pi]
##    for f in range(120):
##        ags=range(maxags)
##        try: ags.remove(lasta)
##        except ValueError: pass
##        if search: agent=ags[np.random.randint(len(ags))]
##        if agent in [CHASER,CHASEE]: sac2chase+=1;sac2other=0
##        elif agent==lasta : sac2other+=1;sac2chase=0
##        else: sac2chase=0; sac2other=0
##        if sac2chase>=12 or sac2other>=12: break
##        # flags for experiment control
##        tc= phase[pi]==0 and sac2chase>3 or phase[pi]>0 and sac2chase>5
##        if tc:  trackedChase=True
##        if phase[pi]!=2 and tc: sawPurple=True
##        elif f>0: Qtable[lasttc,lastsp,search]+= -1
##        # flags for Qtable
##        lasta=agent
##        lasttc= agent in [CHASER,CHASEE]
##        lastsp= tc and phase[pi]!=2
##        # choose action
##        search= np.argmax(Qtable[lasttc,lastsp,:])
##        
##        print '\t',f,agent, lasttc, lastsp,search, Qtable[lasttc,lastsp,:]
##    #print '\t',sawPurple, trackedChase
##    if trackedChase: pi+=1
        
            
    
