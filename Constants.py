
import numpy as np
# some constants
WINDOWS=1
LINUX=0
MIN=0
MAX=1
CHASEE=0
CHASER=1
DISTRACTOR=2
DISTRACTOR2=3
X=0;Y=1;PHI=2;T=2
I=0;J=1
# ring mask
N=64
RING=np.ones((N,N))*-1
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)>2*N/5 and np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)<N/2:
            RING[i,j]= 1
ANAMES=('Chasee','Chaser','Distractor')
# color stack
COLORS=['r','g','b','y','k','c']
