import os
import numpy as np
missing=[]
for vp in range(1,5):
    for b in range(1,24):
        for t in range(40):
            if not os.path.exists('vp%03db%dtr%02d.trc'%(vp,b,t)):
                missing.append([vp,b,t])

np.savetxt('missing',missing,fmt='%d')
