# diffGeo
#
import numpy as np
import matplotlib.pyplot as plt

def plotIndicatrix(G_func, u0, v0, xlim, ylim, ax=None, scale=1):
    if ax is None:
        fig, ax = plt.subplots()
    
    uRange = np.linspace(xlim[0], xlim[1], 500)
    vRange = np.linspace(ylim[0], ylim[1], 500)
    U, V = np.meshgrid(uRange, vRange)
    
    # v^T G v = [Uc, Uc] . G . [Uc, Uc]^T
        
    for u, v in zip(u0, v0):
        t = np.linspace(0, 2*np.pi, 1000)

        lamb, e = np.linalg.eigh(G_func(u,v))
        e1 = e[:,0]
        e2 = e[:,1]

        scaleU = scale/np.sqrt(lamb[0])
        scaleV = scale/np.sqrt(lamb[1])
                
        I_uVals = scaleU*np.cos(t)*e1[0] + scaleV*np.sin(t)*e2[0] + u
        I_vVals = scaleU*np.cos(t)*e1[1] + scaleV*np.sin(t)*e2[1] + v
            
        ax.plot(I_uVals, I_vVals, '-', color='black', linewidth=2.5)
                
        ax.plot(u,v, 'ro')
        
    ax.set_aspect('equal')
    