import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad

import graph
import fire
import adaptiveHexagonMesh

#################################### Finsler Fire ####################################
############## The following fire is a Finsler fire of type Randers
############## In Master:
############## See Section "Randers Metric" for Indicatrix field
############## See Section "Finsler: Randers" (Randers 2) for fire simulation
####################################
################### Constants
# Constants to rotate graph
angles = 4
fullAngle = np.pi/3
thetaAll = np.linspace(0, fullAngle, angles, endpoint=False)
l = 1

# Fire constants
ignitionPoint = np.array([0,0])
extinguishFactor = 1
errVal = 0.01

# Area
xrange = np.array([-8,8])
yrange = np.array([-8,11])

# N: Describe size of basemesh in adaptive hexagon mesh
N = 6

# Time
Tmax = 20
t0 = 7

#################### Functions
def aFunc(u1,u2):
    return 1*np.ones(np.shape(u1))

def bFunc(u1,u2):
    return 1.1*np.ones(np.shape(u1))

def thetaFunc(u1,u2):
    return np.arctan2(np.sin(u1), np.cos(u1))

def WFunc(u1,u2):
    thetaVal = thetaFunc(u1,u2)
    W = np.array([np.sin(thetaVal), np.cos(thetaVal)])
    
    return 0.3*W

def HFunc(u1,u2):
    aVals = aFunc(u1,u2)
    bVals = bFunc(u1,u2)
    thetaVals = thetaFunc(u1,u2)
    
    h11 = (aVals**2)*(np.sin(thetaVals)**2) + (bVals**2)*(np.cos(thetaVals)**2)
    h12 = (aVals**2 - bVals**2)*np.sin(thetaVals)*np.cos(thetaVals)
    h21 = h12
    h22 = (aVals**2)*(np.cos(thetaVals)**2) + (bVals**2)*(np.sin(thetaVals)**2)
    
    factor = (aVals**2)*(bVals**2)
    factor = np.maximum(factor, 1e-12)
    
    return (1/factor)*np.array([[h11, h12], [h21, h22]])

def hFunc(u1,u2,V1,V2):
    # H(u1, u2) is a (2, 2) matrix, and V1, V2 are (N, M) meshgrids
    
    Hvals = HFunc(u1,u2)
    
    # V1^T . HVals . V2
    if (np.size(V1) > 2) and (np.size(V2) > 2):
        alpha1 = V1[0,:,:]
        alpha2 = V1[1,:,:]
        beta1 = V2[0,:,:]
        beta2 = V2[1,:,:]
    elif (np.size(V1) > 2) and (np.size(V2) == 2):
        alpha1 = V1[0,:,:]
        alpha2 = V1[1,:,:]
        beta1 = V2[0]
        beta2 = V2[1]
    elif (np.size(V1) == 2) and (np.size(V2) == 2):
        alpha1 = V1[0]
        alpha2 = V1[1]
        beta1 = V2[0]
        beta2 = V2[1]
    
    term1 = beta1*(alpha1*Hvals[0,0] + alpha2*Hvals[1,0])
    term2 = beta2*(alpha1*Hvals[0,1] + alpha2*Hvals[1,1])

    return term1 + term2
    
def lambFunc(u1,u2):
    WArr = WFunc(u1,u2)
    return 1 - hFunc(u1,u2,WArr,WArr)

def F(u1,u2,y1,y2):
    # p = [u1,u2] in M ([u1,u2]: Array)
    # y = [y1,y2] in TpM ([y1,y2]: Array)
    
    VArr = np.array([y1,y2])
    WArr = WFunc(u1,u2)
    
    lambVal = lambFunc(u1,u2)
    lambVal = np.maximum(lambVal, 1e-12)
    
    term1 = np.sqrt(lambVal*hFunc(u1,u2,VArr,VArr) + hFunc(u1,u2,VArr,WArr)**2)/lambVal
    term2 = hFunc(u1,u2,VArr,WArr)/lambVal
    
    return term1 - term2

def findWeights(p1, p2, xrange, yrange):
    if ((p1[0] < xrange[0] or p1[0] > xrange[1] or p1[1] < yrange[0] or p1[1] > yrange[1]) and
        (p2[0] < xrange[0] or p2[0] > xrange[1] or p2[1] < yrange[0] or p2[1] > yrange[1])):
        return 0  # Return 0 if both points are outside region
    else:
        return w_func(p1[0], p1[1], p2[0], p2[1])

def generateWeightFinsler(F_func):
    # F_func: Functions handle (input, (y1,y2,u1,u2))
    #    (y1,y2) : Vectors in tangent space
    #    (u1,u2) : Point on tangentspace and surface
    
    # Straight curve between p and q
    def gam_func(tau, p1, p2, q1, q2):
        p = np.array([p1, p2])
        q = np.array([q1, q2])
        return p + tau * (q - p)

    def dgam_func(p1, p2, q1, q2):
        return np.array([q1 - p1, q2 - p2])
    
    # Build integrand
    def build_integrand(p1, p2, q1, q2):
        def integrand(tau):
            p = gam_func(tau, p1, p2, q1, q2)
            v = dgam_func(p1, p2, q1, q2)
            return F_func(v[0], v[1], p[0], p[1])
        return integrand

    # Integrate over t in [0,1]
    def integral_I_func(p1, p2, q1, q2):
        integrand = build_integrand(p1, p2, q1, q2)
        val, _ = quad(integrand, 0, 1)
        return val

    return integral_I_func

# p = (u1,u2) in M
# v = (y1,y2) in TpM
w_func = generateWeightFinsler(lambda y1,y2,u1,u2: F(u1,u2,y1,y2))

########################## Illustrate One Base Mesh
theta0 = 0

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
M = adaptiveHexagonMesh.adHexMesh(N = N, dist=lambda p1, p2: 1, theta=theta0)
M.generateBaseMesh(rotationPoint=ignitionPoint)
M.plotMesh(ax)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Illustration of One Base Mesh")

########################## Dijkstra's Algorithm
startDijkstra = time.time()
FAll = []

for theta0 in thetaAll:
    startDijkstraSubtime = time.time()
    
    ############# Adaptive Hexagon
    M = adaptiveHexagonMesh.adHexMesh(N = N, dist=lambda p1, p2: findWeights(p1, p2, xrange, yrange), theta=theta0)
    M.generateBaseMesh(rotationPoint=ignitionPoint)
    M.generateAdaptiveHexMesh()
    M.generateGraph(ignitionPoint=ignitionPoint)
    
    pointsRot = M.points.copy()
    gRot = M.g.copy()

    fireIdx = graph.findNode(pointsRot, ignitionPoint)

    ############# Fire
    FRot = fire.Fire(gRot, fireIdx, pointsRot)
    FRot.generateFire(Tmax, extinguishFactor=extinguishFactor)
    FAll.append(FRot)

    endDijkstraSubtime = time.time()
    print(f"Sub time (Dijkstra): {(endDijkstraSubtime - startDijkstraSubtime)/60:.3f} minutes")

endDijkstra = time.time()

print(f"Time (Dijkstra): {(endDijkstra - startDijkstra)/60:.3f} minutes")

########################## Illustration
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for F0 in FAll:
    dist = F0.distances
    points = F0.points

    # Area burned to time t0
    burnedIdx = dist <= t0+errVal
    burned = points[burnedIdx,:]
    
    ax.plot(burned[:,0], burned[:,1], 'o', color='gray', alpha=0.5)
    
ax.plot(ignitionPoint[0], ignitionPoint[1], 'ro')

ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
ax.tick_params(labelsize=18)
ax.grid(True)
