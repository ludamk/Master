import numpy as np
import matplotlib.pyplot as plt

import graph
import fire
import diffGeo
import adaptiveHexagonMesh

#################################### Simple Fire ####################################
################### Constants
# Surface: r(u1,u2) = c1*u1 + c2*u2
c1 = 2
c2 = 3

# Constants to rotate graph
angles = 4
fullAngle = np.pi/3
thetaAll = np.linspace(0, fullAngle, angles, endpoint=False)

# Fire constants
ignitionPoint = np.array([0,0])
extinguishFactor = 1
errVal = 0.01

# Area
xrange = np.array([-5,5])
yrange = np.array([-5,5])

# N: Describe size of basemesh in adaptive hexagon mesh
N = 2

# Time
Tmax = 20
t0 = 5

###################
def findWeights(p1,p2,xrange,yrange):
    if ((p1[0] < xrange[0] or p1[0] > xrange[1] or p1[1] < yrange[0] or p1[1] > yrange[1]) and
        (p2[0] < xrange[0] or p2[0] > xrange[1] or p2[1] < yrange[0] or p2[1] > yrange[1])):
        return 0  # Return 0 if both points are outside region
    else:
        deltaX = p2[0]-p1[0]
        deltaY = p2[1]-p1[1]
        
        C = (deltaX**2)*(1 + c1**2) + (deltaY**2)*(1 + c2**2) + 2*deltaX*deltaY*c1*c2
        
        return np.sqrt(C)

def G(u1,u2):
    g11 = c1**2 + 1
    g12 = c1*c2
    g21 = g12
    g22 = c2**2 + 1
    return np.array([[g11,g12],[g21,g22]])

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
FAll = []
for theta0 in thetaAll:
    ############# Adaptive Hexagon
    ## Generate class M
    M = adaptiveHexagonMesh.adHexMesh(N = N, dist=lambda p1, p2: findWeights(p1, p2, xrange, yrange), theta=theta0)
    
    ## Generate a basemesh which is not refined
    M.generateBaseMesh(rotationPoint=ignitionPoint)
    
    ## Refine the basemesh (only in region xrange and yrange)
    M.generateAdaptiveHexMesh()
    
    ## Convert adaptive hexagon mesh to a graph
    M.generateGraph(ignitionPoint=ignitionPoint)
    
    ## Extract graph and points
    pointsRot = M.points
    gRot = M.g
    fireIdx = graph.findNode(pointsRot, ignitionPoint)
    
    ############# Fire
    ## Generate class Fire
    FRot = fire.Fire(gRot, fireIdx, pointsRot)
    
    ## Simulate fire
    FRot.generateFire(Tmax, extinguishFactor=extinguishFactor+errVal)
    FAll.append(FRot)

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
diffGeo.plotIndicatrix(G, [ignitionPoint[0]], [ignitionPoint[1]], xrange, yrange, ax=ax, scale=t0)

ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
ax.tick_params(labelsize=18)
ax.grid(True)
