import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad

import graph
import fire
import adaptiveHexagonMesh

#################################### Finsler Fire (Snells Law) ####################################
############## The following fire is a Finsler fire of type Randers
############## In Master:
############## See Section "Finsler: Randers" (Randers 1) for fire simulation
####################################
################### Constants
# Ignition point
p0 = np.array([0,0])
ignitionPoint = p0
extinguishFactor = 1.1
errVal = 0.001
l = 1
numAngles = 4
Tmax = 20

# Finsler metric
beta1 = np.array([-1/2, 0])
beta2 = np.array([0, -1/2])

# interface (eta)
etaX = 3

# Critical angle and critical point
thetaC = np.pi/6
pC = np.array([etaX, (etaX-p0[0])/np.sqrt(3)])

# Reflected angle
theta3 = np.pi-thetaC

# Range
xrange = np.array([-10,15])
yrange = np.array([-10,15])

# Size of base mesh in adaptive hexagon
N_adaptive = 6

##################### Number of Points used in Geodesics
# Note: If the number of geodesics changes, also change the modulo factors in the loops
#       - these determine which geodesics is plotted and which only contribute to the fire front
N = 5000
Ntheta = 1000

N_cutLoci = 100

# Dijkstra angles
thetaAll = np.linspace(0, np.pi/3, numAngles, endpoint=False)

############# Cut-Loci
def cutLoci(C1):
    return np.sqrt(3) * C1 / 9 + 0.4e1 / 0.9e1 * np.sqrt(3) * np.sqrt((C1 - 3) ** 2) + 0.2e1 / 0.3e1 * np.sqrt(3) + 0.4e1 / 0.9e1 * np.sqrt(6 * C1 ** 2 + 6 * C1 * np.sqrt((C1 - 3) ** 2) - 63 * C1 + 36 * np.sqrt((C1 - 3) ** 2) + 135)

############# Refracted angle
def findTheta2(theta1):
    return np.arcsin(np.sin(theta1) + beta1[1]-beta2[1])

############# Finsler Metric
def F1(v1,v2):
    return np.sqrt(v1**2 + v2**2) + beta1[0]*v1 + beta1[1]*v2

def F2(v1,v2):
    return np.sqrt(v1**2 + v2**2) + beta2[0]*v1 + beta2[1]*v2

def F(u1,u2,v1,v2):
    if u1 <= etaX:
        return F1(v1,v2)
    else:
        return F2(v1,v2)

##################### Time constants
# Time for different solutions
tauEta = F1(etaX-p0[0], 0-p0[1])
tauPlus = F1(pC[0]-p0[0], pC[1]-p0[1])

#t0 = tauEta-0.5
#t0 = tauPlus
t0 = 8

################### Functions
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

w_func = generateWeightFinsler(lambda v1,v2,u1,u2: F(u1,u2,v1,v2))

def direction(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def finsler_velocity_F1(theta):
    e = direction(theta)
    return e / F1(e[0], e[1])

def finsler_velocity_F2(theta):
    e = direction(theta)
    return e / F2(e[0], e[1])

def getSpeed(theta, type=1):
    if type == 1:
        return finsler_velocity_F1(theta)
    elif type == 2:
        return finsler_velocity_F2(theta)

########################## Illustrate One Base Mesh From Dijkstra
theta0 = 0

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
M = adaptiveHexagonMesh.adHexMesh(N = N_adaptive, dist=lambda p1, p2: 1, theta=theta0)
M.generateBaseMesh(rotationPoint=ignitionPoint)
M.plotMesh(ax)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Illustration of One Base Mesh")

################################################### Dijkstra fire
startDijkstra = time.time()
FAll = []

for theta0 in thetaAll:
    startDijkstraSubtime = time.time()
    
    ############# Adaptive Hexagon
    M = adaptiveHexagonMesh.adHexMesh(N = N_adaptive, dist=lambda p1, p2: findWeights(p1, p2, xrange, yrange), theta=theta0)
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

##################### Illustration
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for F0 in FAll:
    dist = F0.distances
    points = F0.points
    
    burnedIdx = dist <= t0+errVal
    burned = points[burnedIdx,:]
    
    ax.plot(burned[:,0], burned[:,1], 'o', color='gray', alpha=0.5)

################################################### Geodesics
fireFront = np.zeros((Ntheta,2))
cutLociPoints = []

t = np.linspace(0, 1, N)

### Find geodesics
def Gam1(t, p0, theta):
    v = finsler_velocity_F1(theta)
    
    Gam1X = p0[0] + t * v[0]
    Gam1Y = p0[1] + t * v[1]
    return Gam1X, Gam1Y

def Gam2(t, p0, theta):
    v = finsler_velocity_F2(theta)
    
    Gam1X = p0[0] + t * v[0]
    Gam1Y = p0[1] + t * v[1]
    return Gam1X, Gam1Y

# Find refracted points
timeCritical = F1(pC[0]-p0[0], pC[1]-p0[1])
pCEndX, pCEndY = Gam2(t0-timeCritical, pC, np.pi/2)

pCY = np.linspace(pC[1], pCEndY, N_cutLoci)

theta = np.linspace(thetaC, thetaC+2*np.pi, Ntheta)

# Find endpoints for launched curves
Gam1X, Gam1Y = Gam1(t0, p0, theta)

for i,(gam1X,gam1Y) in enumerate(zip(Gam1X, Gam1Y)):
    # Curve
    GamX = p0[0] + t*(gam1X-p0[0])
    GamY = p0[1] + t*(gam1Y-p0[1])
    
    # Figure out if the curve hit cut-loci, interface, or only traverse in Q1
    C2 = cutLoci(GamX)
    idxCutLoci = GamY <= C2
    idxEtaX = GamX <= etaX
    
    idx = idxCutLoci & idxEtaX
    
    # Plot curve in Q1
    GamX = GamX[idx]
    GamY = GamY[idx]
    
    if (i % 42) == 0:
        ax.plot(GamX, GamY, 'g-', linewidth=2)
    
    # Save endpoint if not reflected/refracted
    if sum(idx) == N:
        fireFront[i,0] = gam1X
        fireFront[i,1] = gam1Y
    else: # Curve hit interface
        # Refracted curve
        t1 = F1(GamX[-1]-p0[0], GamY[-1]-p0[1])
        theta2 = findTheta2(theta[i])
        Gam2X, Gam2Y = Gam2(t0-t1, [GamX[-1],GamY[-1]], theta2)
        
        GamRX = GamX[-1] + t*(Gam2X-GamX[-1])
        GamRY = GamY[-1] + t*(Gam2Y-GamY[-1])
        
        if (i % 42) == 0:
            ax.plot(GamRX, GamRY, 'g-', linewidth=2)
        
        if sum(idxEtaX) < sum(idxCutLoci):
            fireFront[i,0] = Gam2X
            fireFront[i,1] = Gam2Y
        else:
            i = i-1

# Reflected curve
if t0 > tauPlus:
    for i,pCY0 in enumerate(pCY):
        t2 = F2(0, pCY0-pC[1])
        t3 = t0-timeCritical-t2
        Gam3X, Gam3Y = Gam1(t3, [etaX, pCY0], theta3)
        
        GamX = etaX + t*(Gam3X-etaX)
        GamY = pCY0 + t*(Gam3Y-pCY0)
        
        # Find Cut-Loci
        C2 = cutLoci(GamX)
        idx = GamY > C2
        
        GamX = GamX[idx]
        GamY = GamY[idx]
        
        if (i % 10) == 0:
            ax.plot(GamX, GamY, 'r-', linewidth=2)
        
        cutLociPoints.append(np.array([GamX[-1], GamY[-1]]))

# Plot firefront
idx = (fireFront[:,0] == p0[0]) & (fireFront[:,1] == p0[1])
ax.plot(fireFront[~idx,0], fireFront[~idx,1], 'k-', linewidth=2)

if t0 > tauPlus:
    cutLociPoints = np.array(cutLociPoints)
    ax.plot(cutLociPoints[:,0], cutLociPoints[:,1], 'k-', linewidth=2)

    ax.plot([fireFront[0,0], cutLociPoints[-1,0]], [fireFront[0,1], cutLociPoints[-1,1]], 'k-', linewidth=2)

# Ignition Point
ax.plot(p0[0], p0[1], 'ro')

# Interface
ax.axvline(x=etaX, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)

ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
ax.tick_params(labelsize=18)