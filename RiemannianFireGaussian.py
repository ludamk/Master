import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import graph
import fire
import adaptiveHexagonMesh

#################################### Gaussian Fire ####################################
################### Constants
# Surface: r(u1,u2) = c1*exp(-c2*(u1**2 + u2**2))
c1 = 10
c2 = 0.5

# Constants to rotate graph
angles = 4
fullAngle = np.pi/3
thetaAll = np.linspace(0, fullAngle, angles, endpoint=False)

# Fire constants
ignitionPoint = np.array([2,0])
extinguishFactor = 1
errVal = 0.01

# Area
xrange = np.array([-7,7])
yrange = np.array([-7,7])

# N: Describe size of basemesh in adaptive hexagon mesh
N = 2

# Time
Tmax = 20
t0 = 5

###################################### Functions
def G(u1,u2):
    g11 = 1 + 4 * c1 ** 2 * c2 ** 2 * u1 ** 2 * np.exp(-c2 * (u1 ** 2 + u2 ** 2)) ** 2
    g12 = 4 * c1 ** 2 * c2 ** 2 * u1 * np.exp(-c2 * (u1 ** 2 + u2 ** 2)) ** 2 * u2
    g21 = g12
    g22 = 1 + 4 * c1 ** 2 * c2 ** 2 * u2 ** 2 * np.exp(-c2 * (u1 ** 2 + u2 ** 2)) ** 2
    
    return np.array([[g11,g12], [g21,g22]])

####### Weight Function
def generateWeight(G_func):
    # G_func(u,v)
    # Straight line between p = (p1,p2) and q = (q1,q2)
    def gam_func(t, p1, p2, q1, q2):
        p = np.array([p1, p2])
        q = np.array([q1, q2])
        return p + t * (q - p)

    # Constant tangent along the curve
    def dgam_func(p1, p2, q1, q2):
        return np.array([q1 - p1, q2 - p2])

    # Build integrand
    def build_integrand(p1, p2, q1, q2):
        def integrand(t):
            p = gam_func(t, p1, p2, q1, q2)
            v = dgam_func(p1, p2, q1, q2)
            G = np.array(G_func(p[0], p[1])).reshape((2, 2))
            return np.sqrt(v @ (G @ v))
        return integrand

    # Integrate over t in [0,1]
    def integral_I_func(p1, p2, q1, q2):
        integrand = build_integrand(p1, p2, q1, q2)
        val, _ = quad(integrand, 0, 1)
        return val

    return integral_I_func

w_func = generateWeight(G)

def findWeights(p1, p2, xrange, yrange):
    if ((p1[0] < xrange[0] or p1[0] > xrange[1] or p1[1] < yrange[0] or p1[1] > yrange[1]) and
        (p2[0] < xrange[0] or p2[0] > xrange[1] or p2[1] < yrange[0] or p2[1] > yrange[1])):
        return 0  # Return 0 if both points are outside region
    else:
        return w_func(p1[0], p1[1], p2[0], p2[1])

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
    M = adaptiveHexagonMesh.adHexMesh(N = N, dist=lambda p1, p2: findWeights(p1, p2, xrange, yrange), theta=theta0)
    M.generateBaseMesh(rotationPoint=ignitionPoint)
    M.generateAdaptiveHexMesh()
    M.generateGraph(ignitionPoint=ignitionPoint)
    
    pointsRot = M.points
    gRot = M.g
    fireIdx = graph.findNode(pointsRot, ignitionPoint)
    
    ############# Fire
    FRot = fire.Fire(gRot, fireIdx, pointsRot)
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

ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
ax.tick_params(labelsize=18)
ax.grid(True)
