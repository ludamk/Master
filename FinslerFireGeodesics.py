import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.integrate import solve_ivp
import time

#################################### Finsler Fire ####################################
############## The following fire is a Finsler fire of type Randers
############## In Master:
############## See Section "Randers Metric" for Indicatrix field
############## See Section "Finsler: Randers" (Randers 2) for fire simulation
####################################
################### Constants
# Controls the underlying Riemannian ellipse (major and minor axes)
a = 1
b = 1.1

# Controls the wind (amplitude)
wAmp = 0.3

# Range
xrange = np.array([-8,8])
yrange = np.array([-8,11])

ignitionPoint = np.array([0,0])
t0 = 7
errVal = 0.001

# Geodesics
N_geodesics = 100
#N_geodesics = 200

# Either choose simply alphaArr or if more focuses directions also add alphaArr1 and alphaArr2
alphaArr = np.linspace(0, 2*np.pi, N_geodesics, endpoint=False)
alphaArr1 = np.linspace(np.pi/3, 2*np.pi/3, 50)
alphaArr2 = np.linspace(3*np.pi/3, 5*np.pi/3, 50)

alphaArr = np.concatenate((alphaArr, alphaArr1))
alphaArr = np.concatenate((alphaArr, alphaArr2))
alphaArr.sort()

# Time interval
tau_span = (0, t0)
N_tau = 5000

# If boundaryNewFlag --> New boundaryPoints
# If boundaryPlotFlag --> Plot boundaryPoints
boundaryNewFlag = True
boundaryPlotFlag = True

#########################
############### Finsler metric
# p = (u1,u2)
# v = (v1,v2)

def G(u1,u2,v1,v2):    
    g11 = ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) / a ** 2 / 2 - wAmp * np.sin(u1)) ** 2 / (b ** 2 - wAmp ** 2) ** 2 + (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * (-(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.3e1 / 0.2e1) * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) ** 2 / a ** 4 / b ** 2 / 4 + (((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * ((2 * b ** 2 - 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * a ** 2 * np.sin(u1) ** 2) / a ** 2 / 2)
    g12 = ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) / a ** 2 / 2 - wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) / a ** 2 / 2 - wAmp * np.sin(u1)) + (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * (-(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.3e1 / 0.2e1) * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) / a ** 4 * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) / b ** 2 / 4 + (((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) / a ** 2)
    g21 = g12
    g22 = ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) / a ** 2 / 2 - wAmp * np.cos(u1)) ** 2 / (b ** 2 - wAmp ** 2) ** 2 + (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * (-(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.3e1 / 0.2e1) * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) ** 2 / a ** 4 / b ** 2 / 4 + (((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (2 * a ** 2 * np.cos(u1) ** 2 + np.sin(u1) ** 2 * (2 * b ** 2 - 2 * wAmp ** 2)) / a ** 2 / 2)

    return g11, g12, g21, g22
    
def inverse_G(u1,u2,v1,v2):
    g11,g12,g21,g22 = G(u1,u2,v1,v2)
    
    Gval = np.array([[g11,g12],[g21,g22]])
    
    try:
        G_inv = np.linalg.inv(Gval)
        return G_inv[0,0],G_inv[0,1],G_inv[1,0],G_inv[1,1]
    
    except np.linalg.LinAlgError:
        print("Matrix is singular and no inverse exist.")
        return None

def F(u1,u2,v1,v2):
    return (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2)

def dF2u1v1(u1,u2,v1,v2):
    return 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) / a ** 2 / 2 - wAmp * np.sin(u1)) / (b ** 2 - wAmp ** 2) ** 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (-2 * (v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v1 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2) * np.cos(u1)) / a ** 2 / 2 - v1 * wAmp * np.cos(u1) + v2 * wAmp * np.sin(u1)) + 2 * (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * (-(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.3e1 / 0.2e1) * (-2 * (v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v1 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2) * np.cos(u1)) / a ** 4 * ((2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + 2 * np.sin(u1) ** 2 * a ** 2 * v1) / b ** 2 / 4 + (((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (-2 * (2 * b ** 2 * v1 - 2 * v1 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 4 * np.sin(u1) * a ** 2 * v1 * np.cos(u1)) / a ** 2 / 2 - wAmp * np.cos(u1))

def dF2u2v1(u1,u2,v1,v2):
    return np.zeros_like(u1)

def dF2u1v2(u1,u2,v1,v2):
    return 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) / a ** 2 / 2 - wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (-2 * (v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v1 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2) * np.cos(u1)) / a ** 2 / 2 - v1 * wAmp * np.cos(u1) + v2 * wAmp * np.sin(u1)) + 2 * (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * (-(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.3e1 / 0.2e1) * (-2 * (v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v1 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2) * np.cos(u1)) / a ** 4 * (2 * a ** 2 * v2 * np.cos(u1) ** 2 + 2 * v1 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2)) / b ** 2 / 4 + (((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (-4 * a ** 2 * v2 * np.cos(u1) * np.sin(u1) + 2 * v1 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (2 * b ** 2 * v2 - 2 * v2 * wAmp ** 2) * np.cos(u1)) / a ** 2 / 2 + wAmp * np.sin(u1))

def dF2u2v2(u1,u2,v1,v2):
    return np.zeros_like(u1)

def dF2u1(u1,u2,v1,v2):
    return 2 * (np.sqrt(((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) * b ** 2 - v1 * wAmp * np.sin(u1) - v2 * wAmp * np.cos(u1)) / (b ** 2 - wAmp ** 2) ** 2 * ((((v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) ** 2 + 2 * v1 * v2 * np.sin(u1) * (a ** 2 - b ** 2 + wAmp ** 2) * np.cos(u1) + np.sin(u1) ** 2 * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2)) / a ** 2 / b ** 2) ** (-0.1e1 / 0.2e1) * (-2 * (v2 ** 2 * a ** 2 + v1 ** 2 * b ** 2 - v1 ** 2 * wAmp ** 2) * np.cos(u1) * np.sin(u1) + 2 * v1 * v2 * np.cos(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) - 2 * v1 * v2 * np.sin(u1) ** 2 * (a ** 2 - b ** 2 + wAmp ** 2) + 2 * np.sin(u1) * (v1 ** 2 * a ** 2 + v2 ** 2 * b ** 2 - v2 ** 2 * wAmp ** 2) * np.cos(u1)) / a ** 2 / 2 - v1 * wAmp * np.cos(u1) + v2 * wAmp * np.sin(u1))

def dF2u2(u1,u2,v1,v2):
    return np.zeros_like(u1)

def findLength(u1,u2,v1,v2,tau):    
    speed = F(u1,u2,v1,v2)
    L = cumtrapz(speed, tau, initial=0)
    
    return L

############### Setup for Geodesic Equations
def G1Func(u1,u2,v1,v2):
    # Inverse of metric matrix
    g11, g12, g21, g22 = inverse_G(u1,u2,v1,v2)
    
    # Derivative
    dF2u1v1_val = dF2u1v1(u1,u2,v1,v2)
    dF2u2v1_val = dF2u2v1(u1,u2,v1,v2)
    dF2u1v2_val = dF2u1v2(u1,u2,v1,v2)
    dF2u2v2_val = dF2u2v2(u1,u2,v1,v2)
    dF2u1_val = dF2u1(u1,u2,v1,v2)
    dF2u2_val = dF2u2(u1,u2,v1,v2)
    
    term1 = g11*(dF2u1v1_val*v1 + dF2u2v1_val*v2 - dF2u1_val)
    term2 = g12*(dF2u1v2_val*v1 + dF2u2v2_val*v2 - dF2u2_val)
    
    return 0.25*(term1 + term2)

def G2Func(u1,u2,v1,v2):
    # Inverse of metric matrix
    g11, g12, g21, g22 = inverse_G(u1,u2,v1,v2)
    
    # Derivative
    dF2u1v1_val = dF2u1v1(u1,u2,v1,v2)
    dF2u2v1_val = dF2u2v1(u1,u2,v1,v2)
    dF2u1v2_val = dF2u1v2(u1,u2,v1,v2)
    dF2u2v2_val = dF2u2v2(u1,u2,v1,v2)
    dF2u1_val = dF2u1(u1,u2,v1,v2)
    dF2u2_val = dF2u2(u1,u2,v1,v2)
    
    term1 = g21*(dF2u1v1_val*v1 + dF2u2v1_val*v2 - dF2u1_val)
    term2 = g22*(dF2u1v2_val*v1 + dF2u2v2_val*v2 - dF2u2_val)
    
    return 0.25*(term1 + term2)

def geodesic_equations(tau, Y):
    u1, u2, v1, v2 = Y
    
    # Compute spray geodesic coefficients by using G1Func and G2Func
    G1 = G1Func(u1, u2, v1, v2)
    G2 = G2Func(u1, u2, v1, v2)
    
    # Geodesic ligninger
    du1_dtau = v1
    du2_dtau = v2
    d2u1_dtau = -2 * G1      # u1''
    d2u2_dtau = -2 * G2      # u2''
    
    return [du1_dtau, du2_dtau, d2u1_dtau, d2u2_dtau]

def normalize_speed(u1_0, u2_0, alpha0, speed=1.0):
    # Speed in tangentplane
    v1_0 = speed * np.cos(alpha0)
    v2_0 = speed * np.sin(alpha0)
    
    # Compute norm factor
    Fvals = F(u1_0, u2_0, v1_0, v2_0)
    norm_factor = Fvals
    
    # Normalise speed
    if norm_factor != 0:
        v1_0 = v1_0 / norm_factor
        v2_0 = v2_0 / norm_factor

    return v1_0, v2_0

############### Solve Geodesic Equations
u1_0 = ignitionPoint[0]
u2_0 = ignitionPoint[1]

geodesicAll = []

startGeodesic = time.time()

for i,alpha0 in enumerate(alphaArr):
    subStart = time.time()
    v1_0, v2_0 = normalize_speed(u1_0, u2_0, alpha0)
    initial_conditions = [u1_0, u2_0, v1_0, v2_0]
    
    # Solve PDE
    solution = solve_ivp(geodesic_equations, tau_span, initial_conditions, t_eval=np.linspace(tau_span[0], tau_span[1], N_tau), max_step=0.01)
    u1_sol, u2_sol, v1_sol, v2_sol = solution.y
    tau_sol = solution.t
    
    L_tau = findLength(u1_sol, u2_sol, v1_sol, v2_sol, tau_sol)
    
    # Save solutions
    geodesicAll.append({'u1_sol': u1_sol, 'u2_sol': u2_sol, 'v1_sol': v1_sol, 'v2_sol': v2_sol, 'tau_sol': tau_sol, 'L_tau': L_tau})
    
    subEnd = time.time()
    print(f"Sub-time (Geodesic: {i+1}): {(subEnd - subStart)/60:.3f} minutes")

endGeodesic = time.time()

print(f"Time (Geodesic): {(endGeodesic - startGeodesic)/60:.3f} minutes")

### Save boundary
if boundaryNewFlag:
    boundaryPoints = np.zeros((len(alphaArr)+1, 2))
    
    for i,geo0 in enumerate(geodesicAll):
        u1 = geo0.get('u1_sol')
        u2 = geo0.get('u2_sol')
        L_tau = geo0.get('L_tau')
        
        idx = L_tau <= t0+errVal
        
        u1 = u1[idx]
        u2 = u2[idx]
        
        boundaryPoints[i,0] = u1[-1]
        boundaryPoints[i,1] = u2[-1]
    
    boundaryPoints[-1,0] = boundaryPoints[0,0]
    boundaryPoints[-1,1] = boundaryPoints[0,1]
    
endOverAll = time.time()

print(f"Time (All): {(endOverAll - startGeodesic)/60:.3f} minutes")

############### Illustrate Solutions
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for i,geo0 in enumerate(geodesicAll):
    if (i % 6) == 0:
        u1 = geo0.get('u1_sol')
        u2 = geo0.get('u2_sol')
        L_tau = geo0.get('L_tau')
        
        idx = L_tau <= t0+errVal
        
        ax.plot(u1[idx], u2[idx], 'g-', linewidth=2.5)

ax.plot(ignitionPoint[0], ignitionPoint[1], 'ro')

if boundaryPlotFlag:
    ax.plot(boundaryPoints[:,0], boundaryPoints[:,1], 'k-')

ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.set_xlabel(r'$u^1$', fontsize=28)
ax.set_ylabel(r'$u^2$', fontsize=28)
ax.tick_params(labelsize=25)
