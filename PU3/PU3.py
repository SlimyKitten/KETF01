import numpy as np
from scipy.constants import R,pi
from scipy.integrate import solve_ivp 
from matplotlib import pyplot as plt
rho_L=998.2 #densitet @20degC
rho_G=1.189
g=9.82

def size_bubble(d0=0.1E-3):
    sigma = 0.072
    return ((6*sigma*d0)/(g*(rho_L-rho_G)))**(1/3)

def terminal_rise_velocity(d_bubble=size_bubble()):
    C_D=0.5
    return ((4*d_bubble*(rho_L-rho_G)*g)/(3*C_D*rho_L))**(1/2)

def model(t, c):
    h=0.2
    q_g = 2E-3/60
    D_O2_Luft = 1.76E-5
    D_02_H2O = 1.8E-9
    delta_x=0.1E-3
    H = 82547
    atm = 101325
    d_tank = 0.16
    V = h*(d_tank/2)**2*pi
    p_O2 = 0.21*atm
    k_G = D_O2_Luft/(R*298.15*delta_x)
    k_L = D_02_H2O/delta_x
    K_L = (1/k_L+1/(H*k_G))**-1
    N_a = K_L*(p_O2/H-c)
    d_bubble = size_bubble()
    v_t=terminal_rise_velocity(d_bubble=d_bubble)
    t_bubble = h/v_t
    vol_b = 4*pi*(d_bubble/2)**3/3
    num_bubbles = q_g*h/(vol_b*v_t)
    A=num_bubbles*pi*d_bubble**2
    dcdt = A*N_a/V
    return dcdt

# print(model(0,0))

t_span = [0,5*60*60]
c0=[0]
sol = solve_ivp(model,t_span,c0)
print(sol.t)
plt.plot(sol.t, sol.y.T, label='Modelldata')

plt.show()