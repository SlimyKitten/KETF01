import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


experimentalDataLong  = [[0,25],[0.17,0]]
experimentalDataShort = [[0,30],[0.17,0]]

def areaFromDiameter(d):
    return np.pi*d*d/4

def tank_modell(t, h, l, ksi, f,d=0.01):
    # Parametrar
    a = areaFromDiameter(d) #m2
    A = areaFromDiameter(0.175)#m2
    g = 9.82                   #m/s2
    dhdt = -a/A*np.sqrt(2*g*h/(1+ksi+4*f*l/d))
    return dhdt

def simulate_tank(l_ut,ksi=0.2,f=0.006,d=0.01):
    h0 = [l_ut+0.07+0.17]
    t_span = [0,1*60]
    sol = solve_ivp(tank_modell,t_span,h0,args=(l_ut,ksi,f,d))
    t=sol.t
    h=sol.y.T
    h-=(l_ut+0.07)
    return (t,h)

r = simulate_tank(0.40)
plt.plot(r[0], r[1], label = f'Längd på rör: {0.40}')
r = simulate_tank(0.07)
plt.plot(r[0], r[1], label = f'Längd på rör: {0.07}')
r = simulate_tank(0.07,d=0.011)
plt.plot(r[0], r[1], label = f'Längd på rör: {0.07}, större diameter')
plt.plot(experimentalDataLong[0],experimentalDataLong[1], 'x', label='Experimentell - 0.40')
plt.plot(experimentalDataShort[0],experimentalDataShort[1], 'x', label='Experimentell - 0.07')
plt.xlabel('Time [s]')
plt.ylabel('Height [m]')
plt.ylim(0)
plt.xlim(0,45)
plt.legend(loc='best')
# plt.show()

figure, axis = plt.subplots(2,2)
ksi = [0.1,0.2,0.4]
f_list = [0.003, 0.006, 0.012]
for k in ksi:
    r = simulate_tank(0.40,ksi=k)
    axis[0,0].plot(r[0],r[1],label= f'ksi={k}')
    r = simulate_tank(0.07,ksi=k)
    axis[0,1].plot(r[0],r[1],label= f'ksi={k}')
for f in f_list:
    r = simulate_tank(0.40,f=f)
    axis[1,0].plot(r[0],r[1],label= f'f={f}')
    r = simulate_tank(0.07,f=f)
    axis[1,1].plot(r[0],r[1],label= f'f={f}')
for x in range(2):
    for y in range(2):
        axis[x,y].set_ylim(0)
        axis[x,y].set_xlim(0,40)
        axis[x,y].legend(loc='best')
        if( y==0): axis[x,y].set_title('Rörlängd=0.40')
        else : axis[x,y].set_title('Rörlängd=0.07')

plt.show()
