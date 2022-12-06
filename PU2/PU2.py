import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize,root
from scipy.integrate import solve_ivp

def pressure_from_temperature(T):
    T=T-273.15
    lgp_s=10.19625-1730.630/(T+233.426)
    return 10**lgp_s

exp_times = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
exp_times = [x*60  for x in exp_times]
exp_ULock = [91, 86, 83, 80, 77, 72, 68, 64, 61, 58, 56, 53.5, 52, 47, 44, 41, 39, 36.5, 35, 33, 31.5, 30.5, 29, 28, 28, 27, 26]
exp_MLock = [92, 90, 88.5, 85.5, 84, 80, 77, 75, 72, 70, 68, 66, 64, 59.5, 56, 52.8, 49.5, 47.2, 45, 43, 41, 39.5, 38, 36.5, 36, 35, 34]

def model(t, T, lock:bool, verbose=False, HEIGHT=8E-2, diameter=6.5E-2,MASSA=180E-3,ProcentTopp =1,papptjocklek=1E-3,ALPHA_H2O=300):
    T=T+273.15
    SIGMA= 5.67E-8                  #W*m^-2*K^-4
    A_TOPP = (diameter+1E-2)**2/4*np.pi*ProcentTopp      #m^2
    A_SIDA = np.pi*diameter*HEIGHT    #m^2
    E_H20=0.96                      #-
    E_PAPP=0.93                     #-
    D_IN=6.5E-2                     #m
    D_OUT=D_IN+papptjocklek         #m
    ALPHA_LUFT=6.5                  #W/(m^2*K)
    LAMBDA_PAPP=0.18                #W/(m*K)
    K=6.5E-3                        #m/s
    T_OUT = 25+273.15               #K
    R=8.3145                        #J/(mol*K)
    H_VAP_H2O = 42.03E3             #J/mol
    C_P=4186                        #J/(kg*K)

    if(lock):
        E_H20=E_PAPP

    #Strålningsvärme
    Q_stral = (A_TOPP*E_H20+A_SIDA*E_PAPP)*SIGMA*(T**4-T_OUT**4)

    #Q_Vägg
    k_inv = D_OUT/(D_IN*ALPHA_H2O) + D_OUT*np.log(D_OUT/D_IN)/(2*LAMBDA_PAPP)+1/ALPHA_LUFT
    k=1/k_inv
    Q_vagg = k*A_SIDA*(T-T_OUT)

    #Q_Vätska
    k_inv = 1/ALPHA_H2O+1/ALPHA_LUFT
    if(lock):
        k_inv+=1E-3/LAMBDA_PAPP
    k=1/k_inv
    Q_vatska=k*A_TOPP*(T-T_OUT)

    #Q_vap
    p_s=pressure_from_temperature(T)
    p0=pressure_from_temperature(T_OUT)*0.5
    N_a=K*(p_s/(R*T)-p0/(R*T_OUT))
    Q_vap = H_VAP_H2O*A_TOPP*N_a
    if(lock):
        Q_vap=0

    dTdt=-(Q_stral+Q_vagg+Q_vap+Q_vatska)/(MASSA*C_P)
    # print(H_VAP_H2O,M_H2O,A_TOPP,N_a)
    if(verbose):
        print(f"{Q_stral=},{Q_vagg=},{Q_vap=},{Q_vatska=}")
    # print(dTdt)
    return dTdt

def optimizeTemp(tGoal=20, TGoal=50, lock=False):
    return minimize(lambda T: (timeForXdeg(T[0],TGoal,lock=lock)/60-tGoal)**2, [85], method='powell').x

def timeForXdeg(T0,TGoal, lock):
    t_span = [0,60*60]
    sol = solve_ivp(lambda t,T: model(t,T,lock), t_span, [T0], t_eval=np.linspace(*t_span,1000))
    time = timeForClosestTemp(tGoal=TGoal, times=sol.t, temps=sol.y.T)
    return time

def timeForClosestTemp(tGoal, times, temps):
    bestTempDiff = np.Inf
    bestTime = np.Inf
    for (T,t) in zip(temps, times):
        if(abs(T-tGoal)<bestTempDiff):
            bestTempDiff = abs(T-tGoal)
            bestTime = t
    return bestTime


# model(0,80,lock=True, verbose=True)
T_0 = [90]
t_span = [0,60*60*1.5]
sol_uLock = solve_ivp(lambda t,T: model(t,T,False),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
plt.plot(sol_uLock.t, sol_uLock.y.T, label='Modelldata utan lock')
sol_mLock = solve_ivp(lambda t,T: model(t,T,True),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
plt.plot(sol_mLock.t, sol_mLock.y.T, label='Modelldata med lock')

plt.plot(exp_times,exp_ULock, 'x', label='Experimentell data utan lock')
plt.plot(exp_times,exp_MLock, 'x', label='Experimentell data med lock')

sol_uLock_stor = solve_ivp(lambda t,T: model(t,T,False,HEIGHT=16E-2,diameter=2*6.5E-2, MASSA=0.18*8),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
plt.plot(sol_uLock_stor.t, sol_uLock_stor.y.T, label='Modelldata-stor utan lock')
sol_uLock_stor = solve_ivp(lambda t,T: model(t,T,lock=True,HEIGHT=16E-2,diameter=2*6.5E-2, MASSA=0.18*8),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
plt.plot(sol_uLock_stor.t, sol_uLock_stor.y.T, label='Modelldata-stor med lock')
sol_mLock = solve_ivp(lambda t,T: model(t,T,False,ProcentTopp=0.5),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
plt.plot(sol_mLock.t, sol_mLock.y.T, label='Modelldata Avsmalnad')
plt.legend(loc='best')
plt.xlabel='t (s)'
plt.ylabel='t (degC)'

figure, axis = plt.subplots(1,3)
alpha_h20=[200,300,10000]
for k in alpha_h20:
    sol_uLock = solve_ivp(lambda t,T: model(t,T,False,ALPHA_H2O=k),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
    axis[0].plot(sol_uLock.t, sol_uLock.y.T,label=f'AlphaH20={k}')
axis[0].legend(loc='best')
procentTopp = [0.5, 1, 2]
for k in procentTopp:
    sol_uLock = solve_ivp(lambda t,T: model(t,T,False,ProcentTopp=k),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
    axis[1].plot(sol_uLock.t, sol_uLock.y.T,label=f'ProcentTopp={k}')
axis[1].legend(loc='best')
tjocklek = [0.5, 1, 2]
for k in procentTopp:
    sol_uLock = solve_ivp(lambda t,T: model(t,T,False,papptjocklek=k),t_span,T_0,t_eval=np.linspace(*t_span,len(exp_times)))
    axis[2].plot(sol_uLock.t, sol_uLock.y.T,label=f'Tjocklek={k}')
axis[2].legend(loc='best')

sol_stat = root(lambda t: model(0,t,False), 90) # Hitta stationär temperatur
print(sol_stat)
# print(optimizeTemp())
# print(timeForXdeg(90,50,False))


print(optimizeTemp())
print(optimizeTemp(lock=True))

plt.show()

