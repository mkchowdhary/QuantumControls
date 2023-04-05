import numpy as np
import matplotlib.pyplot as plt


#define the span over which wave fuction needs to be computed

pi = np.pi
a = 2 * pi

#generate the divisions for the x-axis over a span from 0 to a
xrange = np.linspace(0,a,1000)

#defind the normalization constant
A = (1/a)**(1/2)

#defind the wave function without the time dependence
def wf_time(n,x ,t):
    Psi = 0
    for index in range(n):
        Psi = Psi + A*np.sin(n*pi*x/a) * np.real(np.exp(-1j * t * ((n**2)*(pi**2)/(2* (a**2)))))
    return Psi

Psi = []
n = 4
for t in range(n):
    Psi.append(wf_time(6,xrange,t))

for index in range(n):
    plt.plot(xrange,Psi[index])
# Psi1_t1 = wf_time(1,xrange,1)
# Psi1_t2 = wf_time(1,xrange,2)
# #Psi_t10 = wf_time(50,xrange,20)
# plt.plot(xrange,Psi1_t0,label="t=0")
# plt.plot(xrange,Psi1_t1,label="t=1")
# plt.plot(xrange,Psi1_t2,label="t=2")
# #plt.plot(xrange,Psi_t10)
# plt.legend()
plt.grid()
plt.show()









