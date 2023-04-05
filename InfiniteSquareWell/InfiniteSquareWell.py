import  numpy as np
import matplotlib.pyplot as plt

pi = np.pi


a = 2 * pi
xvals = np.linspace(0,a,1000)
normalization_const = (2/a)**(1/2)

def wavefunction(x,n):
    wf = normalization_const * np.sin( n * pi * x/a )
    return wf

yvals1 = wavefunction(xvals,1)
yvals2 = wavefunction(xvals,2)
yvals3 = wavefunction(xvals,3)

plt.title("Infinite square well")
plt.plot(xvals,yvals1, "b", linewidth = 1.5 ,label = '1')
plt.plot(xvals,yvals2, "k",linewidth=0.5,label = "2")
plt.plot(xvals,yvals3, "r",linewidth=0.5,label = "3")
plt.legend()
plt.xlabel("x")
plt.ylabel("$\psi$(x)")

plt.grid()
plt.show()
