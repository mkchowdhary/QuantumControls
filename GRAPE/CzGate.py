import matplotlib.pyplot as plt
import time
import numpy as np

from qutip import *
from qutip.control import *
from qutip.qip.circuit import QubitCircuit

T = 2 * np.pi
times = np.linspace(0, T, 500)

U = cz_gate()
R = 500
H_ops = [tensor(sigmax(), identity(2)),
         tensor(sigmay(), identity(2)),
         tensor(sigmaz(), identity(2)),
         tensor(identity(2), sigmax()),
         tensor(identity(2), sigmay()),
         tensor(identity(2), sigmaz()),
         tensor(sigmax(), sigmax()) +
          tensor(sigmay(), sigmay()) +
          tensor(sigmaz(), sigmaz())]

H_labels = [r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$',
            r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$',
            r'$u_{xx}$',
            r'$u_{yy}$',
            r'$u_{zz}$',
        ]
H0 = 0 * np.pi * (tensor(sigmax(), identity(2)) + tensor(identity(2), sigmax()))

c_ops = []

from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary

from scipy.interpolate import interp1d
from qutip.ui.progressbar import TextProgressBar

u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.05 for _ in range(len(H_ops))])

u0 = [np.convolve(np.ones(10)/10, u0[idx,:], mode='same') for idx in range(len(H_ops))]

u_limits = None #[0, 1 * 2 * pi]
alpha = None

print("cy_grape_unitary")
result = cy_grape_unitary(U, H0, H_ops, R, times, u_start=u0, u_limits=u_limits,
                          eps=2*np.pi*1, alpha=alpha, phase_sensitive=False,
                          progress_bar=TextProgressBar())

plot_grape_control_fields(times, result.u / (2 * np.pi), H_labels, uniform_axes=True);

print(U)
print(result.U_f)
print(result.U_f/result.U_f[0,0])
olap = _overlap(U, result.U_f).real, abs(_overlap(U, result.U_f)) ** 2
print(olap)



