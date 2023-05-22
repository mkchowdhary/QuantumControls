import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize as opt
from scipy.optimize import minimize
import csv
import time

import matplotlib
matplotlib.rcParams['figure.dpi'] = 50


# Loads a pulse from a CSV File
def load_pulse_from_csv(filename):
    pulse = {'times': [],
             'amps': [],
             'phases': []}
    with open(filename) as pulse_file:
        csv_read = csv.reader(pulse_file, delimiter=',')
        lines = []
        for row in csv_read:
            lines.append(row)
        for l in lines[1:-1]:
            pulse['times'].append(float(l[0]))
            pulse['amps'].append(float(l[1]))
            pulse['phases'].append(float(l[2]))
    pulse['times'].append(float(lines[-1][0]))
    pulse['steps'] = len(pulse['amps'])
    return pulse


# Plots a pulse
def plot_pulse(pulse):
    times_list = []
    amps = []
    phases = []
    for i in range(pulse['steps']):
        times_list.append(pulse['times'][i])
        times_list.append(pulse['times'][i + 1])
        for _ in range(2):
            amps.append(pulse['amps'][i])
            phases.append(pulse['phases'][i])
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 8))
    ax[0].plot(times_list, amps)
    ax[0].set_ylabel("Amplitude $|\Omega(t)|/\Omega_0$")
    ax[1].plot(times_list, phases)
    ax[1].set_ylabel("Phase $\\varphi(t)$")
    ax[1].set_xlabel("Time $t\Omega_0$")

pulse_to = load_pulse_from_csv(r'C:\Users\Manish\Fall 2022 Research\protocol\01_cz.csv')
plot_pulse(pulse_to)
pulse_to_finiteB = load_pulse_from_csv(r'C:\Users\Manish\Fall 2022 Research\protocol\finiteB.csv')

plot_pulse(pulse_to_finiteB)

counter = 0
derivative_counter = 0

# Calculates both Bell state fidelity and average gate fidelity
def calculate_pulse_infidelity(pulse, Gamma=0, B=None):
    # B=None corresponds to B = infty
    #B is the blockade strength
    global counter
    counter = counter+ 1
    psi01 = np.array([1, 0], complex)
    psi11 = np.array([1, 0, 0], complex)
    if B is None:
        psi11 = np.array([1, 0], complex)

    for i in range(pulse['steps']):
        dt = pulse['times'][i + 1] - pulse['times'][i]
        Omega = pulse['amps'][i] * np.exp(1j * pulse['phases'][i])
        H01 = 0.5 * np.array([[0, Omega], [np.conj(Omega), 0]])
        if B is None:
            H11 = np.sqrt(2) * H01
        else:
            H11 = np.sqrt(2) * 0.5 * np.array([[0, Omega, 0], [np.conj(Omega), 0, Omega], [0, np.conj(Omega), 0]])
            H11[2, 2] = B - 1j * Gamma
        H01[1, 1] = -1j * Gamma / 2
        H11[1, 1] = -1j * Gamma / 2
        psi01 = scipy.linalg.expm(-1j * H01 * dt) @ psi01
        psi11 = scipy.linalg.expm(-1j * H11 * dt) @ psi11

    # Apply single qubit gates
    phase = psi01[0] / np.abs(psi01[0])
    psi01 /= phase
    psi11 /= phase ** 2

    # print(psi01)
    # print(psi11)

    F_bell = 1 / 16 * np.abs(1 + 2 * psi01[0] - psi11[0]) ** 2
    F_av = 16 / 20 * F_bell + 1 / 20 * (1 + 2 * np.abs(psi01[0]) ** 2 + np.abs(psi11[0]) ** 2)
    return 1 - F_bell, 1 - F_av


def calculate_pulse_infidelity_opt(phases):
    # B=None corresponds to B = infty
    global counter
    counter = counter + 1
    print("counter",counter)
    print("Type of phases", type(phases))
    print("dimension of phases", len(phases))
    times = pulse_to_finiteB['times']
    Omega0 = 1
    Gamma = (1 / 170) / 2.5
    B = 3.5 / 3.5

    psi01 = np.array([1, 0], complex)
    psi11 = np.array([1, 0, 0], complex)
    if B is None:
        psi11 = np.array([1, 0], complex)


    for i in range(99):
        dt = times[i + 1] - times[i]
        Omega = Omega0 * np.exp(1j * phases[i])
        H01 = 0.5 * np.array([[0, Omega], [np.conj(Omega), 0]])
        if B is None:
            H11 = np.sqrt(2) * H01
        else:
            H11 = np.sqrt(2) * 0.5 * np.array([[0, Omega, 0], [np.conj(Omega), 0, Omega], [0, np.conj(Omega), 0]])
            H11[2, 2] = B - 1j * Gamma
        H01[1, 1] = -1j * Gamma / 2
        H11[1, 1] = -1j * Gamma / 2
        psi01 = scipy.linalg.expm(-1j * H01 * dt) @ psi01
        psi11 = scipy.linalg.expm(-1j * H11 * dt) @ psi11

    # Apply single qubit gates
    phase = psi01[0] / np.abs(psi01[0])
    psi01 /= phase
    psi11 /= phase ** 2

    # print(psi01)
    # print(psi11)

    F_bell = 1 / 16 * np.abs(1 + 2 * psi01[0] - psi11[0]) ** 2
    F_av = 16 / 20 * F_bell + 1 / 20 * (1 + 2 * np.abs(psi01[0]) ** 2 + np.abs(psi11[0]) ** 2)
    print("fidelity ",F_bell)
    return 1 - F_bell

increment_factor = 2
derivatives = []

def derivative(phases):
    global derivative_counter
    derivative_counter = derivative_counter + 1
    print("Derivative-counter", derivative_counter)
    print("Derivative-Type of phases", type(phases))
    print("Derivative-dimension of phases", len(phases))
    new_phases = phases
    for i in range(99):
        new_phases[i] = phases[i] + increment_factor
        temp = (calculate_pulse_infidelity_opt(new_phases) - calculate_pulse_infidelity_opt( \
                    phases)) / increment_factor
        derivatives.append(temp)


    return derivatives

infidelity = calculate_pulse_infidelity_opt(pulse_to['phases'])
print('infidelity',infidelity)
start_time = time.time()
# for index in range(99):
#     pulse_to['phases'][index] = pulse_to['phases'][index-1] + 2* increment_factor
#Set phase guess and pulse time for Pupillo gate
pulse_time= 7.63 # in Omega*t dimension-less units
resolution = 300 # number of phase steps in the pulse
PhaseGuess = [(-0.5*np.sin(2*np.pi*x/pulse_time)-0.5) for x in np.linspace(0,pulse_time,resolution)] #input a phase profile guess




Omega_Rabi=2*np.pi*3.5 #MHz
Blockade = 2*np.pi*7  #MHz
R_lifetime = 170 # microseconds
#print(PhaseGuess)
#phases_out = opt.minimize(fun=calculate_pulse_infidelity_opt,x0=pulse_to['phases'],method='BFGS',jac=derivative,tol=1e-8)
phases_out = opt.minimize(fun=calculate_pulse_infidelity_opt,x0=PhaseGuess,method='BFGS',options={"maxiter":11000})
end_time = time.time()

print("total time=",end_time-start_time)
#phases_out = opt.minimize(fun=calculate_pulse_infidelity_opt,x0=pulse_to['phases'])
print(phases_out)


