import math

import NoiseSynthesis as sigen
from qm.qua import *
from qm.simulate.credentials import create_credentials
from qm.QuantumMachinesManager import (
    QuantumMachinesManager,
    SimulationConfig
)
import matplotlib.pyplot as plt
import numpy as np


config = {
"version" : 1,

"controllers":{
       "con1":{
            "type": "opx1",
            "analog_outputs":{
                1:{"offset": 0.0},
            },
        },
},

"elements":{
        "qe1": {
                "singleInput": {"port": ("con1", 1)},
                "intermediate_frequency": 100e6,
                "operations": {
                    "playOp": "opxPulse",
                },
        },
    },
    "pulses": {
        "opxPulse": {
            "operation": "control",
            "length": 1000, #in ns
            #"waveforms": {"single": "const_wf"},
            "waveforms": {"single": "arb_wf"},
           },
    },
    "waveforms": {
        #"const_wf": {"type": "constant", "sample": 0.2},
        "arv_wf": {"type": "arbitrary", "samples": sigen.fieldIntensityList},

    },
}

with program() as white_noise:
    play("playOp", "qe1")

#define configurations
QMm = QuantumMachinesManager(host='###',
                            port=###, credentials=create_credentials())  #creates a manager instance

QMm.close_all_quantum_machines()
QM1 = QMm.open_qm(config)
job = QM1.simulate(white_noise, SimulationConfig(int(250)))

simulatedWave = job.get_simulated_samples()
simulatedSamples = simulatedWave.con1.analog["1"]

print('num Of Samples=',len(simulatedSamples))
#print('real values of simulated wave samples',simulatedSamples)
#plt.xlim(180,320)
plt.figure(figsize = (8, 6))

plt.ylabel('Amplitude')
plt.xlabel("Time -- NS")
plt.plot(simulatedSamples)

#Find the DFT of the signal output
DFTOutput = np.fft.fft(simulatedSamples)
# plt.ylabel('Amplitude -- real values')
# plt.xlabel("Frequency - Hz")
# plt.xlim(200,300)
# print('DFT output',DFTOutput.real)
# plt.plot(DFTOutput.real)
#
#
#
# lenDFTOutput = len(DFTOutput)
# lenDFTOutput = lenDFTOutput // 2
# print('lenDFTOutput',lenDFTOutput)
#

DFTFrequencies = np.fft.fftfreq(len(simulatedSamples),1e-9)
for index in range(len(DFTFrequencies)):
    if  0 == (abs(DFTOutput[index])**2):
        DFTOutput[index] = 0
    else:
        DFTOutput[index] = math.log((abs(DFTOutput[index]))**2)

plt.figure(figsize = (8, 6))
plt.xlim(-5e8,5e8)
plt.ylim(0,10)
plt.ylabel('Amplitude')
plt.xlabel("Time -- NS")
plt.plot(DFTFrequencies,DFTOutput)
# index = 0
# #for index in range(lenDFTOutput):
# #     DFTFrequencies[index] = round(DFTFrequencies[index],0)
# #
# rearrangeDFTFrequencies = []
# index = 0
# #
# halfLength = lenDFTOutput // 2
# print('half-length',halfLength)
# #regarrange the data
# for index in range(halfLength):
#     rearrangeDFTFrequencies.append(DFTFrequencies [(halfLength) - index])
#
# for index in range(halfLength):
#      rearrangeDFTFrequencies.append(DFTFrequencies[lenDFTOutput-1] - index)
#
# print('rearrangeDFTFrequencies length = ',len(rearrangeDFTFrequencies))
# print('rearrangeDFTFrequencies',rearrangeDFTFrequencies)
#
# #
# powerSepctrum = [];
# dftSample_index = 0;
# # #
# for dftSample_index in range(lenDFTOutput):
#      powerSepctrum.append(round(((np.conj(DFTOutput[dftSample_index]) * DFTOutput[dftSample_index]).real),0))
#
# print('powerSpectrum',powerSepctrum)
# plt.plot(rearrangeDFTFrequencies,powerSepctrum)
# #
# plt.xlim(-150,-80)
# plt.show()
