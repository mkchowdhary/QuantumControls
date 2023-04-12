# from cmath import exp
import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt
import math

import numpy.random

#For the math logic https://arxiv.org/abs/2210.11007 (Sensitivity of quantum gate fidelity to laser phase and intensity noise), should be referred.

#The frequency values are specified in Hz ( refer to the equation 39 in the paper)
whiteNoisePwrSpecAmp = 13  #(Hz**2/Hz) (h0 variable in the equation)
survoBumpPwrSpecAmp = 25 # (hg variable in the equation
survoBumpPwrSpecCentralFreq = 100000  #Hz (sigma_g)
survoBumpPwrSpecGaussianWidth = 18000  #Hz  (fg)


#*************************************************************************************************************
#Class whiteNoisePwrSpec
#Implement the white noise power spectrum  generator
#*************************************************************************************************************
class whiteNoisePwrSpec(object):
    def __init__(self,amp):
        self.amp = amp;
    def getWhiteNoisePwr(self):
        return self.amp;

#*************************************************************************************************************
#Class servoBumpPwrSpec
#Implement the servoBump power spectrum generator
#Methods
#getServoBumpPwrForGivenFreq
#getServoBumpAmp
#*************************************************************************************************************
class servoBumpPwrSpec(object):
    def  __init__(self,amp,centralFreq,gaussianWidth):
        """
        amp - Amplitude of the servo bump frequency
        centralFreq - Central frequency of servo bump
        gaussianWidth - Widh of the servo bump
        """
        self.amp = amp
        self.centralFreq = centralFreq
        self.gaussianWidth = gaussianWidth

    #Computes the  ervo bump power
    def getServoBumpPwrForGivenFreq(self,spectralFreq):
        return  self.amp * np.exp((-(spectralFreq - self.centralFreq)**2) / (2*(self.gaussianWidth)**2)) + \
                self.amp * np.exp((-(spectralFreq + self.centralFreq)**2) / (2*(self.gaussianWidth)**2))

    def  getServoBumpAmp(self):
        return self.amp
    def  getCentralFreq(self):
        return self.centralFreq
    def  getGaussianWidh(self):
        return self.gaussianWidth

#*************************************************************************************************************
#Class combinedNoisePwrSpec
#Implement the noise that is combination of white nose and servo bump spectrum
#Methods
#getCombinedNoisePwrForGivenFreq
#*************************************************************************************************************
class combinedNoisePwrSpec(object):
    def __init__(self,whiteNoisePwrSpecInst,servoBumpPwrSpecInst):
        self.whiteNoisePwrSpec = whiteNoisePwrSpecInst
        self.survoBumpPwrSpec = servoBumpPwrSpecInst

    def getCombinedNoisePwrForGivenFreq(self,spectralFreq):
        #refer to the equation 40 of the document
        return ((self.whiteNoisePwrSpec.getWhiteNoisePwr() + self.survoBumpPwrSpec.getServoBumpPwrForGivenFreq(spectralFreq)) / (survoBumpPwrSpecCentralFreq ** 2));

#Create an instance of the class whiteNoisePwrSpec
whiteNoiseSpec = whiteNoisePwrSpec(whiteNoisePwrSpecAmp)

#Create an instance of the class servoBumpPwrSpec
survoBumpSpec = servoBumpPwrSpec(survoBumpPwrSpecAmp,survoBumpPwrSpecCentralFreq,survoBumpPwrSpecGaussianWidth)

#Create an instance of the class combinedNoisePwrSpec
combinedNoisePwrSpec = combinedNoisePwrSpec(whiteNoiseSpec, survoBumpSpec)

#Generate three lists
#1. List of frequencies to be used in the generation of phase noise.
#2. List containing the time instances at which phase is to be generated.
#3. List containing the computed phase outputs
#define  a list of frequencies comprising the noise spectrum
freuencyList = []
timeList = []
computedPhaseList = []

#pouplate the freuencyList with the 100 frequencies starting with an initial value and then adding a delta
initialFrequency = survoBumpPwrSpecCentralFreq - (3 * survoBumpPwrSpecGaussianWidth) #kHz

# Parameter to get incremental frequencies
frequencyDelta = 1000

#The frequency doesn't needs to have a factor of 10^3 here because the gaussian equations are ratio
#of the frequency.
numFrequencies = int((6 * survoBumpPwrSpecGaussianWidth) / frequencyDelta)

#pouplate the timeList with the 100 time valuses starting with an initial value and then adding a delta
initialTime = 0

frequencyOfLaserField = 100e6 ##Hz
carrierFrequency = 100e6 #Hz

nextFrequency = 0
fieldAmplitude = 0.45
fieldIntensityList = []
timeDepPhaseList = []

#Start
#Following variables are not meant for primary logic of the code, but used for debugging purpose
#Gaussian plot points
combinedNoisePwrList = []
combinedNoisePwrIntList = []
cosineTermList = []
gaussianpoint = 0
nextTime = 0
tempTimeDepPhaseList = []
timeDepPhaseList = []
timeValueList = []
fieldIntensityCosineTermList = []
tempTimeDependentPhase = 0
randomPhaseList = []
nextTime = 0
cosineTerm = 0

#end


durationOfSimulation = 1e-6  #in seconds
samplingRate = 1e9  #number of samples per second
samplingInterval = 1.0/samplingRate #time interval between two samples in seconds

#numSamples = int(durationOfSimulation * samplingRate)
numSamples = 50000
#OR
#numSamples = durationOfSimulation * samplingRate
# numTimeValues = int(durationOfSimulation * samplingRate)

#create a list of random phases
def gen_randomphase():
    for index in range(numFrequencies):
        randomPhase =  numpy.random.uniform(0.0, 2 * np.pi)
        randomPhaseList.append(randomPhase)
    #print(randomPhaseList)

for index in range(numSamples):
    nextTime = initialTime + index * samplingInterval
    #print(nextTime)
    timeValueList.append(nextTime)

servoBumpFrequencyList = [];
for index in range(numFrequencies):
    servoBumpFrequencyList.append(initialFrequency + (index) * frequencyDelta);

def create_modulating_values():
    fieldIntensityList.clear()
    gen_randomphase()
    for t_index in range(numSamples):
        tempTimeDependentPhase = 0
        combinedNoisePwrList.clear()
        combinedNoisePwrIntList.clear()
        tempTimeDepPhaseList.clear()
        cosineTermList.clear()
        for index in range(numFrequencies):
            nextFrequency = initialFrequency + index * frequencyDelta
            combinedNoisePwr =(combinedNoisePwrSpec.getCombinedNoisePwrForGivenFreq(nextFrequency))
            combinedNoisePwrIntTerm = np.sqrt(combinedNoisePwr * frequencyDelta )
            combinedNoisePwrIntList.append(combinedNoisePwrIntTerm)
            cosineTerm = np.cos(2 * np.pi * nextFrequency * timeValueList[t_index] + randomPhaseList[index])
            tempTimeDependentPhase = tempTimeDependentPhase + 2 * combinedNoisePwrIntTerm * cosineTerm
        timeDepPhaseList.append(tempTimeDependentPhase)
        fieldIntensity = fieldAmplitude * (math.cos(tempTimeDependentPhase) - (math.tan(2 * np.pi * carrierFrequency * timeValueList[t_index]) * math.sin(tempTimeDependentPhase)))
        fieldIntensityList.append(fieldIntensity)

signal = [0] * numSamples
loopCount = 1
for l_index in range(loopCount):
    print('loop index=',l_index)
    create_modulating_values()
    for index in range(numSamples):
        signal[index] = signal[index]  + fieldIntensityList[index] * math.cos(2 * np.pi * carrierFrequency * timeValueList[index])

for index in range(numSamples):
    signal[index] = signal[index] / loopCount

"""
The code below is used for generating various plots for debug and analysis
"""
#plot the field intensity list against time
# plt.xlabel('time')
# plt.ylabel('FieldIntensity')
# plt.xlim(0,0.01)
# # plt.ylim(0,0.1e-7)
# plt.plot(timeValueList,fieldIntensityList)

# print('fieldIntensityCosineTermList=',fieldIntensityCosineTermList)
# print('timeDepPhaseList',timeDepPhaseList)
#print('fieldIntensityList',fieldIntensityList)
# plt.xlabel('frequency')
# plt.ylabel('Phase Power Spectrum')
# plt.xlim(-350e3,350e3)
# plt.ylim(0,0.1e-7)
# plt.plot(gausianFrequencies,combinedNoisePwrList)
#
# plt.figure(figsize = (8, 6))
# plt.xlim(initialFrequency,initialFrequency + 10 * survoBumpPwrSpecGaussianWidth)
# plt.ylim(0,2e-3)
# plt.plot(servoBumpFrequencyList,combinedNoisePwrIntList)


# combinedNoisePwrIntListplt.xlabel('frequency')
# plt.ylabel('Cos ( frequency + random phase)')
# plt.xlim(-350e3,350e3)
# plt.ylim(-1.2,1.2)
# #print(gausianFrequencies)
# plt.plot(gausianFrequencies,cosineTermList)
#
# plt.xlim(-350e3,350e3)
# plt.ylim(-10e-3,10e-3)
# plt.plot(gausianFrequencies,tempTimeDepPhaseList)
# plt.figure(figsize = (8, 6))


# plt.ylabel('time dependent phase')
# plt.xlabel('time values')
# # plt.xlim(0, numSamples * 4)
# # plt.ylim(-1,1)
# plt.plot(timeValueList,timeDepPhaseList)
#
# plt.figure(figsize = (8, 6))
# # plt.ylabel('field intensities')
# # plt.xlabel('time in ns')
# # plt.xlim(0,numSamples * 4)
# # plt.ylim(-1,1)
# print('field intensities', fieldIntensityList)
# #plt.scatter(timeValueList,fieldIntensityList)
# plt.plot(timeValueList,fieldIntensityList)


# plt.figure(figsize = (8, 6))
# plt.xlim(0, 2*durationOfSimulation)
# plt.ylim(-1,1)
# plt.ylabel('field intensity')
# plt.xlabel('time in seconds')
# plt.plot(timeValueList, signal, 'r')
#
# #Implement the amplitude modulation
# dftSamples = np.fft.fft(signal)
# print('dftSamples length',len(dftSamples))
# #arrRange = np.arange(numSamples)
# freqArray = np.fft.fftfreq(numSamples,samplingInterval)
#
# for index in range(10):
#     print(freqArray[index])
#
# powerSpectrumArray = [0] * numSamples
# for index in range(numSamples):
#     signalAmplitude = abs(dftSamples[index])
#     #print('signamAmplitude',signalAmplitude)
#     if ( 0 != signalAmplitude):
#         powerSpectrumArray[index] = (math.log(signalAmplitude**2))
#
#
# #freqArray = arange()
# plt.figure(figsize = (8, 6))
# # plt.xlim(- frequencyOfLaserField * 4, frequencyOfLaserField * 4)
# #plt.ylim(0,10)
# # plt.xlim( -150000, +150000 )
# plt.xlabel('Freq (Hz)')
# plt.ylabel('DFT Amplitude |X(freq)|')
# #lt.scatter(freqArray,powerSpectrumArray)
# plt.plot(freqArray,powerSpectrumArray)
# plt.show()
# #print(len(np.arange(1,1001,1)))
# #print((1e9))
# #