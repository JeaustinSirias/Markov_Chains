 
#TAREA 4; GRUPO 01
#MODELOS PROBABILISTICOS DE SENALES Y SISTEMAS
#ESTUDIANTE: JEAUSTIN SIRIAS CHACON, B66861
#I - 2020

##############################
###########PACKAGES###########
##############################
import csv
import numpy as np
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt
from scipy import signal
import scipy as spy

from decimal import *

#########################
#MAKING UP 'bits10k.csv'#
#########################

#WE IMPORT INPUT DATA FILE:
with open('bits10k.csv', 'r') as file:
    reader = csv.reader(file)		
    for row in reader:
        bits = list(reader)

#TURN THIS DATA FILE INTO A INTEGER NUMPY ARRAY
B=[]
for i in range(0,10000):
	mylist = int(bits[i][0])
	B.append(mylist)
	Bits = B
	Bits = np.array(Bits) #'Bits' is the resulting array


########################################################
#ACTIVITY 1: GETTING A BPSK MODULATION SCHEME FROM 'Bits'#
########################################################

getcontext().prec = 10 #float extention 0.0000000000
freq = 5000 # in herts [Hz]
T = float(1/Decimal(freq)) #period
pin = 50 #number of point per period
time_vector = np.linspace(0, T, pin)
sine = np.sin(2*np.pi*freq*time_vector) #porting signal

#SAMPLING

time = np.linspace(0, len(Bits)*T, len(Bits)*pin) #timeline for whole signal
signal = np.zeros(time.shape) #here will be my BPSK modulated signal

#A WAY TO CAT A WHOLE BPSK SIGNAL BY ITERATING 'sine'
for k, b in list(enumerate(Bits)):
	if b == 1:
		signal[k*pin:(k+1)*pin] = b*sine 
	elif b == 0:
		signal[k*pin:(k+1)*pin] = -1*sine

res = 5 #signal plot resolution
plt.figure(0)
plt.plot(signal[0:res*pin], color = 'r')
plt.title('Raw BPSK modulated signal')
plt.ylabel('Amplitude [V]')
plt.xlabel('Period 1 ms = 50 points')
plt.hlines(0,0,250)
plt.grid()
plt.savefig('Raw_BPSK_sgn.png')
#plt.show()

#############################################################
#ACTIVITY 2: GETTING THE MEAN POWER OF BPSK MODULATED SIGNAL#
#############################################################

instant_p = signal**2
avg_p = integrate.trapz(instant_p, time)/(len(Bits)*T)
print('The average power in Watt for BPSK is:' + str(avg_p))

##############################################
#ACTIVITY 3: SIMULATING AN AWGN NOISY CHANNEL#
##############################################

SNR = [-2, -1, 0, 1, 2, 3] #in dB

#we know that Pn is the power for each value of SRN vector, so:

Pn = [0.7765976643059456, 0.6168734517791419, 0.49, 0.3892208350148979, 
0.30916909879529464, 0.24558174447736342]

noise_array = []

#we're gonna get a noised signal array where each of its elements is an array 
#related to a SRN value
for i in Pn:
	noise = np.random.normal(0, (i**2), signal.shape)
	noisy_sgn = noise + signal
	noise_array.append(noisy_sgn)

#plot noised signal for each SRN value
plt.figure(1)
plt.plot(noise_array[0][0:res*pin], label = 'SRN = -2 dB')
plt.plot(noise_array[1][0:res*pin], label = 'SRN = -1 dB')
plt.plot(noise_array[2][0:res*pin], label = 'SRN = 0 dB')
plt.plot(noise_array[3][0:res*pin], label = 'SRN = 1 dB')
plt.plot(noise_array[4][0:res*pin], label = 'SRN = 2 dB')
plt.plot(noise_array[5][0:res*pin], label = 'SRN = 3 dB')
plt.title('BPSK modulated signal throughout a noisy channel')
plt.legend(framealpha=1, frameon=True);
plt.ylabel('Amplitude [V]')
plt.xlabel('Period 1 ms = 50 points')
plt.hlines(0,0,250)
plt.grid()
plt.savefig('noised_BPSK_sgn.png')


#############################################
#ACTIVITY 4: PLOTTING POWER SPECTRAL DENSITY#
#############################################

samplin_freq = pin/T #sampling frequency
#Before noisy channel:
fw, PSD = spy.signal.welch(signal, samplin_freq, nperseg=1024)

plt.figure(2)
plt.semilogy(fw, PSD, color = 'g')
plt.title('Power spectral density Before noisy channel, using Welch method')
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.ylabel('Power spectral density [V**2/Hz]')
plt.savefig('WelchRaw_BPSK_sgn.png')

#after noisy channel

A = []
B = []
for i in np.arange(0,6,1):
	fw, PSD = spy.signal.welch(noise_array[i], samplin_freq, nperseg=1024)
	A.append(fw)
	B.append(PSD)

plt.figure(3)
plt.semilogy(A[0], B[0], label = 'SRN = -2 dB')
plt.semilogy(A[1], B[1], label = 'SRN = -1 dB')
plt.semilogy(A[2], B[2], label = 'SRN = 0 dB')
plt.semilogy(A[3], B[3], label = 'SRN = 1 dB')
plt.semilogy(A[4], B[4], label = 'SRN = 2 dB')
plt.semilogy(A[5], B[5], label = 'SRN = 3 dB')
plt.title('Power spectral density after noisy channel, using Welch method')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectral density [V**2/Hz]')
plt.savefig('WelchNoised_BPSK_sgn.png')

################################################
#ACTIVITY 5: DEMODULATING & DECODING SIGNALS#
################################################

raw_sgn_energy = np.sum(sine**2) #energy in raw signal
recieved_bits = np.zeros(Bits.shape)

#decoding signal by detecting its energy
for k, b in list(enumerate(Bits)):
    Ep = np.sum(noise_array[0][k*pin:(k+1)*pin] * sine)
    if Ep > raw_sgn_energy:
        recieved_bits[k] = 1
    else:
        recieved_bits[k] = 0

recieved_bits = np.array(recieved_bits) 

relative_error = np.sum(np.abs(Bits - recieved_bits))
BER = relative_error/len(Bits)

print(BER)
BER_array = [0.2503, 0.2512, 0.2462, 0.2416, 0.2433, 0.2441 ]

plt.figure(4)
plt.plot(SNR, BER_array)
plt.title('BER probability curve for BPSK modulation')
plt.xlabel('SNR [dB]')
plt.ylabel('Bit Error Rate, BER')
plt.grid()
plt.show()
plt.savefig('bit_error_rate.png')

#plt.savefig('a.png')
