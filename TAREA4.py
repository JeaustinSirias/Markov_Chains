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

print(signal)
res = 5 #signal plot resolution
plt.figure(0)
plt.plot(signal[0:res*pin], color = 'b')
plt.title('Raw BPSK modulated signal')
plt.ylabel('Amplitude [V]')
plt.xlabel('Period 1 ms = 50 points')
plt.hlines(0,0,250)
#plt.savefig()
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


Pn = []
for i in SNR:
	pn = avg_p/(10**(i/10))
	Pn.append(pn)
print(Pn)

BER_array = []
for j in [0, 1, 2, 3, 4, 5]:
	noise = np.random.normal(0, Pn[j], signal.shape)
	noisy_sgn = noise + signal
	plt.figure(1)
	plt.plot(noisy_sgn[0:res*pin], color = 'r')
	plt.title('BPSK modulated signal throughout a noisy channel')
	plt.ylabel('Amplitude [V]')
	plt.xlabel('Period 1 ms = 50 points')
	plt.hlines(0,0,250)
	#plt.show()

	#############################################
	#ACTIVITY 4: PLOTTING POWER SPECTRAL DENSITY#
	#############################################
	samplin_freq = pin/T
	#Before noisy channel:
	fw, PSD = spy.signal.welch(signal, samplin_freq, nperseg=1024)
	plt.figure(2)
	plt.semilogy(fw, PSD, color = 'g')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Power spectral density [V**2/Hz]')


	#after noisy channel
	fw, PSD = spy.signal.welch(noisy_sgn, samplin_freq, nperseg=1024)
	plt.figure(3)
	plt.semilogy(fw, PSD, color = 'g')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Power spectral density [V**2/Hz]')
	#plt.show()

	################################################
	#ACTIVITY 5: DEMODULATING & DECODING SIGNALS#
	################################################

	raw_sgn_energy = np.sum(sine**2) #energy in raw signal
	recieved_bits = np.zeros(Bits.shape)


	#decoding signal by detecting its energy
	for k, b in list(enumerate(Bits)):
	    Ep = np.sum(noisy_sgn[k*pin:(k+1)*pin] * sine)
	    if Ep > raw_sgn_energy/2:
	        recieved_bits[k] = 1
	    else:
	        recieved_bits[k] = 0

	recieved_bits = np.array(recieved_bits) 

	relative_error = np.sum(np.abs(Bits - recieved_bits))
	BER = relative_error/len(Bits)
	BER_array.append(BER)
	

print(BER_array)

plt.figure(5)
plt.plot(SNR, BER_array)
plt.title('BER probability curve for BPSK modulation')
plt.xlabel('SNR [dB]')
plt.ylabel('Bit Error Rate, BER')
plt.grid()
plt.savefig('a.png')