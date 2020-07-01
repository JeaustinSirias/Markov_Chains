#MODULACION IQ TENEMOS 2 PORTADORAS ORTOGONALES SIN Y COSIN
#SI SE LE OTORGA UNA FREQ DE OPERACION A UNA SENAL, NO PUEDE INTERRUMPIRSE PORQUE ES DELITO


#VAMOS A IMPLEMENTAR UNA SENAL MONOPORTADORA
#OOK: ON/OF KEYING
#UN CERO NO ENVIAMOS NADA
#VIENE UN 1 ENVIAMOS UN SIN

import numpy as np
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt
from decimal import *

#******************************************************************
#1 - CREAR UN ESQUEMA DE MODULACION BPSK

#UN PERIODO DE TRANSMISION DEL SIMBOLO ES
#UN PERIODO COMPLETO DE LA PORTADORA
getcontext().prec = 10
N = 5 #numero de bits
X = stats.bernoulli(0.5) #VA BIN

#generando bits para transmision
bits = X.rvs(N) #generar datos aleatorios
#bits = np.array(bits)
print(bits)
freq = 1000 #esta es una freq en Hz de op
T = 1/Decimal(freq) 
T = float(T)
p = 50 #num de ptos. de muestreo por period
time_vector = np.linspace(0, T, p)
print(T)

sine = np.sin(2*np.pi*freq*time_vector)

#plt.plot(time_vector, sine)
#plt.show()

#FREQ DE MUESTREO
fs = p/T #[]
time = np.linspace(0, N*T, N*p) #timeline for whole signal
signal = np.zeros(time.shape) #here will be my modulated signal

#CREACION DE LA SENAL MODULADA OOK
print(time.shape)

for k, b in enumerate(bits):
	if b == 1:
		signal[k*p:(k+1)*p] = b*sine #esta es la modulacion
	elif b == 0:
		signal[k*p:(k+1)*p] = -1*sine

print(signal)
#VISUALIZACION DE LOS 1EROS BITS MODULADOS
pb = 10
plt.figure(0)
plt.plot(signal[0:pb*p], color = 'b')
plt.hlines(0,0,250)
#plt.savefig('mod.png')
#******************************************************************
#2- CALCULAR LA POTENCIA PROMEDIO DE LA SENAL DE SALIDA

P_INSTANTANEA =signal**2 #potencia instantanea
P_avg = np.trapz(P_INSTANTANEA, time) / N*T
print('La potencia media de la senal es:' + str(P_avg))

#******************************************************************
#3-SIMULAR UN CANAL RUIDOSO DEL TIPO AWGN de -2 dB hasta 3 dB

#generar una componente ruidosa:
#en el ruido gausseano la pot. del ruido es sigma^2
SNR = 3 #SNR(dB) =10log10(Ps/Pn    )


noise = np.random.normal(0, 0.1, signal.shape)  #libreria.modulo.funcion()

#se simula el canal

RX = signal + noise #senal con ruido a partir de la original

#ploteamos la salida de la senal ahora con ruido (RX)

pb = 10
plt.figure(1)
plt.plot(RX[0:pb*p], color = 'r')
plt.hlines(0,0,250)
#plt.savefig('mod.png')
plt.show()

#******************************************************************
#4 - GRAFICAR LA DENSIDAD EXPECTRAL DEL LA POTENCIA DE LA SENAL DE SALIDA


E = sum(sine**2) #energia de la onda no ruidosa
RXbits = np.zeros(bits.shape) #un array de ceros con el tamano de la dim de bits


for k, b in enumerate(bits):
	energy_per_period = np.sum(RX[k*p:(k+1)*p]*sine) #energia en el tiempo de emulacion completo
	if energy_per_period > E/3:
		RXbits[k] = 1
	else: 
		RXbits[k] = 0

error = np.sum(np.abs(bits - RXbits))



BER = error/N
print(bits)
print(RXbits)

