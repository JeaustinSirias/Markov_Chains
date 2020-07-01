# UNA SOLUCION A LA TAREA 4 POR JEAUSTIN SIRIAS
===============================================

## Resumen:


## Indice de figuras

- [Esquema de modulacion BPSK](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/Raw_BPSK_sgn.png)
- [Esquema de modulacion BPSK con ruido agregado para valores distintos del SNR](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/noised_BPSK_sgn.png)
- [Densidad espectral de potencia por el metodo de Welch sin contaminacion por ruido](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/WelchRaw_BPSK_sgn.png)
- [Densidad espectral de potencia por el metodo de Welch con contaminacion por ruido](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/WelchNoised_BPSK_sgn.png)
- [Tada de error en los bits vs SNR](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/bit_error_rate.png).

Se muestra una solucion programada para la Tarea 4 del curso Modelos probabilisticos de senales y sistemas empleando las siguientes librerias en el IDE SublimeText3 como interprete de Python:

	import csv
	import numpy as np
	import scipy.stats as stats
	from scipy import integrate
	import matplotlib.pyplot as plt
	from scipy import signal
	import scipy as spy
	from decimal import *

### ACTIVIDAD 1: Construyendo un esquema de modulacion BPSK a partir de un frasco de bits

Primeramente se expuso las condiciones bajo las cuales la modulacion se requeria, entre las cuales se destaca una frecuencia operacion de 5000 Hz a partir de un frasco de 10 000 bits aleatorios. De este modo se ha implementado una solucion programada a saber:

	getcontext().prec = 10 #float extention 0.0000000000
	freq = 5000 # in herts [Hz]
	T = float(1/Decimal(freq)) #period
	pin = 50 #number of point per period
	time_vector = np.linspace(0, T, pin)
	sine = np.sin(2*np.pi*freq*time_vector) #porting signal

	#SAMPLING

	time = np.linspace(0, len(Bits)*T, len(Bits)*pin) #timeline for whole signal
	signal = np.zeros(time.shape) #here will be my BPSK modulated signal
GN
	#A WAY TO CAT A WHOLE BPSK SIGNAL BY ITERATING 'sine'
	for k, b in list(enumerate(Bits)):
		if b == 1:
			signal[k*pin:(k+1)*pin] = b*sine 
		elif b == 0:
			signal[k*pin:(k+1)*pin] = -1*sine

La modulacion BPSK obtenida puede observarse [aqui](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/Raw_BPSK_sgn.png).

### ACTIVIDAD 2: Calculando la potencia promedio de la señal modulada 

Empleando la metodologia vista en clase y en el Py8, entonces una solucion programada es:

	instant_p = signal**2
	avg_p = integrate.trapz(instant_p, time)/(len(Bits)*T)
	print('The average power in Watt for BPSK is:' + str(avg_p))

Al final, entonces se obtiene una potencia para la onda cruda de la actividad anterior de 0.49 W.


### ACTIVIDAD 3: Simulando un canal ruidoso de tipo AWGN para un SNR desde -2 dB hasta 3 dB


En este apartado se debe pretender que BPSK obtenida atraviesa por un canal que le interfiere con ruido. Para esto entonces se establece un vector para el SNR tal que:

	SNR = [-2, -1, 0, 1, 2, 3] #in dB
	Pn = []
	for i in SNR:
		pn = avg_p/(10**(i/10))
		Pn.append(pn)

Siendo el arreglo Pn la potencia del ruido asociado en Watt a cada valor del vector SNR. De este modo entonces se simula el canal ruidoso para cada valor de Pn:

	for j in [0, 1, 2, 3, 4, 5]:
	noise = np.random.normal(0, (Pn[j])**2, signal.shape) #where (Pn[j])**2 equals sigma.
	noisy_sgn = noise + signal

El resultado es una onda modulada por BPSK contaminada en funcion del SNR como se muestra en esta [figura](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/noised_BPSK_sgn.png)

### ACTIVIDAD 4: Graficando la densidad espectral de potencia de la señal con el método de Welch antes y después del canal ruidoso

Una forma de obtener el espectro de potencia antes de la señal ser interferida opr el canal ruidoso es:

	samplin_freq = pin/T #sampling frequency
	#Before noisy channel:
	fw, PSD = spy.signal.welch(signal, samplin_freq, nperseg=1024)

Estas lineas arrojan como resultado la presente [figura](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/WelchRaw_BPSK_sgn.png). Por el otro extremo suponiendo que la señal ya ha dejado el canal ruidoso, entonces, una forma programada para determinar el espectro de frecuencia para cada valor de SNR es:

	#after noisy channel
	A = []
	B = []
	for i in np.arange(0,6,1):
	fw, PSD = spy.signal.welch(noise_array[i], samplin_freq, nperseg=1024)
		A.append(fw) #this is an array which contanins all fw cases for SNR vaules
		B.append(PSD) #this is an array which contanins all PSD cases for SNR vaules

Al final el resultado es una unica [figura](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/WelchNoised_BPSK_sgn.png) que contiene todas las curvas de los espectros de frecuencia.


### ACTIVIDAD 5: Demodulando y decodificando la señal para elaborar un conteo de la tasa de error de bits (BER) para cada nivel SNR.

	raw_sgn_energy = np.sum(sine**2) #energy in raw signal
	recieved_bits = np.zeros(Bits.shape) #size dimention where HIGH bits (1) will be stored after demodulation
	for k, b in list(enumerate(Bits)): #decoding signal by detecting its energy
	    Ep = np.sum(noise_array[0][k*pin:(k+1)*pin] * sine)
	    if Ep > raw_sgn_energy:
		recieved_bits[k] = 1
	    else:			#bit selection criteria
		recieved_bits[k] = 0

	recieved_bits = np.array(recieved_bits) 
	relative_error = np.sum(np.abs(Bits - recieved_bits))
	BER = relative_error/len(Bits)
	print(BER)


Luego de probar valores del 0 al 5 en noise_array[i] se obtiene el siguiente vector de la taza de error de bits:
	
	BER_array = [0.2503, 0.2512, 0.2462, 0.2416, 0.2433, 0.2441 ]

### ACTIVIDAD 6: Graficar el BER vs el SNR

Implementando las siguientes lineas:

	plt.plot(SNR, BER_array)
	plt.title('BER probability curve for BPSK modulation')
	plt.xlabel('SNR [dB]')
	plt.ylabel('Bit Error Rate, BER')
	plt.grid()
	plt.show()

se obtiene el resultado que se muestra en esta [figura](https://github.com/JeaustinSirias/Tarea4_B66861/blob/master/bit_error_rate.png).





ia 
