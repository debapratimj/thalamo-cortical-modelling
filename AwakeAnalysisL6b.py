# David Kaplan
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd



# Import file and list channels in order

# List of headers taken from the CSV file
col_list = ["timestamps","timesamples","pri_0","pri_1","pri_2","pri_3","pri_4","pri_5","pri_6","pri_7","pri_8",
                "pri_9","pri_10","pri_11","pri_12","pri_13","pri_14","pri_15","aux_1","aux_2","din_1","din_2"]
# Electrode labels in order from most superficial to deep
Probe_order = ["pri_0","pri_15","pri_1","pri_14","pri_4","pri_11","pri_3","pri_12","pri_6","pri_9","pri_7","pri_8","pri_5","pri_10","pri_2","pri_13"]
# Path name (broken into fragments to avoid error when a number or lower case letter follows a backslash), obviously modify this for your own path
Path1 = "/home/ananym/Documents/CCO/invivodataanalysis/"
Path2 = "sbpro_15-01-23"
Path3 = "1 s"
# Name of the filename
Filename = "ChR2DRD_awake_Mouse2_sbpro_5__uid1221-19-16-35"
# Glue together the path and filename in the next three lines
Path_File = os.path.join(Path1, Path2, Path3, Filename)
File_Extension = ".csv"
Filename_complete = Path_File + File_Extension
# Convert the .csv file to a dataframe
filepath_complete  = "ChR2DRD_awake_Mouse2_sbpro_5__uid1221-19-16-35_reduced.csv"
df = pd.read_csv(filepath_complete, usecols=col_list)

# Assign variables from the csv file
Time = df["timesamples"].to_numpy()[0:-1:50]
LightPulse = df["din_2"].to_numpy()[0:-1:50] # Digit output where 1s = pulses, 0s = gaps
LightPulse_ON = np.where(np.diff(LightPulse)>0.5)[0] # Pulse onset is where trace diff exceeds 0.5
LightPulse_ON = LightPulse_ON
#LightPulse_OFF = np.where(np.diff(LightPulse)<-0.5)[0]
LightPulse_OFF = np.add(LightPulse_ON,600) #int(np.mean(np.subtract(LightPulse_OFF,LightPulse_ON))))
dt = Time[2]-Time[1] # time step
srate = 1/dt # sample rate
nyq = 0.5 * srate # Nyquist frequency
V1 = df["pri_7"].to_numpy()[0:-1:50] # Example voltage trace



####################################################
###### Power spectrum function #####################
def Spectrum(V_win):
	LFP_win = V_win - np.mean(V_win)
	# Windowing if you want
	w = np.hanning(len(V_win))
	LFP_win = w * LFP_win
	# Calculate power spectrum for window
	Fs = srate
	N = len(LFP_win)
	xdft = np.fft.fft(LFP_win)
	xdft = xdft[0:int((N / 2) + 1)]
	psdx = (1 / (Fs * N)) * np.abs(xdft) ** 2
	freq = np.arange(0, (Fs / 2) + Fs / N, Fs / N)
	Pow = psdx
	#Pow = np.zeros((501, 1))
	#for j in range(0, 200):
	#	Pow[j] = psdx[j]
	return Pow, freq

n=1000

Pow, freq = Spectrum(V1[LightPulse_ON[0]+50:LightPulse_OFF[0]])
Pulse_freq = freq
Pulse_Power = np.zeros((len(LightPulse_ON),len(Pow)))
Pow, freq = Spectrum(V1[LightPulse_ON[0] - 1000:LightPulse_ON[0]])
PrePulse_freq = freq
PrePulse_Power = np.zeros((len(LightPulse_ON),len(Pow)))
Fig, axs = plt.subplots(1,3)
Gamma_Pow_sum=np.zeros((len(Probe_order),len(LightPulse_ON)))
for ii in range(len(Probe_order)):
	LFP = df[Probe_order[ii]].to_numpy()[0:-1:50]
	for i in range(len(LightPulse_ON)):
		print('Iteration ',i,' of ',len(LightPulse_ON))
		#Fig, axs = plt.subplots(3,1)
		#axs[0].plot(Time[LightPulse_ON[i]-n:LightPulse_OFF[i]+n],V1[LightPulse_ON[i]-n:LightPulse_OFF[i]+n])
		#axs[1].plot(Time[LightPulse_ON[i]-n:LightPulse_OFF[i]+n], LightPulse[LightPulse_ON[i]-n:LightPulse_OFF[i]+n])
		print('Light pulse duration = ',LightPulse_OFF[i]-LightPulse_ON[i])
		Pow, freq = Spectrum(LFP[LightPulse_ON[i]+50:LightPulse_OFF[i]])
		print('Freq size = ',len(freq))
		print('Power size = ', len(Pow), len(Pulse_Power[i, :]))
		Pulse_Power[i,:] = Pow

		#axs[2].plot(freq,Pow,label = 'Light')
		Pow, freq = Spectrum(LFP[LightPulse_ON[i] - 1000:LightPulse_ON[i]])
		PrePulse_Power[i,:] = Pow
		#axs[2].plot(freq,Pow,label = 'Pre')
		#axs[2].legend()
		#axs[2].set_yscale('log')
		#plt.show()
		#print('Gamma values: ', np.sum(Pulse_Power[np.where(np.logical_and(Pulse_freq > 100, Pulse_freq < 120))[0]]))
		#Gamma_Pow_sum[ii,i] = np.sum(Pulse_Power[np.where(np.logical_and(Pulse_freq > 100, Pulse_freq < 120))[0]])
	max_power = 200
	axs[0].plot(PrePulse_freq,np.mean(PrePulse_Power,axis=0),label = 'Pre-pulse',color='blue',alpha=0.001+(0.999*(ii/len(Probe_order))),linewidth=0.2)
	axs[1].plot(Pulse_freq,np.mean(Pulse_Power,axis=0),label = 'Pulse',color='red',alpha=0.001+(0.999*(ii/len(Probe_order))),linewidth=0.2)
	axs[2].plot(Pulse_freq[0:200],np.subtract(np.mean(Pulse_Power[:,0:200],axis=0),np.mean(PrePulse_Power[:,0:200],axis=0)))



	axs[0].set_title('Pre-pulse')
	axs[0].set_ylim([0, 200])
	axs[0].set_yscale('log')
	#axs[0].set_ylabel('Power')
	axs[0].set_xlim([0, 150])
	#axs[0].set_xscale('log')
	axs[1].set_title('Pulse')
	axs[1].set_ylim([0, 200])
	axs[1].set_yscale('log')
	axs[1].set_yticklabels([])
	axs[1].set_yticks([])
	axs[1].set_xlabel('Frequency (Hz)')
	axs[1].set_xlim([0, 150])
	#axs[1].set_xscale('log')That
	axs[2].set_title('Power difference')
	axs[2].set_ylim([-200, 50])
	#axs[2].set_yscale('log')
	axs[2].set_xlim([0, 150])
	#axs.yscale('log')
	#plt.legend()
plt.show()

# Generate 2 x 2 subplot
# 1.1, 1.2 Plot LFP response for each channel before pulse (1.1) and during pulse (1.2) color code based on depth
# 2.1, 2.2 Plot the power spectra color coded for corresponding time windows above

# Generate 1 x 2 plot showing averaged power spectra for pre-pulse and pulsed windows for all depths

