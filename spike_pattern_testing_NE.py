#this one is right code use this to make dataset again
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt


# Define sampling rate and time interval
fs = 1000  # Hz
t = np.arange(0, 0.2, 1/fs)  # 100 ms


# Load data from files
#data90=pd.read_csv('FPV_result.csv')
data90=pd.read_csv('All_data.csv')


#1-201 soft, 801-1001 hard
fsr_data = data90['F-volt'].values[1601:1801]#[1601:1801]#[801:1601]#[0:801]
vib_data = data90['V-volt'].values[1601:1801]#[801:1601]#[802:1002]
softpot_data =data90['P-volt'].values[1601:1801]#[1:801]#[802:1002]


# Define filter parameters
lowcut = 5  # Hz
highcut = 200  # Hz
order = 2


# Define threshold for spike detection
threshold = 0.5  # V


# Define carrier frequency and modulation index for FM encoding of vibration data
fm_carrier_freq = 100  # Hz
fm_modulation_index = 10


# Define carrier frequency and modulation index for AM encoding of softpot data
am_carrier_freq = 50  # Hz
am_modulation_index = 0.5


# Butterworth bandpass filter
def butter_bandpass_filter(data90, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data90)
    return y


# Generate receptor potentials and spikes for FSR data
# Separate FSR data into FA1 and SA1 channels based on their temporal encoding
fsr_filtered = butter_bandpass_filter(fsr_data, lowcut, highcut, fs, order)
fsr_receptor_potential = fsr_filtered - np.mean(fsr_filtered)
fa1_spikes = np.where(fsr_receptor_potential > threshold)[0] / fs
sa1_spikes = np.where(fsr_receptor_potential < -threshold)[0] / fs

#fa1_spikes = np.where(fsr_receptor_potential > threshold)[0] / fs
#sa1_spikes = np.where(fsr_receptor_potential < -threshold)[0] / fs

###############################################
# Generate FM encoded spikes for vibration sensor data
vib_filtered = butter_bandpass_filter(vib_data, lowcut, highcut, fs, order)
vib_carrier = np.sin(2*np.pi*fm_carrier_freq*t)
vib_modulation = fm_modulation_index * vib_filtered
vib_fm = vib_carrier * np.sin(2*np.pi*(fm_carrier_freq+vib_modulation)*t)
vib_spikes = np.where(vib_fm > threshold)[0] / fs
#print('vib_spikes=',vib_spikes)

# Generate AM encoded spikes for softpot membrane data
softpot_filtered = butter_bandpass_filter(softpot_data, lowcut, highcut, fs, order)
softpot_receptor_potential = softpot_filtered - np.mean(softpot_filtered)
sa2_spikes = np.where(softpot_receptor_potential < -threshold)[0] / fs


# Plot data and spikes
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axs[0].plot(t, fsr_data, 'k')
#axs[0].plot(t, fsr_receptor_potential, c='r')
axs[0].set_ylabel('FSR (V)')
axs[0].set_xlim(0, 0.2)

#axs[1].plot(t, vib_fm, c='r')
axs[1].plot(t, vib_data, 'k')
#axs[1].plot(t, vib_filtered, c='r')
axs[1].set_ylabel('Vibration (V)')

axs[2].plot(t, softpot_data, 'k')
#axs[2].plot(t, softpot_filtered , c='r')
axs[2].set_ylabel('Softpot(P)')
axs[2].set_xlim(0, 0.2)
axs[2].set_xlabel('time')
all_spikes = [fa1_spikes, sa1_spikes, vib_spikes, sa2_spikes]
#all_spikes = [sa2_spikes]
# Plot raster plot of all spike times
fig, ax = plt.subplots(figsize=(15, 5))
colors = ['r', 'g', 'b', 'm']
for i, spikes in enumerate(all_spikes):
    ax.eventplot(spikes, color=colors[i], lineoffsets=i+1, linelengths=0.8)
ax.set_xlim(0, 0.1)
ax.set_xlabel('Time (s)')
#ax.set_ylabel('Sensor')
ax.set_yticks([1, 2, 3, 4])#
ax.set_yticklabels(['FA1', 'SA1', 'Vibration (FA2)','Softpot(SA2)'])#'FA1', 'SA1', 'Vibration (FA2)', 'Softpot(SA2)'
ax.set_title('Raster spike plot PLA')
plt.show()
