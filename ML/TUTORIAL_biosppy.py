import pandas as pd
import matplotlib.pyplot as plt
from biosppy.signals import ecg

# Step 1: Reading the ECG Data
# Load ECG data from a CSV file. Assuming ECG data is in the first column without a header.
ecg_data = pd.read_csv('./csv/A03.csv', header=None)
ecg_signal = ecg_data[0].values[:5000]  # Extracting the ECG values

# It is important to know the sampling rate for accurate analysis.
# Replace this with your ECG's actual sampling rate.
sampling_rate = 500  # 500 Hz

# Step 2: ECG Signal Processing with BioSPPy
# The `ecg.ecg()` function processes the ECG signal and extracts relevant features.
# `show=False` stops it from automatically showing a plot.
out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)

# The `out` object is a dictionary containing several important items:
# 'filtered' - the filtered ECG signal
# 'rpeaks' - indices of R-peaks detected in the ECG signal
# 'templates_ts' - a time axis for the extracted heartbeat templates
# 'templates' - the extracted heartbeat templates
# 'heart_rate_ts' - time values for heart rate
# 'heart_rate' - instantaneous heart rate values

# Step 3: Visualizing the Processed ECG Signal
# Plotting the filtered ECG signal and detected R-peaks
plt.figure(figsize=(12, 4))
plt.plot(out['filtered'], label='Filtered ECG')
plt.plot(out['rpeaks'], out['filtered'][out['rpeaks']], 'ro', label='R-peaks')
plt.title('Filtered ECG Signal with R-peaks')
plt.legend()
plt.show()

# Step 4: Visualizing Heartbeat Templates
# Plotting the average heartbeat template
plt.figure(figsize=(12, 4))
plt.plot(out['templates_ts'], out['templates'].mean(axis=0))
plt.title('Average Heartbeat Template')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

# Step 5: Heart Rate Over Time
# Plotting heart rate over time
plt.figure(figsize=(12, 4))
plt.plot(out['heart_rate_ts'], out['heart_rate'])
plt.title('Heart Rate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (beats/min)')
plt.show()


