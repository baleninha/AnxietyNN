import pandas as pd
import matplotlib.pyplot as plt
from biosppy.signals import ecg, resp
from scipy.signal import butter, filtfilt
import numpy as np
import nolds
from scipy import signal



"""
Provide tools to visualize the processing of the ECG and RES signals.
Created on Dec. 2 2023
"""

FILE_PATH = './csv/A06.csv' # select path to desired file
FILE_DATA = pd.read_csv(FILE_PATH, header=None)

ECG_RAW, RES_RAW = FILE_DATA [0], FILE_DATA [1] #ECG in 1st column, RES in 2nd column

VIEW = 2 # Which signal to visualize: 0-ECG  1-RES  2-both

def plot_raw_vs_filtered(outputs=VIEW, ecg_length=2500, res_length = 50000):
    """
    Args:
    - length: How many datapoints to display. default is 10k values.
    - outputs: how many and which signals to display. Default set to VIEW value.
    """
    ecg_raw = ECG_RAW[:ecg_length]
    ecg_raw = (ecg_raw - np.mean(ecg_raw)) / np.std(ecg_raw) # zero-mean normalization
    processed_ecg = ecg.ecg(signal=ecg_raw, sampling_rate=500, show=False)



    res_raw = RES_RAW[:res_length]
    res_raw = (res_raw - np.mean(res_raw)) / np.std(res_raw)

    # Option 1- Butterworth filter
    fs = 500  # sampling rate
    lowcut = 0.07
    highcut = 1.3
    processed_res = butter_bandpass_filter(res_raw, lowcut, highcut, fs, order=2)

    # Option 2- Biosppy filtering
    #processed_res = resp.resp(signal=res_raw, sampling_rate=500, show=False)


    if outputs == 0:
        plt.figure(figsize=(12, 4))
        plt.plot(ecg_raw, label='RAW ECG')
        plt.plot(processed_ecg['filtered'], label='filtered ECG')
        plt.title('Unfiltered vs Filtered ECG Signal')
        plt.legend()
        plt.show()

    elif outputs == 1:
        plt.figure(figsize=(12, 4))
        plt.plot(res_raw, label='RAW RES')
        #plt.plot(processed_res['filtered'], label='filtered RES') #use if option 2
        plt.plot(processed_res, label='filtered RES')
        plt.title('Unfiltered vs Filtered RES Signal')
        plt.legend()
        plt.show()

    else:
        plt.figure(figsize=(12, 4))
        plt.plot(ecg_raw, label='RAW ECG')
        plt.plot(processed_ecg['filtered'], label='filtered ECG')
        plt.title('Unfiltered vs Filtered ECG Signal')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(res_raw, label='RAW RES')
        #plt.plot(processed_res['filtered'], label='filtered RES') #use if option 2
        plt.plot(processed_res, label='filtered RES')
        plt.title('Unfiltered vs Filtered RES Signal')
        plt.legend()
        plt.show()



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def extract_ecg_res_features(file_path=FILE_PATH, ecg_sampling_rate=500, res_sampling_rate=500):
    data = pd.read_csv(file_path, header=None)
    ecg_signal = data[0].values
    res_signal = data[1].values

    # Normalize ECG signal
    normalized_ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # ECG processing
    ecg_out = ecg.ecg(signal=normalized_ecg_signal, sampling_rate=ecg_sampling_rate, show=False)
    rpeaks = ecg_out['rpeaks']
    rr_intervals = np.diff(rpeaks) / ecg_sampling_rate
    
    # Time-domain ECG feature: RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

    # Frequency-domain ECG features: LF and HF components
    f, Pxx = signal.welch(rr_intervals, fs=1/(rr_intervals.mean()), nperseg=len(rr_intervals))
    lf = np.trapz(Pxx[(f >= 0.04) & (f <= 0.15)])
    hf = np.trapz(Pxx[(f >= 0.15) & (f <= 0.4)])

    # Non-linear ECG feature: Sample Entropy
    sampen = nolds.sampen(rr_intervals)

    # Normalize RES signal
    normalized_res_signal = (res_signal - np.mean(res_signal)) / np.std(res_signal)
    
    
    fs = 500  # sampling rate
    lowcut = 0.07
    highcut = 1.3
    #normalized_res_signal = butter_bandpass_filter(normalized_res_signal, lowcut, highcut, fs, order=2)

    #the two lines below have reduced the algorithm's accuracy, likely due to overprocessing the signal. (Excessive smoothing)
    #normalized_res_signal = resp.resp(signal=normalized_res_signal, sampling_rate=500, show=False)
    #normalized_res_signal=normalized_res_signal['filtered']

    # Extract features from RES signal
    # Example: Respiratory Rate
    peaks, _ = signal.find_peaks(normalized_res_signal, height=0)
    res_rate = len(peaks) / (len(normalized_res_signal) / res_sampling_rate)

    # Combine features from both ECG and RES
    features = [np.mean(rr_intervals), np.std(rr_intervals), rmssd, lf, hf, sampen, res_rate]
    
    return features, rpeaks



def show_ecg_data_analysis(file_data=ECG_RAW):
    """
    Provide a visualization of main signal information
    Args:
    - file_data: raw ECG data
    """
    
    ecg_data = file_data#.values[:2500] # reduce number of samples in order to better visualize the data
    proccesed_ecg = ecg.ecg(signal=ecg_data, sampling_rate=500, show=False)

    
    # Plot the filtered ECG signal and detected R-peaks
    plt.figure(figsize=(12, 4))
    plt.plot(proccesed_ecg['filtered'], label='Filtered ECG')
    plt.plot(proccesed_ecg['rpeaks'], proccesed_ecg['filtered'][proccesed_ecg['rpeaks']], 'ro', label='R-peaks')
    plt.title('Filtered ECG Signal with R-peaks')
    plt.legend()
    plt.show()

    # Plot the average heartbeat template
    plt.figure(figsize=(12, 4))
    plt.plot(proccesed_ecg['templates_ts'], proccesed_ecg['templates'].mean(axis=0))
    plt.title('Average Heartbeat Template')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot heart rate over time
    plt.figure(figsize=(12, 4))
    plt.plot(proccesed_ecg['heart_rate_ts'], proccesed_ecg['heart_rate'])
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (beats/min)')
    plt.show()



def plot_ecg_features(rr_intervals, rmssd, lf, hf, sampen):
    """
    Plot ECG extracted features: RR intervals, RMSSD, LF, HF, and Sample Entropy.
    Args:
    - rr_intervals: array of RR interval data
    - rmssd: calculated RMSSD value
    - lf: calculated LF power value
    - hf: calculated HF power value
    - sampen: calculated Sample Entropy value
    """


    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle('Extracted ECG features')
    # Plot the RR intervals histogram
    axs[0, 0].hist(rr_intervals, bins=30)
    axs[0, 0].set_title('RR Intervals Histogram')

    # Plot the RMSSD
    axs[0, 1].bar(['RMSSD'], [rmssd])
    axs[0, 1].set_title('RMSSD Value')
    axs[0, 1].text(0, rmssd, f'{rmssd:.4f}', ha='center', va='bottom')

    # Plot the LF and HF power in a bar chart
    axs[1, 0].bar(['LF', 'HF'], [lf, hf])
    axs[1, 0].set_title('LF and HF Power')

    # Plot the Sample Entropy
    axs[1, 1].bar(['Sample Entropy'], [sampen])
    axs[1, 1].set_title('Sample Entropy')
    axs[1, 1].text(0, sampen, f'{sampen:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



def plot_res_features():
    pass




if __name__ == "__main__":

    sampling_rate = 500 
    plot_raw_vs_filtered()

    features, rpeaks = extract_ecg_res_features()

    
    mean_rr, std_rr, rmssd, lf, hf, sampen, res_rate = features
    rr_intervals = np.diff(rpeaks) / sampling_rate  

    # Plot ECG features
    plot_ecg_features(rr_intervals, rmssd, lf, hf, sampen)
    
    show_ecg_data_analysis()  
