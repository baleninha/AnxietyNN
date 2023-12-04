import pandas as pd
import numpy as np
from biosppy.signals import ecg, resp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import xgboost as xgb
from scipy.signal import butter, filtfilt
import nolds



# Function to process the ECG recordings and extract information (features) from data
def extract_ecg_features(file_path, sampling_rate):
    ecg_data = pd.read_csv(file_path, header=None)
    ecg_signal = ecg_data[0].values
    normalized_ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    out = ecg.ecg(signal=normalized_ecg_signal, sampling_rate=sampling_rate, show=False)
    
    # Extract R-R intervals
    rpeaks = out['rpeaks']
    rr_intervals = np.diff(rpeaks) / sampling_rate 
    
    # Time-domain feature: RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

    # Frequency-domain features: LF and HF components
    f, Pxx = signal.welch(rr_intervals, fs=1/(rr_intervals.mean()), nperseg=len(rr_intervals))
    lf = np.trapz(Pxx[(f >= 0.04) & (f <= 0.15)])
    hf = np.trapz(Pxx[(f >= 0.15) & (f <= 0.4)])

    # Non-linear feature: Sample Entropy
    sampen = nolds.sampen(rr_intervals)

    return [np.mean(rr_intervals), np.std(rr_intervals), rmssd, lf, hf, sampen]  # Feature vector





def extract_ecg_res_features(file_path, ecg_sampling_rate=500, res_sampling_rate=500):
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
    
    return features



def process_dataset():   # Function to load each ECG data file and extract features
    sampling_rate = 500  # 500Hz
    features = []
    # grab all 19 files, named A01.csv to A19.csv
    for i in range(1, 20): 
        file_path = f'./csv/A{i:02d}.csv'
        features.append(extract_ecg_res_features(file_path, sampling_rate)) #create features vector
    
    return features



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






features = process_dataset()

# anxiety_scores = [20,11,0,9,13,25,10,2,7,2,14,12,1,7,23,6,7,4,7] #original scores
# 0 - 'low', 1 - 'medium', 2 - 'high'
#0-9: 0
#10-19: 1
#20+: 2
anxiety_scores = [2,1,0,0,1,2,1,0,0,0,1,1,0,0,2,0,0,0,0]

anxiety_scores = [1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0]
#0-11: 0
#12+: 1


# Preparing the dataset
X = np.array(features)  # Feature matrix
y = np.array(anxiety_scores)  # Target vector

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Training
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predicting and Evaluating the Model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(xgb_model, X_scaled, y, cv=3, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", scores)
print("Average Cross-Validation Accuracy:", np.mean(scores))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Plotting function for classification results
def plot_classification_results(y_test, y_pred):
    # Number of test samples
    num_samples = len(y_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_samples), y_test, color='blue', label='True Values')
    plt.scatter(range(num_samples), y_pred, color='red', label='Predicted Values')

    # Connecting lines between true and predicted values
    for i in range(num_samples):
        plt.plot([i, i], [y_test[i], y_pred[i]], color='gray', linestyle='--')

    plt.title("True vs Predicted Anxiety Levels")
    plt.xlabel('Sample Index')
    plt.ylabel('Anxiety Level')
    plt.xticks(range(num_samples))  # Dynamic x-ticks based on the number of samples
    plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting the classification results
plot_classification_results(y_test, y_pred)
