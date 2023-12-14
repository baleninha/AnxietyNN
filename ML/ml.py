from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from biosppy.signals import ecg, resp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt
import nolds

def extract_ecg_res_features(file_path, ecg_sampling_rate=500, res_sampling_rate=500):
    data = pd.read_csv(file_path, header=None)
    ecg_signal = data[0].values
    res_signal = data[1].values

    # Normalize ECG 
    normalized_ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # signal processing
    ecg_out = ecg.ecg(signal=normalized_ecg_signal, sampling_rate=ecg_sampling_rate, show=False)
    rpeaks = ecg_out['rpeaks']
    rr_intervals = np.diff(rpeaks) / ecg_sampling_rate
    
    # RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

    # LF / HF 
    f, Pxx = signal.welch(rr_intervals, fs=1/(rr_intervals.mean()), nperseg=len(rr_intervals))
    lf = np.trapz(Pxx[(f >= 0.04) & (f <= 0.15)])
    hf = np.trapz(Pxx[(f >= 0.15) & (f <= 0.4)])

    # Sample Entropy
    sampen = nolds.sampen(rr_intervals)

    # Normalize RES 
    normalized_res_signal = (res_signal - np.mean(res_signal)) / np.std(res_signal)
    
    
    fs = 500  # sampling rate
    lowcut = 0.07
    highcut = 1.3
    #normalized_res_signal = butter_bandpass_filter(normalized_res_signal, lowcut, highcut, fs, order=2)

    #the two lines below have reduced the algorithm's accuracy, likely due to overprocessing the signal. (Excessive smoothing)
    #normalized_res_signal = resp.resp(signal=normalized_res_signal, sampling_rate=500, show=False)
    #normalized_res_signal=normalized_res_signal['filtered']

    # Extract features from RES 
    # Respiratory Rate
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
# 0 - 'low', 1 - 'medium/high'

anxiety_scores = [1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0]
anxiety_scores = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0]

#anxiety_scores = [20,11,0,9,13,25,10,2,7,2,14,12,1,7,23,6,7,4,7]

#0-11: 0
#12+: 1


# prepare the dataset
X = np.array(features)  # feature matrix
y = np.array(anxiety_scores)  # target vector

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# splitting the into training / testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=39)

# define each model and its respective pipeline
models_pipelines = {
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier())
    ]),
    'KNN': Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ])
}

# train each model using its pipeline
for model_name, pipeline in models_pipelines.items():
    pipeline.fit(X_train, y_train)
    print(f"{model_name} training completed.")

# Evaluate each model
for model_name, pipeline in models_pipelines.items():
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} - Accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Cross-validation
for model_name, pipeline in models_pipelines.items():
    scores = cross_val_score(pipeline, X_scaled, y, cv=3, scoring='accuracy')
    print(f"\n{model_name} - Cross-Validation Accuracy Scores: {scores}")
    print(f"{model_name} - Average Cross-Validation Accuracy: {np.mean(scores)}")