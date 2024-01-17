from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from biosppy.signals import ecg, resp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, welch
import nolds
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from time import perf_counter





start = perf_counter()
show_graphs = True # Display graphical information


def extract_ecg_features(file_path, ecg_sampling_rate=250):

    data = pd.read_csv(file_path, header=None)
    ecg_signal = data[1].values

    # Normalize ECG 
    normalized_ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Signal processing
    ecg_out = ecg.ecg(signal=normalized_ecg_signal, sampling_rate=ecg_sampling_rate, show=False)
    rpeaks = ecg_out['rpeaks']
    rr_intervals = np.diff(rpeaks) / ecg_sampling_rate
    
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) # RMSSD

    # Frequency domain analysis using Welch's method
    f, Pxx = welch(rr_intervals, fs=1/(rr_intervals.mean()), nperseg=len(rr_intervals))
    vlf = np.trapz(Pxx[(f >= 0.003) & (f < 0.04)]) # very low frequency
    lf = np.trapz(Pxx[(f >= 0.04) & (f <= 0.15)]) # low frequency
    hf = np.trapz(Pxx[(f >= 0.15) & (f <= 0.4)]) # high frequency
    
    sampen = nolds.sampen(rr_intervals) # Sample Entropy

    features = [np.mean(rr_intervals), np.std(rr_intervals), rmssd, lf, hf, vlf, sampen]
    
    #features = [lf, hf, vlf]

    
    return features



def process_dataset():
    sampling_rate = 250  # 250Hz
    features = []
    anxiety_labels = []  # 0 - 'low', 1 - 'medium/high'

    def process_group(base_path, anxiety_label, directories):
        for directory_name in directories:
            dir_path = os.path.join(base_path, directory_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        features.append(extract_ecg_features(file_path, sampling_rate))
                        anxiety_labels.append(anxiety_label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    tasks_directories = {
        'bug_box_task': ['avoidance_response', 'confrontation_response', 'escape_response', 'safety_behavior_response'],
        'speaking_task': ['confrontation_response', 'safety_behavior_response']
    }

    anxiety_levels = [('high_anxiety_group', 1), ('low_anxiety_group', 0)]
    base_path_template = r'C:\Users\baleninha\Desktop\TFG\AnxietyPhasesDataset\data\electrocardiogram_data\{}\{}'

    for task, directories in tasks_directories.items():
        for anxiety_level, label in anxiety_levels:
            base_path = base_path_template.format(task, anxiety_level)
            process_group(base_path, label, directories)

    return features, anxiety_labels



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


features, anxiety_labels = process_dataset()

# prepare the dataset
X = np.array(features)  # feature matrix
y = np.array(anxiety_labels)  # target vector

# Data Scaling
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)


# splitting the into training / testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# define each model and its respective pipeline
models_pipelines = {
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier())
    ]),
        'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss',objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5))
    ]),
    'KNN': Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier(n_estimators=100))
    ]),
        'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier())
    ]),
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300))
    ])
}

# train each model using its pipeline
for model_name, pipeline in models_pipelines.items():
    pipeline.fit(X_train, y_train)
    print(f"► {model_name} training completed.")


stop = perf_counter()

elapsed_time = stop - start
print(f"\n Time elapsed: {elapsed_time} (s)")

# Evaluate each model
for model_name, pipeline in models_pipelines.items():
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred) # Ragne: -1 to 1, when 0 equals chance accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print(f"\n{model_name} - Accuracy: {accuracy}")
    print(f"{model_name} - Cohen's Kappa: {kappa}")
    print(f"{model_name} - Precision: {precision}")
    print(f"{model_name} - Recall: {recall}")
    print(f"{model_name} - F1: {f1}")



    #print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_pctg = cm / np.sum(cm) * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_pctg, annot=True, fmt=".2f", cmap="Blues", xticklabels=['LowAnxiety', 'HighAnxiety'], yticklabels=['LowAnxiety', 'HighAnxiety'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.savefig(os.path.join('./reports/', f'{model_name}_confusion_matrix.png'))

    if not show_graphs:
        plt.close()

    plt.show()



    # Save the figure
    

# Cross-validation
for model_name, pipeline in models_pipelines.items():
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"\n{model_name} - Cross-Validation Accuracy Scores: {scores}")
    print(f"{model_name} - Average Cross-Validation Accuracy: {np.mean(scores)}")


def plot_roc():
    # Initialize plot
    plt.figure(figsize=(10, 8))

    # For storing AUC scores for display
    auc_scores = {}

    # Evaluate each model and plot ROC
    for model_name, pipeline in models_pipelines.items():
        # Predict probabilities for the test data
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

    # Plot Base Rate ROC
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Chance')

    # Adding labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join('./reports/', 'roc_curve.png'))

    if not show_graphs:
        plt.close()

    # Show plot
    plt.show()



    # Print AUC scores
    for model, score in auc_scores.items():
        print(f"{model}: AUC = {score:.2f}")



def create_metrics_table(models_pipelines, X_train, y_train, X_test, y_test):
    metrics_dict = { 
        'Classifier': ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
    }

    # Evaluate each model and store the metrics
    for model_name, pipeline in models_pipelines.items():
        #pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics_dict[model_name] = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            cohen_kappa_score(y_test, y_pred)
        ]

    # Convert dictionary to DataFrame for easy file writing
    metrics_df = pd.DataFrame(metrics_dict)

    # Write DataFrame to a text file
    with open('./reports/metrics.txt', 'w') as file:
        metrics_df.to_string(file, index=False, justify='left')

    print("Metrics table written to 'metrics.txt'.")



plot_roc()
create_metrics_table(models_pipelines, X_train, y_train, X_test, y_test)

