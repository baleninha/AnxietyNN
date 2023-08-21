import torch
import torch.nn.functional as F
import pandas as pd
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from time import perf_counter
import os
import json
import numpy as np
 

t_start = perf_counter()

FILE_PARTITIONS = 1000 #number of sets we're splitting our data into
ROWS_CUTOFF = 1230000 #where we trim the file so that we get an equal number of samples from each patient

# read fron JSON auxiliary file
data_filename = open("patientData.json")
datafile = json.load(data_filename)
csv_dir_path = datafile["data_folder_path"]



# Custom Dataset for loading and preprocessing the data
class BeckAnxietyDataset(Dataset):
    def __init__(self, csv_list, labels):
        """
        self.data = []
        for csv in csv_list:
            df = pd.read_csv(csv, usecols=[0], nrows=ROWS_CUTOFF)
            self.data.append(df.iloc[:, 0].values)
        self.labels = labels[:len(self.data)]
        #print('DATA:', self.data[:10])
        """
        self.input_layer_len = 1230
        self.data = []
        self.labels = labels
        #for csv in os.scandir(csv_dir_path):
        for csv in csv_list:

            df = pd.read_csv(csv, usecols=[0], nrows=ROWS_CUTOFF)
            

            split_data = [group.values for group in np.array_split(df, len(df) // 1230)]
            self.data.extend(split_data)
        #print('LEN DATA IS IS: ', len(self.data))


        #print("Num of arrays is, ", len(self.data))
        #print("len of array in array is, ", len(self.data[0][0]), " ", len(self.data[5][0]))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(self.data[idx], dtype=torch.float32).view(-1, 1230), torch.tensor(self.labels[idx], dtype=torch.float32)

# Define Neural Network Architecture for Regression
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()

        internal_neurons = 150

        self.fc1 = nn.Linear(input_size, internal_neurons)
        self.fc2 = nn.Linear(internal_neurons, internal_neurons)
        self.fc3 = nn.Linear(internal_neurons, internal_neurons)

        self.fc4 = nn.Linear(internal_neurons, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)


        return x

# Performance metric
def check_mae(loader, model):
    total_error = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).float()
            scores = model(x).squeeze(1)
            total_error += torch.abs(scores - y).sum().item()
            num_samples += y.size(0)
    model.train()
    return total_error / num_samples

# Paths to CSV and labels
csv_files = ["C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A01.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A02.csv",  "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A03.csv",  "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A04.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A05.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A06.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A07.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A08.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A09.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A10.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A11.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A13.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A14.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A15.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A16.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A18.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A19.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A20.csv", "C:\\Users\\baleninha\\Desktop\\TFG\\codeTests\\DataMat\\csv\\A21.csv"] # Paths to your 19 CSV files", ...] # Paths to your 19 CSV files
labels =  [20,11,0,9,13,25,10,2,7,2,14,12,1,7,23,6,7,4,7] # 19 Beck Anxiety Test scores
labels = [label for label in labels for _ in range(1000)]
print(len(labels))

# Splitting the dataset into training and testing
train_files = csv_files[:16]
train_labels = labels[:16000]
test_files = csv_files[16:]
test_labels = labels[16000:]

# Create datasets and loaders
train_dataset = BeckAnxietyDataset(train_files, train_labels)
test_dataset = BeckAnxietyDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Assuming each CSV has the same number of rows
input_size = len(pd.read_csv(csv_files[0], usecols=[0], nrows=1230000))
input_size = 1230

# Model, Loss, and Optimizer
model = RegressionNN(input_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# Training loop
num_epochs = 10
# Train Network
for epoch in range(num_epochs):
    for data, targets in train_loader:
        # Get data to cuda if possible
        #data = data.to(device=device).squeeze(dim=1)
        data = data.to(device=device).squeeze(dim=1)

        targets = targets.to(device=device)
        targets = targets.unsqueeze(1)


        # Forward
        #scores = model(data).squeeze(dim=1)  # Squeeze the scores tensor to match target shape
        scores = model(data)  # This will ensure the output shape is [1] if the batch size is 1

        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()


t_stop = perf_counter()


print("Data type: {} Neurons: {}, Epochs: {}, Training patients: {}, Internal layers: {}".format('ECG', 150, 10, 16, 2))

# Evaluate model on training data
print(f"Mean Absolute Error on training dataset: {check_mae(train_loader, model):.2f}")

# Evaluate model on test data
print(f"Mean Absolute Error on test dataset: {check_mae(test_loader, model):.2f}")

print("Time elapsed (seconds): ", t_stop - t_start )