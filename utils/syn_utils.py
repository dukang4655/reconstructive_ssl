import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from syn_data import generate_syn_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


class PretextModel(nn.Module):
    def __init__(self, d1, d2):
        super(PretextModel, self).__init__()
        self.fc1 = nn.Linear(d1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, d2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DirectModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DirectModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_pretext_model(d1, d2, X1_train, X2_train, epochs=10, batch_size=32, device='cpu'):
    model = PretextModel(d1, d2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in DataLoader(TensorDataset(X1_train, X2_train), batch_size=batch_size, shuffle=True):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model

def model_predict(model, input_data, device='cpu', return_numpy=True):
    model.eval()
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(input_data)
    return predictions.cpu().numpy() if return_numpy else predictions

def sl_training_model(input_dim, output_dim, X_train, Y_train, epochs=10, batch_size=32, device='cpu'):
    model = DirectModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X_test, Y_test, device='cpu'):
    model.eval()
    correct = 0
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == Y_test).sum().item()
    accuracy = correct / Y_test.size(0)
    return accuracy


def run_exp(n_samples_pretext=10000, s=5, d1=10, d2=20, p=3, n_samples_downstream=300, n_samples_test=1000, epochs=10, reproduce =0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Generation

    X1, X2, Y_one_hot, _ = generate_syn_data(n=n_samples_pretext, s=s, d1=d1, d2=d2, p=p, reproduce= reproduce)

    # Split the data for the pretext task
    X1_train, X1_val, X2_train, X2_val = train_test_split(X1, X2, test_size=0.2, random_state=66)

    # Convert numpy arrays to PyTorch tensors
    X1_train_torch = torch.Tensor(X1_train)
    X2_train_torch = torch.Tensor(X2_train)

    model_pretext = train_pretext_model(d1=d1, d2=d2, X1_train=X1_train_torch, X2_train=X2_train_torch, epochs=10,
                                        batch_size=32, device=device)

    X1_downstream, X2_downstream, Y_downstream_one_hot, _ = generate_syn_data(n=n_samples_downstream, s=s, d1=d1, d2=d2,
                                                                          p=p, reproduce = reproduce)

    f_X1_downstream = model_predict(model_pretext, X1_downstream, device)

    # Train linear regression model
    model_downstream = LinearRegression().fit(f_X1_downstream, Y_downstream_one_hot)

    # Convert to PyTorch tensors
    X1_downstream_torch = torch.tensor(X1_downstream, dtype=torch.float32)
    X2_downstream_torch = torch.tensor(X2_downstream, dtype=torch.float32)
    Y_downstream_one_hot_torch = torch.tensor(np.argmax(Y_downstream_one_hot, axis=1),
                                              dtype=torch.long)  # for CrossEntropyLoss

    # Combine X1 and X2 for the second model
    X_downstream_combined = torch.cat([X1_downstream_torch, X2_downstream_torch], dim=1)

    # Supervise learning that direct predict Y with X1

    model_direct = sl_training_model(input_dim=d1, output_dim=p, X_train=X1_downstream_torch,
                                     Y_train=Y_downstream_one_hot_torch, epochs=10, batch_size=32, device=device)

    # Supervise learning that direct predict Y with X1 and X2

    model_direct2 = sl_training_model(input_dim=d1 + d2, output_dim=p, X_train=X_downstream_combined,
                                      Y_train=Y_downstream_one_hot_torch, epochs=10, batch_size=32, device=device)

    # generate test data

    X1_test, X2_test, Y_test_one_hot, Y_test = generate_syn_data(n=n_samples_test, s=s, d1=d1, d2=d2, p=p, reproduce = reproduce)

    X1_test_torch = torch.tensor(X1_test, dtype=torch.float32)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.long)  # Assuming Y_test are integer labels

    # If you have a combined test dataset (e.g., for model_direct2)
    X_test_combined_torch = torch.tensor(np.concatenate([X1_test, X2_test], axis=1), dtype=torch.float32)

    f_X1_test = model_predict(model_pretext, X1_test, device)

    # Make predictions

    Y_pred = model_downstream.predict(f_X1_test)

    Y_true_labels = np.argmax(Y_test_one_hot, axis=1)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    accuracy = accuracy_score(Y_true_labels, Y_pred_labels)
    print('Accuracy:', accuracy)

    # make predicitons with model_direct

    accuracy_direct = evaluate_model(model_direct, X1_test_torch, Y_test_torch, device)

    # matke predicitons with model_direct2

    accuracy_direct2 = evaluate_model(model_direct2, X_test_combined_torch, Y_test_torch, device)

    return accuracy, accuracy_direct, accuracy_direct2