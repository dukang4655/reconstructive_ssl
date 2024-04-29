import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from mnist_data import *

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
        # Forward pass till the fc1 layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        return x


class ShapeClassifier(nn.Module):
    def __init__(self, input_features, num_classes=2):  # circle or rectangle
        super(ShapeClassifier, self).__init__()
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class SimpleCNN_SL(nn.Module):
    def __init__(self):
        super(SimpleCNN_SL, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_pretext_model(X1_train, X2_train, epochs=15, batch_size=64, device='cpu'):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
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

def pretext_feature_extraction(model, test_loader, device='cpu'):

    X_features = []
    y_shape = []

    model.eval()
    with torch.no_grad():
        for images, true_shapes in test_loader:
            # Extract features
            features = model.extract_features(images.to(device))
            # Get shape predictions
            X_features.append(features)
            y_shape.append(true_shapes)
    X_features = torch.cat(X_features, 0)
    y_shape = torch.cat(y_shape, 0)

    return X_features.cpu(), y_shape.cpu()



def sl_training_model(X_train, Y_train, epochs=15, batch_size=32, device='cpu'):
    model = SimpleCNN_SL().to(device)
    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model,test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            outputs = model(X_test.to(device))
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == Y_test.to(device)).sum().item()
            total+= Y_test.size(0)
    accuracy = correct / total
    return accuracy



def run_exp(num_training_samples = 20000, num_labled_samples=50, noise_level=32, noise_type = 0):

    # SSL pre-training
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    X_data, y_angle, _ = generate_rotated_mnist_samples(dataset, num_training_samples, dim=28,
                                                        noise_max=noise_level, noise_type=0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_angle_tensor = torch.tensor(y_angle, dtype=torch.long).to(device)

    pretext_model = train_pretext_model(X_tensor, y_angle_tensor, epochs=15, batch_size=64, device=device)


    ################################################################################################

    # SSL downstream task


    X_down,y_shape_down   = generate_labled_rotated_mnist_samples(dataset, num_labled_samples,dim=28,noise_max=noise_level, noise_type=noise_type)


    X_down_tensor = torch.tensor(X_down, dtype=torch.float32).to(device)
    y_down_shape_tensor = torch.tensor(y_shape_down, dtype=torch.long).to(device)
    
    dataset_down = TensorDataset(X_down_tensor, y_down_shape_tensor)

    train_down_loader = DataLoader(dataset_down, batch_size=1, shuffle=True)

    X_features_down, y_shape_down= pretext_feature_extraction(pretext_model, train_down_loader,device=device)


    encoder = OneHotEncoder(sparse=False)
    y_shape_onehot = encoder.fit_transform(y_shape_down.reshape(-1, 1))

    # Set alpha to a suitable value; smaller values imply less regularization
    rr = RidgeCV(alphas=np.logspace(-3, 2, 200))

    rr.fit(X_features_down, y_shape_onehot)

    # Predict
    y_pred_onehot = rr.predict(X_features_down)

    #  Convert the predictions back from one-hot encoding to label encoding
    y_pred = y_pred_onehot.argmax(axis=1)

    # Compare the predicted labels with the true labels and compute the accuracy
    accuracy = accuracy_score(y_shape_down, y_pred)


    ################################################################################################
    # SL training

    model_sl = sl_training_model(X_down_tensor, y_down_shape_tensor, epochs=15, batch_size=32, device=device)

    ################################################################################################

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    Xt_features_tensor, yt_shape_tensor =  pretext_feature_extraction(pretext_model, test_loader, device=device)



    yt_shape_onehot = encoder.fit_transform(yt_shape_tensor.reshape(-1, 1))

    yt_pred_onehot = rr.predict(Xt_features_tensor)

    # 2. Convert these predictions back from one-hot encoding to label encoding
    yt_pred = yt_pred_onehot.argmax(axis=1)
    yt_shape_true_test = yt_shape_onehot.argmax(axis=1)

    # 3. Compare the predicted labels with the true labels and compute the accuracy
    accuracy = accuracy_score(yt_shape_true_test, yt_pred)

    ################################################################################################
    accuracy_sl = evaluate_model(model_sl, test_loader, device=device)

    return accuracy, accuracy_sl