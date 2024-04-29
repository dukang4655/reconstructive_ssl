import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from geo_data import *



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 *14 *14, 128)
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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 *14 *14, 128)
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


def train_pretext_model(X1_train, X2_train, epochs=15, batch_size=64, device='cpu'):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in DataLoader(TensorDataset(X1_train, X2_train), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def pretext_feature_extraction(model, input_data, label, device='cpu'):
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
    label = torch.tensor(label, dtype=torch.long).to(device)

    
    X_features = []
    y_shape = []

    model.eval()
    with torch.no_grad():
        for images, true_shapes in DataLoader(TensorDataset(input_data, label), batch_size=1, shuffle=True):
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
    criterion = nn.CrossEntropyLoss()
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

def evaluate_model(model, X_test, Y_test, device='cpu'):
    model.eval()
    correct = 0
    X_test, Y_test = X_test, Y_test
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == Y_test).sum().item()
    accuracy = correct / Y_test.size(0)
    return accuracy


def run_exp(num_training_samples = 20000, num_labled_samples=50, num_test_samples =1000, noise_level=32, pair = 0, noise_type = 0):

    # SSL pre-training
    

    X_data ,y_angle ,y_shape = generate_data(n = num_training_samples ,dim=64 ,noise_level=noise_level, pair = pair, noise_type =noise_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_angle_tensor = torch.tensor(y_angle, dtype=torch.long).to(device)
    

    pretext_model = train_pretext_model(X_tensor, y_angle_tensor, epochs=15, batch_size=64, device=device)


    ################################################################################################
    

    # SSL downstream task


    X_down ,y_angle_down ,y_shape_down = generate_labeled_data(n = num_labled_samples ,dim=64 ,pair= pair, noise_level=noise_level,noise_type =noise_type)


    X_down_tensor = torch.tensor(X_down, dtype=torch.float32).to(device)
    y_down_shape_tensor = torch.tensor(y_shape_down, dtype=torch.long).to(device)

    X_features_down, y_shape_down= pretext_feature_extraction(pretext_model, X_down_tensor, y_down_shape_tensor,device=device)


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


    X_test ,y_angle_test ,y_shape_test = generate_data(n = num_test_samples ,dim=64 ,noise_level=noise_level, noise_type =noise_type, pair = pair)


    Xt_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    yt_tensor = torch.tensor(y_shape_test, dtype=torch.long).to(device)

    Xt_features_tensor, yt_shape_tensor =  pretext_feature_extraction(pretext_model, Xt_tensor, yt_tensor, device=device)



    yt_shape_onehot = encoder.fit_transform(yt_shape_tensor.reshape(-1, 1))

    yt_pred_onehot = rr.predict(Xt_features_tensor)

    # 2. Convert these predictions back from one-hot encoding to label encoding
    yt_pred = yt_pred_onehot.argmax(axis=1)
    yt_shape_true_test = yt_shape_onehot.argmax(axis=1)

    # 3. Compare the predicted labels with the true labels and compute the accuracy
    accuracy = accuracy_score(yt_shape_true_test, yt_pred)

    ################################################################################################
    accuracy_sl = evaluate_model(model_sl, Xt_tensor, yt_tensor, device='gpu')
    


    return accuracy, accuracy_sl

