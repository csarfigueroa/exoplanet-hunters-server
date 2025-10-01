import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class NeuralNetworkService:
    def __init__(self):
        self.model = None
        self.trained = False

    def create_sample_data(self, n_samples=1000):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, epochs=50, learning_rate=0.001):
        X_train, X_test, y_train, y_test = self.create_sample_data()

        self.model = SimpleNeuralNetwork()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        train_losses = []

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        self.trained = True

        return {
            "message": "Model trained successfully",
            "epochs": epochs,
            "final_loss": train_losses[-1],
            "test_accuracy": accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }

    def evaluate_model(self, input_data=None):
        if not self.trained or self.model is None:
            return {"error": "Model not trained yet. Please train the model first."}

        if input_data is None:
            _, X_test, _, y_test = self.create_sample_data()
            input_data = X_test[:5]
            actual_labels = y_test[:5]
        else:
            actual_labels = None

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

        results = []
        for i, (pred, prob) in enumerate(zip(predicted, probabilities)):
            result = {
                "sample_index": i,
                "predicted_class": pred.item(),
                "confidence": prob[pred].item(),
                "probabilities": prob.tolist()
            }
            if actual_labels is not None:
                result["actual_class"] = actual_labels[i]
                result["correct"] = pred.item() == actual_labels[i]
            results.append(result)

        return {
            "message": "Model evaluation completed",
            "predictions": results,
            "model_status": "trained" if self.trained else "not_trained"
        }