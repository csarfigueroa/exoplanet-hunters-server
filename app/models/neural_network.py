import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.services.data_processor import ExoplanetDataProcessor

class ExoplanetClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(ExoplanetClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer4 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout(x)
        x = self.layer4(x)
        return x

class ExoplanetNeuralNetworkService:
    def __init__(self):
        self.model = None
        self.trained = False
        self.data_processor = ExoplanetDataProcessor()
        self.training_accuracy = 0.0
        self.test_accuracy = 0.0
        self.input_size = 0
        self.scaler_fitted = False

    def train_model(self, epochs=100, learning_rate=0.001, batch_size=64):
        try:
            # Load and preprocess data
            X, y = self.data_processor.load_data()
            X_train, X_test, y_train, y_test = self.data_processor.preprocess_data(X, y)

            self.input_size = X_train.shape[1]

            # Store the scaler state after training preprocessing
            self.scaler_fitted = True

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test.values)

            # Initialize model
            self.model = ExoplanetClassifier(input_size=self.input_size)

            # Handle class imbalance with weighted loss
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts])

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            train_losses = []
            train_accuracies = []

            # Training loop
            for epoch in range(epochs):
                self.model.train()

                # Mini-batch training
                total_loss = 0
                correct_train = 0
                total_train = 0

                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += batch_y.size(0)
                    correct_train += (predicted == batch_y).sum().item()

                epoch_loss = total_loss / (len(X_train_tensor) // batch_size + 1)
                epoch_train_acc = correct_train / total_train

                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_train_acc)

                scheduler.step(epoch_loss)

            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                # Training accuracy
                train_outputs = self.model(X_train_tensor)
                _, train_predicted = torch.max(train_outputs.data, 1)
                self.training_accuracy = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)

                # Test accuracy
                test_outputs = self.model(X_test_tensor)
                _, test_predicted = torch.max(test_outputs.data, 1)
                self.test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

            self.trained = True

            # Calculate class-specific metrics
            exoplanet_correct = ((test_predicted == 1) & (y_test_tensor == 1)).sum().item()
            exoplanet_total = (y_test_tensor == 1).sum().item()
            exoplanet_precision = exoplanet_correct / max(1, (test_predicted == 1).sum().item())
            exoplanet_recall = exoplanet_correct / max(1, exoplanet_total)

            return {
                "message": "Exoplanet classification model trained successfully",
                "objective": "Binary classification: Exoplanet (1) vs Non-Exoplanet (0)",
                "epochs": epochs,
                "final_loss": train_losses[-1],
                "training_accuracy_percentage": round(self.training_accuracy * 100, 2),
                "test_accuracy_percentage": round(self.test_accuracy * 100, 2),
                "exoplanet_precision_percentage": round(exoplanet_precision * 100, 2),
                "exoplanet_recall_percentage": round(exoplanet_recall * 100, 2),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_used": self.data_processor.feature_columns,
                "data_info": self.data_processor.get_data_info()
            }

        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

    def evaluate_model(self, input_data=None):
        if not self.trained or self.model is None:
            return {"error": "Model not trained yet. Please train the model first."}

        try:
            if input_data is None:
                # Use test data for evaluation
                X, y = self.data_processor.load_data()
                _, X_test, _, y_test = self.data_processor.preprocess_data(X, y)
                input_data = X_test[:10]  # Get first 10 test samples
                actual_labels = y_test.values[:10]
            else:
                actual_labels = None

            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

            results = []
            correct_predictions = 0

            for i, (pred, prob) in enumerate(zip(predicted, probabilities)):
                exoplanet_confidence = prob[1].item() * 100  # Confidence for exoplanet class
                non_exoplanet_confidence = prob[0].item() * 100  # Confidence for non-exoplanet class

                is_exoplanet = pred.item() == 1
                classification_status = "EXOPLANET" if is_exoplanet else "NOT_EXOPLANET"

                # Determine confidence level
                max_confidence = max(exoplanet_confidence, non_exoplanet_confidence)
                if max_confidence >= 90:
                    confidence_level = "VERY_HIGH"
                elif max_confidence >= 75:
                    confidence_level = "HIGH"
                elif max_confidence >= 60:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"

                result = {
                    "object_id": f"sample_{i + 1}",
                    "is_exoplanet": is_exoplanet,
                    "classification": classification_status,
                    "confidence_level": confidence_level,
                    "accuracy_percentage": round(max_confidence, 2),
                    "exoplanet_probability_percentage": round(exoplanet_confidence, 2),
                    "non_exoplanet_probability_percentage": round(non_exoplanet_confidence, 2),
                    "prediction_summary": {
                        "result": classification_status,
                        "confidence": confidence_level,
                        "accuracy": f"{round(max_confidence, 2)}%"
                    }
                }

                if actual_labels is not None:
                    actual_is_exoplanet = actual_labels[i] == 1
                    actual_classification = "EXOPLANET" if actual_is_exoplanet else "NOT_EXOPLANET"
                    is_correct = pred.item() == actual_labels[i]

                    result["ground_truth"] = actual_classification
                    result["prediction_correct"] = bool(is_correct)
                    result["match_status"] = "CORRECT" if is_correct else "INCORRECT"

                    if is_correct:
                        correct_predictions += 1

                results.append(result)

            response = {
                "message": "Exoplanet classification evaluation completed",
                "model_accuracy_percentage": round(self.test_accuracy * 100, 2),
                "predictions": results
            }

            if actual_labels is not None:
                sample_accuracy = (correct_predictions / len(actual_labels)) * 100
                response["sample_accuracy_percentage"] = round(float(sample_accuracy), 2)
                response["correct_predictions"] = int(correct_predictions)
                response["total_samples"] = int(len(actual_labels))

            return response

        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

    def get_model_info(self):
        """Get information about the trained model"""
        if not self.trained:
            return {"error": "Model not trained yet"}

        return {
            "model_trained": self.trained,
            "training_accuracy_percentage": round(self.training_accuracy * 100, 2),
            "test_accuracy_percentage": round(self.test_accuracy * 100, 2),
            "input_features": self.input_size,
            "objective": "Binary classification: Confirmed Exoplanet vs Non-Confirmed",
            "data_source": "TESS Objects of Interest (TOI) catalog"
        }