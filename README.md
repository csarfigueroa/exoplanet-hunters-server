# 🪐 Exoplanet Hunter - Neural Network Classification System

A FastAPI-powered machine learning system that classifies celestial objects as exoplanets or non-exoplanets using neural networks and real astronomical data from NASA's TESS mission.

## 📋 Table of Contents

- [Overview](#overview)
- [How Neural Networks Work for Exoplanet Detection](#how-neural-networks-work-for-exoplanet-detection)
- [End-to-End Process](#end-to-end-process)
- [Data Science Concepts](#data-science-concepts)
- [Architecture & Source Code](#architecture--source-code)
- [API Endpoints](#api-endpoints)
- [Setup & Usage](#setup--usage)
- [Technical Details](#technical-details)

---

## 🌟 Overview

This system uses artificial intelligence to identify exoplanets by analyzing patterns in astronomical data. Think of it as training a computer to recognize the "fingerprint" characteristics that distinguish confirmed exoplanets from other celestial objects.

**Key Question**: Given the properties of a stellar system (star brightness, planet size, orbital period, etc.), can we predict if it contains a confirmed exoplanet?

---

## 🧠 How Neural Networks Work for Exoplanet Detection

### What is a Neural Network?

A neural network is like a sophisticated pattern recognition system inspired by how the human brain works. Instead of manually programming rules, we show the computer thousands of examples and let it learn the patterns.

```
INPUT FEATURES → NEURAL NETWORK → OUTPUT CLASSIFICATION
[Star data]    →    [Learning]   →    [Exoplanet: Yes/No]
```

### The Learning Process

1. **Feature Extraction**: We convert astronomical observations into numerical features
2. **Pattern Recognition**: The network learns which combinations of features indicate exoplanets
3. **Decision Making**: Given new data, it predicts the likelihood of an exoplanet

### Why Neural Networks for Exoplanets?

- **Complex Patterns**: Exoplanet detection involves subtle relationships between multiple variables
- **Non-Linear Relationships**: Traditional linear methods miss complex interactions
- **Large Datasets**: Neural networks excel with thousands of astronomical observations
- **Automation**: Can process new discoveries faster than human analysis

---

## 🔄 End-to-End Process

### Phase 1: Data Preparation
```
Raw TOI Data → Data Cleaning → Feature Selection → Normalization → Training/Test Split
```

1. **Data Loading**: Read NASA's TESS Objects of Interest (TOI) catalog
2. **Feature Selection**: Choose 15 key astronomical measurements
3. **Data Cleaning**: Handle missing values and outliers
4. **Normalization**: Scale all features to similar ranges (critical for neural networks)
5. **Label Creation**: Mark confirmed exoplanets (CP = Confirmed Planet) vs others

### Phase 2: Model Training
```
Training Data → Neural Network → Loss Calculation → Weight Updates → Trained Model
```

1. **Architecture Design**: Create multi-layer neural network
2. **Forward Pass**: Data flows through layers, making predictions
3. **Loss Calculation**: Compare predictions with known answers
4. **Backpropagation**: Adjust network weights to reduce errors
5. **Iteration**: Repeat thousands of times until accuracy stabilizes

### Phase 3: Evaluation & Prediction
```
New Data → Preprocessing → Trained Model → Probability Scores → Classification
```

1. **Data Preprocessing**: Apply same normalization as training
2. **Inference**: Run data through trained network
3. **Probability Calculation**: Get confidence scores for each class
4. **Classification**: Determine final prediction based on highest probability

---

## 📊 Data Science Concepts

### Features (Input Variables)
Our model uses 15 astronomical measurements:

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `ra`, `dec` | Sky coordinates | Location patterns |
| `st_pmra`, `st_pmdec` | Stellar motion | Kinematic properties |
| `pl_orbper` | Orbital period | Fundamental exoplanet characteristic |
| `pl_trandurh` | Transit duration | How long planet blocks star |
| `pl_trandep` | Transit depth | How much light is blocked |
| `pl_rade` | Planet radius | Physical size |
| `pl_insol` | Stellar irradiation | Energy received |
| `pl_eqt` | Equilibrium temperature | Habitability indicator |
| `st_tmag` | Star brightness | Observational quality |
| `st_dist` | Distance to star | Measurement reliability |
| `st_teff` | Star temperature | Stellar type |
| `st_logg` | Surface gravity | Stellar characteristics |
| `st_rad` | Star radius | System scale |

### Target Variable (Output)
- **Binary Classification**: `EXOPLANET` (1) vs `NOT_EXOPLANET` (0)
- **Based on**: TFOPWG disposition where `CP = Confirmed Planet`

### Model Architecture
```
Input Layer (15 features)
    ↓
Hidden Layer 1 (128 neurons) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 2 (64 neurons) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 3 (32 neurons) + ReLU + BatchNorm + Dropout
    ↓
Output Layer (2 neurons) → Softmax → Probabilities
```

### Key Techniques
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting
- **Class Weighting**: Handles imbalanced data (few confirmed exoplanets)
- **Learning Rate Scheduling**: Adaptive optimization

---

## 🏗️ Architecture & Source Code

### Project Structure
```
exoplanet-hunters-server/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── models/
│   │   └── neural_network.py   # Core ML model and training logic
│   ├── services/
│   │   └── data_processor.py   # Data loading and preprocessing
│   └── routers/
│       └── neural_network.py   # API endpoints and request handling
├── data/
│   └── TOI_*.csv              # NASA TESS Objects of Interest data
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Orchestration
└── requirements.txt           # Python dependencies
```

### Key Source Files

#### 1. `app/models/neural_network.py`
**Purpose**: Core machine learning implementation

**Key Classes**:
- `ExoplanetClassifier`: PyTorch neural network architecture
- `ExoplanetNeuralNetworkService`: Training and inference logic

**Key Methods**:
- `train_model()`: Handles full training pipeline
- `evaluate_model()`: Model evaluation and prediction
- `get_model_info()`: Model metadata and performance

```python
# Neural Network Architecture
class ExoplanetClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        # Multi-layer architecture with normalization and dropout
```

#### 2. `app/services/data_processor.py`
**Purpose**: Data pipeline and preprocessing

**Key Class**:
- `ExoplanetDataProcessor`: Handles all data operations

**Key Methods**:
- `load_data()`: Reads and cleans TOI catalog
- `preprocess_data()`: Feature scaling and train/test split
- `prepare_single_prediction()`: Processes individual predictions

```python
# Data Processing Pipeline
class ExoplanetDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()  # Feature normalization
        self.feature_columns = []       # Selected features
```

#### 3. `app/routers/neural_network.py`
**Purpose**: REST API endpoints

**Key Endpoints**:
- `POST /exoplanet/train`: Model training
- `POST /exoplanet/evaluate`: Model evaluation
- `POST /exoplanet/predict`: Single object classification
- `GET /exoplanet/status`: Model information
- `GET /exoplanet/data-info`: Dataset statistics

#### 4. `app/main.py`
**Purpose**: FastAPI application setup and configuration

---

## 🚀 API Endpoints

### Training
```bash
POST /exoplanet/train
{
  "epochs": 100,
  "learning_rate": 0.001,
  "batch_size": 64
}
```

### Evaluation
```bash
POST /exoplanet/evaluate
{}  # Uses test data automatically
```

### Prediction
```bash
POST /exoplanet/predict
{
  "stellar_data": {
    "ra": 120.8376880,
    "dec": -51.3456260,
    "st_pmra": -23.3660000,
    # ... 12 more features
  }
}
```

### Information
```bash
GET /exoplanet/status     # Model performance
GET /exoplanet/data-info  # Dataset statistics
```

---

## 🛠️ Setup & Usage

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### Quick Start
```bash
# Clone and start
git clone <repository>
cd exoplanet-hunters-server
docker-compose up --build

# Access API
curl http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Training Workflow
1. **Start the server**: `docker-compose up`
2. **Train the model**: `POST /exoplanet/train`
3. **Evaluate performance**: `POST /exoplanet/evaluate`
4. **Make predictions**: `POST /exoplanet/predict`

---

## 🔬 Technical Details

### Data Source
- **NASA Exoplanet Archive**: TESS Objects of Interest (TOI) catalog
- **Real astronomical data**: Measured by NASA's TESS space telescope
- **Updated regularly**: Latest observations from ongoing sky surveys

### Model Performance
- **Accuracy**: ~80-85% on test data
- **Class Imbalance**: ~8% confirmed exoplanets, 92% other objects
- **Metrics**: Precision, recall, and F1-score for both classes

### Scalability
- **Containerized**: Runs consistently across environments
- **Stateless API**: Horizontal scaling ready
- **Batch Processing**: Efficient for large datasets
- **Memory Efficient**: Optimized data pipeline

### Future Improvements
- **Feature Engineering**: Add derived astronomical features
- **Ensemble Methods**: Combine multiple models
- **Real-time Updates**: Automatic retraining with new data
- **Uncertainty Quantification**: Bayesian neural networks

---

## 📚 Learn More

### Concepts to Explore
- **Exoplanet Detection Methods**: Transit photometry, radial velocity
- **Machine Learning**: Supervised learning, classification
- **Neural Networks**: Deep learning, backpropagation
- **Astronomy**: TESS mission, Kepler's laws, stellar classification

### Related Technologies
- **PyTorch**: Neural network framework
- **FastAPI**: Modern Python web framework
- **Pandas**: Data manipulation
- **Docker**: Containerization
- **NASA APIs**: Astronomical data sources

---

## Relevant notes
Acknowledgment: The repository and project documentation include attribution lines such as
Co-Authored-By: Claude <noreply@anthropic.com>
to indicate AI assistance in code generation.


*This system demonstrates how artificial intelligence can assist in astronomical discovery, helping identify potentially habitable worlds beyond our solar system.*