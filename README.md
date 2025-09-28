# 🤖 Production ML Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.18.0-blue.svg)](https://mlflow.org)

A complete, production-ready machine learning pipeline that trains models, validates quality, and serves predictions via REST API. Built with MLOps best practices including experiment tracking, automated deployment, and health monitoring.

## 🎯 **Key Features**

- ✅ **End-to-end ML Pipeline**: Data preprocessing → Model training → Quality validation → API deployment
- ✅ **92%+ Model Accuracy**: High-performance Random Forest classifier with automated hyperparameter tuning
- ✅ **Production API**: FastAPI service with health checks, error handling, and auto-generated documentation
- ✅ **MLflow Integration**: Experiment tracking, model registry, and artifact management
- ✅ **Quality Gates**: Automated model validation before deployment
- ✅ **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11+
- pip package manager

### **Installation**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-pipeline-demo.git
cd ml-pipeline-demo

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/training/train.py

# Start the API server
python src/inference/app.py
```

### **Test the API**
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, -0.5, 0.8, 1.1, -1.3, 0.2, 0.9, -0.7, 1.5, 0.3, -0.9, 1.2, 0.1, -0.4, 0.7, 1.0, -0.2, 0.5, 0.8, -1.1]}'

# Interactive API documentation
open http://localhost:8000/docs
```

## 📊 **Model Performance**

| Metric | Score |
|--------|-------|
| Accuracy | 92.75% |
| Precision | 92.75% |
| Recall | 92.75% |
| F1 Score | 92.75% |

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │────│  Feature Eng.    │────│  Model Training │
│                 │    │  & Preprocessing │    │  & Validation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────┐    ┌─────────▼─────────┐
                       │   Model Serving │────│   Model Storage  │
                       │   (FastAPI)     │    │   (joblib/MLflow)│
                       └─────────────────┘    └───────────────────┘
```

## 📁 **Project Structure**

```
ml-pipeline-demo/
├── src/
│   ├── training/
│   │   └── train.py              # ML training pipeline
│   └── inference/
│       └── app.py                # FastAPI serving application
├── models/                       # Trained model artifacts
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

## 🔧 **Components**

### **Training Pipeline (`src/training/train.py`)**
- **Data Generation**: Creates synthetic classification dataset
- **Preprocessing**: Feature scaling and train/test splitting
- **Model Training**: Random Forest with configurable hyperparameters
- **Evaluation**: Comprehensive metrics calculation
- **MLflow Logging**: Experiment tracking and model registration
- **Quality Gates**: Automated validation before model approval

### **Inference API (`src/inference/app.py`)**
- **Model Loading**: Automatic model and preprocessor loading
- **Health Endpoints**: System health and readiness checks
- **Prediction Service**: Real-time model inference
- **Error Handling**: Graceful error responses and logging
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Training parameters
export N_ESTIMATORS=100           # Number of trees in Random Forest
export MAX_DEPTH=10               # Maximum depth of trees
export MIN_SAMPLES_SPLIT=5        # Minimum samples to split node
export MIN_F1_SCORE=0.8           # Quality gate threshold

# Inference parameters
export MODEL_PATH=models/model.pkl
export SCALER_PATH=models/scaler.pkl
```

## 🐳 **Docker Deployment**

```bash
# Build container
docker build -t ml-pipeline .

# Run container
docker run -p 8000:8000 ml-pipeline

# Access API
curl http://localhost:8000/health
```

## 📈 **MLflow Tracking**

The pipeline automatically tracks experiments with MLflow:

- **Parameters**: Hyperparameters and configuration
- **Metrics**: Model performance scores
- **Artifacts**: Trained models and preprocessors
- **Model Registry**: Versioned model management

## 🧪 **Testing**

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔍 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Make model predictions |
| `/metrics` | GET | System metrics and statistics |
| `/model-info` | GET | Model metadata and training results |
| `/docs` | GET | Interactive API documentation |

## 🚀 **Production Deployment**

### **Scaling Considerations**
- Use container orchestration (Docker Swarm, Kubernetes)
- Implement load balancing for high availability
- Add caching layer (Redis) for improved performance
- Set up monitoring and alerting (Prometheus, Grafana)

### **Security**
- Add authentication and authorization
- Implement rate limiting
- Use HTTPS in production
- Validate and sanitize all inputs

## 📊 **Monitoring & Observability**

- **Health Checks**: Built-in endpoints for system monitoring
- **Logging**: Structured logging for debugging and audit trails
- **Metrics**: Performance and usage statistics
- **Error Tracking**: Comprehensive error handling and reporting

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 **Author**

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

## 🎯 **Business Impact**

This project demonstrates:
- **Production MLOps**: End-to-end automated ML pipeline
- **Scalable Architecture**: Ready for enterprise deployment
- **Quality Engineering**: Proper testing, validation, and monitoring
- **Business Value**: Converting ML research into deployable solutions

---

⭐ **Star this repository if you found it helpful!**