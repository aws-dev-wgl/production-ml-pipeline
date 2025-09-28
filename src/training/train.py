# project2-ml-cicd/src/training/train.py - Windows Compatible
import os
import json
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import joblib
from datetime import datetime, timezone
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Simple but production-ready ML training pipeline"""
    
    def __init__(self, experiment_name="ml-pipeline-demo"):
        self.experiment_name = experiment_name
        # Create output directory
        self.output_dir = "models"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set MLflow tracking (you'll set this up next)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
    
    def create_sample_data(self):
        """Create sample dataset for demo purposes"""
        logger.info("Creating sample dataset...")
        
        # Generate classification dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame for easier handling
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def preprocess_data(self, df):
        """Simple preprocessing pipeline"""
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for later use
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, hyperparameters=None):
        """Train the model"""
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        
        logger.info(f"Training model with params: {hyperparameters}")
        
        # Train model
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Model metrics: {metrics}")
        return metrics
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting ML training pipeline...")
        
        try:
            # Try to start MLflow run
            mlflow.start_run()
            mlflow_available = True
        except Exception:
            logger.warning("MLflow not available, continuing without tracking")
            mlflow_available = False
        
        try:
            # Step 1: Create/load data
            df = self.create_sample_data()
            logger.info(f"Dataset shape: {df.shape}")
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            
            # Step 3: Train model
            hyperparameters = {
                'n_estimators': int(os.getenv('N_ESTIMATORS', '100')),
                'max_depth': int(os.getenv('MAX_DEPTH', '10')),
                'min_samples_split': int(os.getenv('MIN_SAMPLES_SPLIT', '5')),
                'random_state': 42
            }
            
            model = self.train_model(X_train, y_train, hyperparameters)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Step 5: Log to MLflow (if available)
            if mlflow_available:
                mlflow.log_params(hyperparameters)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
            
            # Step 6: Check quality gate
            min_f1_score = float(os.getenv('MIN_F1_SCORE', '0.8'))
            meets_quality = metrics['f1_score'] >= min_f1_score
            
            # Step 7: Save model locally
            model_path = os.path.join(self.output_dir, 'model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Step 8: Save results (convert numpy types to Python types for JSON)
            result = {
                'success': True,
                'metrics': {k: float(v) for k, v in metrics.items()},  # Convert numpy types
                'meets_quality_gate': bool(meets_quality),  # Convert to Python bool
                'model_path': model_path,
                'scaler_path': os.path.join(self.output_dir, 'scaler.pkl'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Save result for CI/CD
            result_path = os.path.join(self.output_dir, 'training_result.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Training completed! F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"Quality gate: {'PASSED' if meets_quality else 'FAILED'}")
            logger.info(f"Results saved to {result_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            result = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            result_path = os.path.join(self.output_dir, 'training_result.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            raise e
        finally:
            if mlflow_available:
                mlflow.end_run()

def main():
    """Main entry point"""
    pipeline = MLTrainingPipeline()
    result = pipeline.run_training()
    
    # Exit with appropriate code for CI/CD
    if result['success'] and result['meets_quality_gate']:
        logger.info("✅ Training successful - ready for deployment!")
        exit(0)
    else:
        logger.error("❌ Training failed or quality gate not met")
        exit(1)

if __name__ == "__main__":
    main()