import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time
from utils import setup_logging, TrainingTimer, calculate_classification_metrics, save_metrics, plot_confusion_matrix, plot_training_time, generate_report

# Paths
MODEL_STORAGE_PATH = os.getenv('MODEL_STORAGE_PATH', 'models')
DATASET_BASE_PATH = os.getenv('DATASET_BASE_PATH', 'datasets')

def load_network_dataset(logger):
    """Load Network dataset"""
    try:
        df = pd.read_csv(f'{DATASET_BASE_PATH}/network/network_anomalies.csv')
        if 'is_anomalous' not in df.columns:
            raise ValueError("Network dataset missing 'is_anomalous' column")
        df['is_malicious'] = df['is_anomalous']
        df = df.drop('is_anomalous', axis=1)
        logger.info("Network dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading Network dataset: {e}")
        raise

def engineer_features(df):
    """Feature engineering for Network"""
    # Assuming columns like packet_count, bytes_transferred, etc.
    # Add more features if needed
    df['log_packet_count'] = np.log1p(df['packet_count'])
    df['log_bytes'] = np.log1p(df['bytes_transferred'])
    df['packet_to_byte_ratio'] = df['packet_count'] / (df['bytes_transferred'] + 1)
    df = df.drop(['source_ip', 'dest_ip'], axis=1, errors='ignore')  # Assuming IPs are not used directly
    return df

def train_network_model():
    """Train Network model with multiple algorithms and hyperparameter tuning"""
    logger = setup_logging('train_network')
    timer = TrainingTimer(logger)

    timer.start()
    logger.info("Starting Network model training")

    df = load_network_dataset(logger)
    df = engineer_features(df)
    logger.info("Features engineered")

    X = df.drop('is_malicious', axis=1)
    y = df['is_malicious']

    X = X.fillna(0)
    X.columns = X.columns.astype(str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Data scaled")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define models and parameters
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=50),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [50]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, n_estimators=50),
            'params': {
                'n_estimators': [50],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42, n_estimators=50),
            'params': {
                'n_estimators': [50],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'svc': {
            'model': SVC(random_state=42, probability=True, max_iter=50),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'max_iter': [50]
            }
        }
    }

    best_model = None
    best_accuracy = 0
    best_name = ''

    for name, config in models.items():
        logger.info(f"Tuning {name}")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"{name} best accuracy: {acc:.6f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = grid.best_estimator_
            best_name = name

    logger.info(f"Best model: {best_name} with accuracy {best_accuracy:.6f}")

    # Final evaluation on best model
    y_pred = best_model.predict(X_test)
    metrics = calculate_classification_metrics(y_test, y_pred, logger)
    metrics['training_duration'] = timer.elapsed()

    save_metrics(metrics, 'train_network')

    plot_confusion_matrix(np.array(metrics['confusion_matrix']), 'train_network')
    plot_training_time(timer.elapsed(), 'train_network')

    # Save model and scaler
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    model_path = f'{MODEL_STORAGE_PATH}/network_model.pkl'
    scaler_path = f'{MODEL_STORAGE_PATH}/scaler_network.pkl'
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Report
    end_time = time.time()
    start_time = timer.start_time
    graph_paths = [
        'training_artifacts/graphs/individual/train_network_confusion_matrix.png',
        'training_artifacts/graphs/individual/train_network_training_time.png'
    ]
    generate_report('train_network', best_name, ['network'], time.ctime(start_time), time.ctime(end_time), timer.elapsed(), metrics, model_path, graph_paths)

    logger.info("Network model trained and saved successfully")

if __name__ == "__main__":
    train_network_model()