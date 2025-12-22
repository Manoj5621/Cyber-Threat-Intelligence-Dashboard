import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def load_hash_dataset(logger):
    """Load Hash dataset"""
    try:
        df = pd.read_csv(f'{DATASET_BASE_PATH}/hash/malware_hashes.csv')
        if 'is_malicious' not in df.columns:
            raise ValueError("Hash dataset missing 'is_malicious' column")
        logger.info("Hash dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading Hash dataset: {e}")
        raise

def engineer_features(df):
    """Feature engineering for Hash"""
    df['hash_length'] = df['hash'].apply(len)
    le = LabelEncoder()
    df['hash_type_encoded'] = le.fit_transform(df['hash_type'])
    df['starts_with_letter'] = df['hash'].apply(lambda x: 1 if x[0].isalpha() else 0)
    df['has_upper'] = df['hash'].apply(lambda x: 1 if any(c.isupper() for c in x) else 0)
    df['has_lower'] = df['hash'].apply(lambda x: 1 if any(c.islower() for c in x) else 0)
    df = df.drop(['hash', 'hash_type', 'risk_score', 'source'], axis=1, errors='ignore')
    return df

def train_hash_model():
    """Train Hash model with multiple algorithms and hyperparameter tuning"""
    logger = setup_logging('train_hash')
    timer = TrainingTimer(logger)

    timer.start()
    logger.info("Starting Hash model training")

    df = load_hash_dataset(logger)
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

    save_metrics(metrics, 'train_hash')

    plot_confusion_matrix(np.array(metrics['confusion_matrix']), 'train_hash')
    plot_training_time(timer.elapsed(), 'train_hash')

    # Save model and scaler
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    model_path = f'{MODEL_STORAGE_PATH}/hash_model.pkl'
    scaler_path = f'{MODEL_STORAGE_PATH}/scaler_hash.pkl'
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Report
    end_time = time.time()
    start_time = timer.start_time
    graph_paths = [
        'training_artifacts/graphs/individual/train_hash_confusion_matrix.png',
        'training_artifacts/graphs/individual/train_hash_training_time.png'
    ]
    generate_report('train_hash', best_name, ['hash'], time.ctime(start_time), time.ctime(end_time), timer.elapsed(), metrics, model_path, graph_paths)

    logger.info("Hash model trained and saved successfully")

if __name__ == "__main__":
    train_hash_model()