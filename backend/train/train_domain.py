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

def load_domain_dataset(logger):
    """Load Domain dataset"""
    try:
        df = pd.read_csv(f'{DATASET_BASE_PATH}/domain/malicious_domains.csv')
        if 'is_malicious' not in df.columns:
            raise ValueError("Domain dataset missing 'is_malicious' column")
        logger.info("Domain dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading Domain dataset: {e}")
        raise

def engineer_features(df):
    """Feature engineering for Domain"""
    df['domain_length'] = df['domain'].apply(len)
    df['num_dots'] = df['domain'].apply(lambda x: x.count('.'))
    df['num_hyphens'] = df['domain'].apply(lambda x: x.count('-'))
    df['has_digits'] = df['domain'].apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
    df['tld'] = df['domain'].apply(lambda x: x.split('.')[-1] if '.' in x else '')
    df['tld_length'] = df['tld'].apply(len)
    df['subdomain_count'] = df['domain'].apply(lambda x: len(x.split('.')) - 1)
    df = df.drop(['domain', 'tld', 'risk_score', 'source'], axis=1, errors='ignore')
    return df

def train_domain_model():
    """Train Domain model with multiple algorithms and hyperparameter tuning"""
    logger = setup_logging('train_domain')
    timer = TrainingTimer(logger)

    timer.start()
    logger.info("Starting Domain model training")

    df = load_domain_dataset(logger)
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

    save_metrics(metrics, 'train_domain')

    plot_confusion_matrix(np.array(metrics['confusion_matrix']), 'train_domain')
    plot_training_time(timer.elapsed(), 'train_domain')

    # Save model and scaler
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    model_path = f'{MODEL_STORAGE_PATH}/domain_model.pkl'
    scaler_path = f'{MODEL_STORAGE_PATH}/scaler_domain.pkl'
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Report
    end_time = time.time()
    start_time = timer.start_time
    graph_paths = [
        'training_artifacts/graphs/individual/train_domain_confusion_matrix.png',
        'training_artifacts/graphs/individual/train_domain_training_time.png'
    ]
    generate_report('train_domain', best_name, ['domain'], time.ctime(start_time), time.ctime(end_time), timer.elapsed(), metrics, model_path, graph_paths)

    logger.info("Domain model trained and saved successfully")

if __name__ == "__main__":
    train_domain_model()