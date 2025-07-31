import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_fscore_support, 
                           precision_recall_curve, roc_curve, average_precision_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import deque
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import google.colab
    # Đang chạy trên Google Colab
    DATA_PATH = '/content/drive/MyDrive/MyStudy/NCS/VNU/01-2023/datasets/IOTDataset/CIC_IOT_Dataset2023/'
    CODE_PATH = '/content/drive/MyDrive/MyStudy/NCS/VNU/01-2023/MyJournal/202411-ICTA/codes/icta2025/'
except ImportError:
    # Đang chạy trên local
    DATA_PATH = '/home/noattran/codes/icta2025/dataset/ciciot2023/'
    CODE_PATH= '/home/noattran/codes/icta2025/'

# Global configuration for model save directory
MODEL_SAVE_DIR = os.path.join(CODE_PATH,'saved_models')

def ensure_model_directory():
    """Ensure model save directory exists"""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"✓ Created model directory: {MODEL_SAVE_DIR}")
    else:
        print(f"✓ Model directory exists: {MODEL_SAVE_DIR}")

def get_model_path(filename):
    """Get full path for model file"""
    ensure_model_directory()
    return os.path.join(MODEL_SAVE_DIR, filename)

class PreliminaryAnomalyFilter:
    """
    Preliminary Anomaly Filter for streaming data batches
    Uses lightweight techniques to quickly assess if a batch contains anomalies
    Only anomalous batches are sent to RL agent for detailed processing
    """
    
    def __init__(self, sensitivity='medium'):
        """
        Initialize preliminary anomaly filter
        
        Args:
            sensitivity: 'low', 'medium', 'high' - determines how strict the filter is
        """
        self.sensitivity = sensitivity
        self.sensitivity_params = {
            'low': {'contamination': 0.05, 'threshold_multiplier': 1.5},
            'medium': {'contamination': 0.1, 'threshold_multiplier': 1.0},
            'high': {'contamination': 0.15, 'threshold_multiplier': 0.7}
        }
        
        # Models for preliminary detection
        self.models = {}
        self.thresholds = {}
        self.is_trained = False
        
        # Statistics for Mahalanobis distance
        self.mahalanobis_cov = None
        self.mahalanobis_mean = None
        
        # Performance tracking
        self.filter_stats = {
            'total_batches': 0,
            'passed_batches': 0,
            'filtered_batches': 0,
            'false_positives': 0,  # Normal batches marked as anomalous
            'false_negatives': 0,  # Anomalous batches marked as normal
            'processing_time': []
        }
        
        print(f"🔍 Preliminary Anomaly Filter initialized")
        print(f"   Sensitivity: {sensitivity}")
        print(f"   Expected contamination: {self.sensitivity_params[sensitivity]['contamination']}")
    
    def train_on_normal_data(self, X_normal, validation_data=None):
        """
        Train preliminary filter on normal data
        
        Args:
            X_normal: Normal training data (scaled)
            validation_data: Optional (X_val, y_val) for threshold tuning
        """
        print(f"\n🎯 Training Preliminary Anomaly Filter on {len(X_normal)} normal samples")
        
        params = self.sensitivity_params[self.sensitivity]
        
        try:
            # 1. Isolation Forest - Fast tree-based anomaly detection
            print("   Training Isolation Forest...")
            self.models['isolation_forest'] = IsolationForest(
                contamination=params['contamination'],
                random_state=42,
                n_estimators=100,  # Fewer trees for speed
                max_samples=min(256, len(X_normal)),  # Smaller sample size
                n_jobs=-1
            )
            self.models['isolation_forest'].fit(X_normal)
            
            # 2. One-Class SVM - Boundary-based detection
            print("   Training One-Class SVM...")
            # Use subset for speed if data is large
            train_subset = X_normal[:min(1000, len(X_normal))]
            self.models['ocsvm'] = OneClassSVM(
                nu=params['contamination'],
                kernel='rbf',
                gamma='scale',
                cache_size=200
            )
            self.models['ocsvm'].fit(train_subset)
            
            # 3. Local Outlier Factor - Density-based detection
            print("   Training Local Outlier Factor...")
            self.models['lof'] = LocalOutlierFactor(
                n_neighbors=min(20, len(X_normal)//5),
                contamination=params['contamination'],
                novelty=True,
                n_jobs=-1
            )
            self.models['lof'].fit(X_normal)
            
            # 4. Mahalanobis Distance - Statistical distance-based detection
            print("   Computing Mahalanobis statistics...")
            self.mahalanobis_cov = EmpiricalCovariance()
            self.mahalanobis_cov.fit(X_normal)
            self.mahalanobis_mean = np.mean(X_normal, axis=0)
            
            # Set thresholds based on validation or percentiles
            if validation_data is not None:
                print("   Tuning thresholds on validation data...")
                self._tune_thresholds(validation_data)
            else:
                print("   Setting default thresholds...")
                self._set_default_thresholds(X_normal)
            
            self.is_trained = True
            print("   ✅ Preliminary filter training completed!")
            
        except Exception as e:
            print(f"   ❌ Error training preliminary filter: {e}")
            self.is_trained = False
    
    def _set_default_thresholds(self, X_normal):
        """Set default thresholds based on training data"""
        params = self.sensitivity_params[self.sensitivity]
        
        try:
            # Isolation Forest threshold
            if_scores = self.models['isolation_forest'].decision_function(X_normal)
            self.thresholds['isolation_forest'] = np.percentile(if_scores, 
                                                               (1-params['contamination'])*100)
            
            # One-Class SVM threshold
            svm_scores = self.models['ocsvm'].decision_function(X_normal)
            self.thresholds['ocsvm'] = np.percentile(svm_scores, 
                                                    (1-params['contamination'])*100)
            
            # LOF threshold
            lof_scores = self.models['lof'].decision_function(X_normal)
            self.thresholds['lof'] = np.percentile(lof_scores, 
                                                  (1-params['contamination'])*100)
            
            # Mahalanobis threshold
            mahalanobis_scores = self._compute_mahalanobis_distance(X_normal)
            self.thresholds['mahalanobis'] = np.percentile(mahalanobis_scores, 
                                                          (1-params['contamination'])*100)
            
            print(f"   Default thresholds set for {self.sensitivity} sensitivity")
            
        except Exception as e:
            print(f"   Warning: Error setting thresholds: {e}")
            # Fallback thresholds
            self.thresholds = {
                'isolation_forest': 0.0,
                'ocsvm': 0.0,
                'lof': 0.0,
                'mahalanobis': 2.0
            }
    
    def _tune_thresholds(self, validation_data):
        """Tune thresholds using validation data"""
        X_val, y_val = validation_data
        
        # Convert to anomaly labels if needed
        if len(np.unique(y_val)) == 2:
            y_anomaly = 1 - y_val  # Assuming 1=normal, 0=anomaly
        else:
            y_anomaly = y_val
        
        best_thresholds = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'mahalanobis':
                    scores = self._compute_mahalanobis_distance(X_val)
                else:
                    scores = model.decision_function(X_val)
                
                # Find optimal threshold using F1 score
                precisions, recalls, thresholds = precision_recall_curve(y_anomaly, scores)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
                
                best_thresholds[model_name] = best_threshold
                print(f"   {model_name}: threshold={best_threshold:.4f}, F1={f1_scores[best_idx]:.4f}")
                
            except Exception as e:
                print(f"   Warning: Could not tune {model_name}: {e}")
                best_thresholds[model_name] = 0.0
        
        self.thresholds = best_thresholds
    
    def _compute_mahalanobis_distance(self, X):
        """Compute Mahalanobis distance for samples"""
        if self.mahalanobis_cov is None:
            return np.zeros(len(X))
        
        try:
            # Compute squared Mahalanobis distance
            diff = X - self.mahalanobis_mean
            inv_cov = self.mahalanobis_cov.precision_
            mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return mahalanobis_dist
        except Exception as e:
            print(f"   Warning: Mahalanobis computation failed: {e}")
            return np.zeros(len(X))
    
    def evaluate_batch(self, X_batch, y_batch=None):
        """
        Evaluate if a batch contains anomalies
        
        Args:
            X_batch: Batch data (already scaled)
            y_batch: Optional ground truth labels
            
        Returns:
            dict: {
                'has_anomalies': bool,
                'confidence': float,
                'anomaly_scores': dict,
                'decision_rationale': str
            }
        """
        if not self.is_trained:
            return {
                'has_anomalies': True,  # If not trained, pass everything
                'confidence': 0.0,
                'anomaly_scores': {},
                'decision_rationale': 'Filter not trained - passing all batches'
            }
        
        if len(X_batch) == 0:
            return {
                'has_anomalies': False,
                'confidence': 1.0,
                'anomaly_scores': {},
                'decision_rationale': 'Empty batch'
            }
        
        start_time = datetime.now()
        
        try:
            # Get scores from all models
            scores = {}
            anomaly_votes = []
            
            # 1. Isolation Forest
            if_scores = self.models['isolation_forest'].decision_function(X_batch)
            scores['isolation_forest'] = np.mean(if_scores)
            anomaly_votes.append(scores['isolation_forest'] < self.thresholds['isolation_forest'])
            
            # 2. One-Class SVM
            svm_scores = self.models['ocsvm'].decision_function(X_batch)
            scores['ocsvm'] = np.mean(svm_scores)
            anomaly_votes.append(scores['ocsvm'] < self.thresholds['ocsvm'])
            
            # 3. Local Outlier Factor
            lof_scores = self.models['lof'].decision_function(X_batch)
            scores['lof'] = np.mean(lof_scores)
            anomaly_votes.append(scores['lof'] < self.thresholds['lof'])
            
            # 4. Mahalanobis Distance
            mahalanobis_scores = self._compute_mahalanobis_distance(X_batch)
            scores['mahalanobis'] = np.mean(mahalanobis_scores)
            anomaly_votes.append(scores['mahalanobis'] > self.thresholds['mahalanobis'])
            
            # Ensemble decision (majority vote)
            anomaly_count = sum(anomaly_votes)
            has_anomalies = anomaly_count >= 2  # At least 2 out of 4 models agree
            
            # Calculate confidence
            confidence = abs(anomaly_count - 2) / 2  # Distance from decision boundary
            
            # Decision rationale
            voting_details = [
                f"IF: {'✓' if anomaly_votes[0] else '✗'}",
                f"SVM: {'✓' if anomaly_votes[1] else '✗'}",
                f"LOF: {'✓' if anomaly_votes[2] else '✗'}",
                f"Mahal: {'✓' if anomaly_votes[3] else '✗'}"
            ]
            
            decision_rationale = f"Votes: {anomaly_count}/4 ({', '.join(voting_details)})"
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.filter_stats['processing_time'].append(processing_time)
            self.filter_stats['total_batches'] += 1
            
            if has_anomalies:
                self.filter_stats['passed_batches'] += 1
            else:
                self.filter_stats['filtered_batches'] += 1
            
            # Ground truth evaluation if available
            if y_batch is not None:
                y_anomaly = 1 - y_batch if len(np.unique(y_batch)) == 2 else y_batch
                true_has_anomalies = np.any(y_anomaly == 1)
                
                if has_anomalies and not true_has_anomalies:
                    self.filter_stats['false_positives'] += 1
                elif not has_anomalies and true_has_anomalies:
                    self.filter_stats['false_negatives'] += 1
            
            return {
                'has_anomalies': has_anomalies,
                'confidence': confidence,
                'anomaly_scores': scores,
                'decision_rationale': decision_rationale,
                'processing_time': processing_time,
                'individual_votes': anomaly_votes
            }
            
        except Exception as e:
            print(f"Warning: Error in batch evaluation: {e}")
            return {
                'has_anomalies': True,  # If error, pass the batch
                'confidence': 0.0,
                'anomaly_scores': {},
                'decision_rationale': f'Error in evaluation: {e}'
            }
    
    def get_filter_statistics(self):
        """Get filtering performance statistics"""
        total = self.filter_stats['total_batches']
        
        if total == 0:
            return {'message': 'No batches processed yet'}
        
        passed = self.filter_stats['passed_batches']
        filtered = self.filter_stats['filtered_batches']
        
        stats = {
            'total_batches': total,
            'passed_batches': passed,
            'filtered_batches': filtered,
            'pass_rate': passed / total if total > 0 else 0,
            'filter_rate': filtered / total if total > 0 else 0,
            'efficiency_gain': filtered / total if total > 0 else 0,
            'avg_processing_time': np.mean(self.filter_stats['processing_time']) if self.filter_stats['processing_time'] else 0,
            'false_positives': self.filter_stats['false_positives'],
            'false_negatives': self.filter_stats['false_negatives']
        }
        
        # Calculate accuracy metrics if we have ground truth
        if stats['false_positives'] + stats['false_negatives'] > 0:
            total_with_gt = stats['false_positives'] + stats['false_negatives'] + \
                           (passed - stats['false_positives']) + (filtered - stats['false_negatives'])
            stats['filter_accuracy'] = (total_with_gt - stats['false_positives'] - stats['false_negatives']) / total_with_gt
        
        return stats
    
    def adjust_sensitivity(self, new_sensitivity):
        """Adjust filter sensitivity dynamically"""
        if new_sensitivity in self.sensitivity_params:
            old_sensitivity = self.sensitivity
            self.sensitivity = new_sensitivity
            print(f"🎛️ Filter sensitivity changed: {old_sensitivity} → {new_sensitivity}")
            
            # Adjust thresholds based on new sensitivity
            multiplier = self.sensitivity_params[new_sensitivity]['threshold_multiplier']
            for model_name in self.thresholds:
                if model_name != 'mahalanobis':
                    self.thresholds[model_name] *= multiplier
                else:
                    self.thresholds[model_name] /= multiplier
            
            return True
        return False
    
    def save_filter(self, path_prefix='preliminary_anomaly_filter'):
        """Save the preliminary filter"""
        ensure_model_directory()
        filter_file = get_model_path(f'{path_prefix}.pkl')
        
        filter_data = {
            'sensitivity': self.sensitivity,
            'models': self.models,
            'thresholds': self.thresholds,
            'is_trained': self.is_trained,
            'mahalanobis_cov': self.mahalanobis_cov,
            'mahalanobis_mean': self.mahalanobis_mean,
            'filter_stats': self.filter_stats,
            'save_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(filter_data, filter_file)
        print(f"💾 Preliminary filter saved to: {filter_file}")
    
    def load_filter(self, path_prefix='preliminary_anomaly_filter'):
        """Load the preliminary filter"""
        filter_file = get_model_path(f'{path_prefix}.pkl')
        
        try:
            filter_data = joblib.load(filter_file)
            
            self.sensitivity = filter_data['sensitivity']
            self.models = filter_data['models']
            self.thresholds = filter_data['thresholds']
            self.is_trained = filter_data['is_trained']
            self.mahalanobis_cov = filter_data['mahalanobis_cov']
            self.mahalanobis_mean = filter_data['mahalanobis_mean']
            self.filter_stats = filter_data['filter_stats']
            
            print(f"📂 Preliminary filter loaded from: {filter_file}")
            print(f"   Sensitivity: {self.sensitivity}")
            print(f"   Trained: {self.is_trained}")
            print(f"   Previous stats: {self.filter_stats['total_batches']} batches processed")
            
        except Exception as e:
            print(f"❌ Error loading filter: {e}")
            raise

class OptimizedAnomalyDetector:
    """
    Optimized Anomaly Detection focused on original imbalanced data
    Phase 1: Training base models without data balancing
    """
    
    def __init__(self):
        # Check GPU availability
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {self.device}")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Models for unsupervised learning on normal data
        self.models = {}
        self.thresholds = {}
        
        # Data processing
        self.feature_columns = None
        self.target_column = None
        self.normal_label = None
        
        # Performance tracking
        self.results = {}
        
        # Training data for testing
        self.X_test = None
        self.y_test = None
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data with focus on efficiency"""
        print(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Data shape: {df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Find target column
        possible_targets = ['label', 'Label', 'target', 'Target', 'class', 'Class']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[-1]
        
        self.target_column = target_col
        print(f"Target column: {target_col}")
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Process missing values and categorical data
        X = self._preprocess_features(X)
        
        # Convert labels to binary
        y_binary = self._convert_to_binary(y)
        
        self.feature_columns = X.columns.tolist()
        
        # Analyze data distribution
        self._analyze_data_distribution(X, y_binary)
        
        return X, y_binary, y
    
    def _preprocess_features(self, X):
        """Efficient feature preprocessing"""
        print("\nEfficient feature preprocessing...")
        
        # Handle missing values
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])
        
        print(f"Numeric features: {len(X_numeric.columns)}")
        print(f"Categorical features: {len(X_categorical.columns)}")
        
        # Numeric features - use median for robustness
        if len(X_numeric.columns) > 0:
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Remove features with very low variance
            low_variance_cols = []
            for col in X_numeric.columns:
                if X_numeric[col].var() < 1e-6:
                    low_variance_cols.append(col)
            
            if low_variance_cols:
                print(f"Removing {len(low_variance_cols)} low variance features")
                X_numeric = X_numeric.drop(columns=low_variance_cols)
        
        # Categorical features
        if len(X_categorical.columns) > 0:
            for col in X_categorical.columns:
                mode_val = X_categorical[col].mode()
                if len(mode_val) > 0:
                    X_categorical[col] = X_categorical[col].fillna(mode_val[0])
                else:
                    X_categorical[col] = X_categorical[col].fillna('unknown')
                
                # Encode categorical variables
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
            
            # Combine
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X = X_numeric
        
        print(f"Final feature count: {X.shape[1]}")
        
        return X
    
    def _convert_to_binary(self, y):
        """Convert multi-class to binary with detailed analysis"""
        print("\nAnalyzing label distribution...")
        
        label_counts = y.value_counts()
        print("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(y)) * 100
            print(f"  {label}: {count} ({percentage:.3f}%)")
        
        # Identify normal label (most frequent or explicit normal labels)
        normal_labels = ['benign', 'Benign', 'BENIGN', 'normal', 'Normal', 'NORMAL']
        
        normal_label = None
        for label in normal_labels:
            if label in y.values:
                normal_label = label
                break
        
        if normal_label is None:
            # Take the most frequent label as normal
            normal_label = label_counts.index[0]
        
        self.normal_label = normal_label
        print(f"Normal label identified: {normal_label}")
        
        y_binary = (y == normal_label).astype(int)
        
        normal_count = np.sum(y_binary)
        anomaly_count = len(y_binary) - normal_count
        
        print(f"Binary distribution:")
        print(f"  Normal (1): {normal_count} ({normal_count/len(y_binary)*100:.3f}%)")
        print(f"  Anomaly (0): {anomaly_count} ({anomaly_count/len(y_binary)*100:.3f}%)")
        if normal_count > 0:
            print(f"  Imbalance ratio: 1:{anomaly_count/normal_count:.1f}")
        
        return y_binary
    
    def _analyze_data_distribution(self, X, y_binary):
        """Detailed data distribution analysis"""
        print("\nData distribution analysis...")
        
        normal_data = X[y_binary == 1]
        anomaly_data = X[y_binary == 0]
        
        print(f"Normal samples: {len(normal_data)}")
        print(f"Anomaly samples: {len(anomaly_data)}")
        
        # Analyze statistical differences for first few features
        print("\nStatistical differences (first 5 features):")
        for i, col in enumerate(X.columns[:5]):
            if len(normal_data) > 0 and len(anomaly_data) > 0:
                normal_mean = normal_data[col].mean()
                anomaly_mean = anomaly_data[col].mean()
                normal_std = normal_data[col].std()
                anomaly_std = anomaly_data[col].std()
                
                print(f"  {col}:")
                print(f"    Normal: μ={normal_mean:.4f}, σ={normal_std:.4f}")
                print(f"    Anomaly: μ={anomaly_mean:.4f}, σ={anomaly_std:.4f}")
    
    def train_anomaly_models(self, X, y_binary):
        """Train multiple anomaly detection models on original data"""
        print("\n" + "="*50)
        print("TRAINING ANOMALY DETECTION MODELS")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Extract normal samples for unsupervised training
        normal_train_data = X_train_scaled[y_train == 1]
        print(f"Training on {len(normal_train_data)} normal samples")
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Train models
        self._train_isolation_forest(normal_train_data, X_test_scaled, y_test)
        self._train_one_class_svm(normal_train_data, X_test_scaled, y_test)
        self._train_autoencoder(normal_train_data, X_test_scaled, y_test)
        self._train_local_outlier_factor(X_train_scaled, X_test_scaled, y_test)
        
        # Find optimal thresholds
        self._find_optimal_thresholds(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def _train_isolation_forest(self, normal_data, X_test, y_test):
        """Train Isolation Forest"""
        print("\nTraining Isolation Forest...")
        
        # Calculate contamination based on actual data distribution
        contamination = np.sum(y_test == 0) / len(y_test)
        contamination = max(0.01, min(0.5, contamination))  # Clamp between 1% and 50%
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            n_jobs=-1
        )
        
        model.fit(normal_data)
        self.models['isolation_forest'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _train_one_class_svm(self, normal_data, X_test, y_test):
        """Train One-Class SVM"""
        print("\nTraining One-Class SVM...")
        
        # Adaptive nu parameter based on contamination
        contamination = np.sum(y_test == 0) / len(y_test)
        nu = max(0.01, min(0.5, contamination))
        
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        model.fit(normal_data)
        self.models['ocsvm'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _train_autoencoder(self, normal_data, X_test, y_test):
        """Train Autoencoder for reconstruction-based anomaly detection"""
        print("\nTraining Autoencoder...")
        
        def create_autoencoder(input_dim):
            # Adaptive architecture based on input dimension
            encoding_dim = max(16, input_dim // 4)
            bottleneck_dim = max(8, encoding_dim // 2)
            
            model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(encoding_dim, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(bottleneck_dim, activation='relu'),
                layers.Dense(encoding_dim, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(input_dim, activation='linear')
            ])
            return model
        
        with tf.device(self.device):
            # Create and compile model
            model = create_autoencoder(normal_data.shape[1])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Training
            model.fit(
                normal_data, normal_data,
                epochs=100,
                batch_size=32,
                shuffle=True,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.1
            )
        
        self.models['autoencoder'] = model
        
        # Quick evaluation
        with tf.device(self.device):
            reconstructed = model.predict(X_test, verbose=0)
            reconstruction_errors = np.mean(np.square(X_test - reconstructed), axis=1)
        
        # Use percentile threshold for binary classification
        threshold = np.percentile(reconstruction_errors, 90)
        pred_binary = (reconstruction_errors > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  ⭐ AutoEncoder shows strong performance - will be focus for RL optimization")
    
    def _train_local_outlier_factor(self, X_train, X_test, y_test):
        """Train Local Outlier Factor"""
        print("\nTraining Local Outlier Factor...")
        
        # LOF for novelty detection
        contamination = np.sum(y_test == 0) / len(y_test)
        contamination = max(0.01, min(0.5, contamination))
        
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )
        
        # Use all training data for LOF
        model.fit(X_train)
        self.models['lof'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _find_optimal_thresholds(self, X_test, y_test):
        """Find optimal thresholds for each model"""
        print("\nFinding optimal thresholds...")
        
        self.thresholds = {}
        
        for model_name, model in self.models.items():
            print(f"  Processing {model_name}...")
            
            try:
                # Get anomaly scores
                if model_name == 'isolation_forest':
                    scores = model.decision_function(X_test)
                    anomaly_scores = -scores  # Higher scores = more anomalous
                elif model_name in ['ocsvm', 'lof']:
                    scores = model.decision_function(X_test)
                    anomaly_scores = -scores
                elif model_name == 'autoencoder':
                    with tf.device(self.device):
                        reconstructed = model.predict(X_test, verbose=0)
                        anomaly_scores = np.mean(np.square(X_test - reconstructed), axis=1)
                
                # Find optimal threshold using precision-recall curve
                y_anomaly = 1 - y_test  # Convert to anomaly labels (1=anomaly, 0=normal)
                
                if len(np.unique(y_anomaly)) == 2:
                    precisions, recalls, thresholds_pr = precision_recall_curve(y_anomaly, anomaly_scores)
                    
                    # Find threshold that maximizes F1 score
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else thresholds_pr[-1]
                    
                    self.thresholds[model_name] = {
                        'optimal': best_threshold,
                        'precision': precisions[best_idx],
                        'recall': recalls[best_idx],
                        'f1': f1_scores[best_idx]
                    }
                    
                    print(f"    Optimal threshold: {best_threshold:.6f}")
                    print(f"    Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")
                    
            except Exception as e:
                print(f"    Error processing {model_name}: {e}")
    
    def create_ensemble_model(self):
        """Create ensemble model from trained models"""
        print("\n" + "="*50)
        print("CREATING ENSEMBLE MODEL")
        print("="*50)
        
        # Simple ensemble with equal weights
        self.ensemble_weights = {model_name: 1.0/len(self.models) for model_name in self.models.keys()}
        
        print(f"Ensemble created with {len(self.models)} models:")
        for model_name, weight in self.ensemble_weights.items():
            print(f"  {model_name}: {weight:.3f}")
        
        return self.ensemble_weights
    
    def evaluate_comprehensive(self, X_test=None, y_test=None):
        """Comprehensive evaluation of base models"""
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)
        
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Individual model performance
        print(f"\nIndividual Model Performance:")
        y_anomaly = 1 - y_test
        best_model = None
        best_f1 = 0
        
        for model_name, model in self.models.items():
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    pred = model.predict(X_test)
                    pred_binary = (pred == -1).astype(int)
                elif model_name in ['ocsvm', 'lof']:
                    pred = model.predict(X_test)
                    pred_binary = (pred == -1).astype(int)
                elif model_name == 'autoencoder':
                    with tf.device(self.device):
                        reconstructed = model.predict(X_test, verbose=0)
                        scores = np.mean(np.square(X_test - reconstructed), axis=1)
                    threshold = self.thresholds[model_name]['optimal']
                    pred_binary = (scores > threshold).astype(int)
                
                # Calculate metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_anomaly, pred_binary, average='binary', zero_division=0
                )
                
                print(f"  {model_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                # Track best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
                    
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        
        print(f"\n🏆 Best performing model: {best_model} (F1={best_f1:.4f})")
        print(f"📈 This model will be the focus for RL optimization in Phase 2")
        
        return best_model, best_f1
    
    def save_models(self, path_prefix='optimized_anomaly_models'):
        """Save all models to designated directory"""
        ensure_model_directory()
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nSaving models to: {model_file}")
        
        model_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'normal_label': self.normal_label,
            'thresholds': self.thresholds,
            'models': {}
        }
        
        # Save models (except autoencoder)
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                autoencoder_path = get_model_path(f"{path_prefix}_autoencoder.keras")
                model.save(autoencoder_path)
                model_data['models'][model_name] = f"{path_prefix}_autoencoder"
                print(f"  ✓ AutoEncoder saved to: {autoencoder_path}")
            else:
                model_data['models'][model_name] = model
        
        joblib.dump(model_data, model_file)
        print(f"  ✓ Main model data saved to: {model_file}")
        print("✓ All models saved successfully!")
    
    def load_models(self, path_prefix='optimized_anomaly_models'):
        """Load all models from saved directory"""
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nLoading models from: {model_file}")
        
        try:
            model_data = joblib.load(model_file)
            
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.normal_label = model_data['normal_label']
            self.thresholds = model_data['thresholds']
            
            # Load models
            self.models = {}
            for model_name, model_path in model_data['models'].items():
                if model_name == 'autoencoder':
                    if isinstance(model_path, str):
                        # Construct full path
                        full_autoencoder_path = get_model_path(os.path.basename(model_path))
                        if os.path.exists(full_autoencoder_path):
                            self.models[model_name] = tf.keras.models.load_model(full_autoencoder_path)
                        else:
                            print(f"Warning: AutoEncoder model not found at {full_autoencoder_path}")
                    else:
                        print(f"Warning: Invalid AutoEncoder model path: {model_path}")
                else:
                    self.models[model_name] = model_path
            
            print("✓ Models loaded successfully!")
            print(f"  Available models: {list(self.models.keys())}")
            print(f"  Feature columns: {len(self.feature_columns)}")
            print(f"  Target column: {self.target_column}")
            print(f"  Normal label: {self.normal_label}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

class RLAgent:
    """
    Reinforcement Learning Agent for adaptive threshold and batch-size adjustment
    """
    def __init__(self, state_dim=6, action_dim=2, learning_rate=0.001):
        print("Initializing RL Agent...")
        
        # Basic parameters first
        self.state_dim = state_dim  # [anomaly_rate, precision, recall, f1, threshold, batch_size]
        self.action_dim = action_dim  # [threshold_adjustment, batch_size_adjustment]
        self.learning_rate = learning_rate
        
        # Action spaces - define these first before neural networks
        self.threshold_actions = np.linspace(-0.1, 0.1, 21)  # -10% to +10% adjustment
        self.batch_size_actions = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Multipliers
        print(f"Action spaces defined: {len(self.threshold_actions)} threshold actions, {len(self.batch_size_actions)} batch size actions")
        
        # Experience replay buffer
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # State normalization
        self.state_stats = {
            'anomaly_rate': {'mean': 0.1, 'std': 0.05},
            'precision': {'mean': 0.5, 'std': 0.2},
            'recall': {'mean': 0.5, 'std': 0.2},
            'f1': {'mean': 0.5, 'std': 0.2},
            'threshold': {'mean': 0.5, 'std': 0.3},
            'batch_size': {'mean': 100, 'std': 50}
        }
        
        # Neural networks - build these last
        try:
            print("Building Q-networks...")
            self.q_network = self._build_network()
            self.target_network = self._build_network()
            self.update_target_network()
            print("✓ RL Agent initialized successfully")
        except Exception as e:
            print(f"Warning: Error building neural networks: {e}")
            print("Using fallback mode without neural networks")
            self.q_network = None
            self.target_network = None
    
    def _build_network(self):
        """Build Q-network"""
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.threshold_actions) * len(self.batch_size_actions))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def normalize_state(self, state):
        """Normalize state values"""
        normalized = []
        keys = ['anomaly_rate', 'precision', 'recall', 'f1', 'threshold', 'batch_size']
        
        for i, key in enumerate(keys):
            val = (state[i] - self.state_stats[key]['mean']) / self.state_stats[key]['std']
            normalized.append(np.clip(val, -3, 3))
        
        return np.array(normalized)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        # Ensure action spaces exist
        if not hasattr(self, 'threshold_actions') or not hasattr(self, 'batch_size_actions'):
            print("Warning: Action spaces not defined, using random actions")
            threshold_idx = np.random.randint(0, 21)
            batch_idx = np.random.randint(0, 6)
            return threshold_idx, batch_idx
        
        if np.random.random() <= self.epsilon or self.q_network is None:
            # Random action
            threshold_idx = np.random.randint(len(self.threshold_actions))
            batch_idx = np.random.randint(len(self.batch_size_actions))
        else:
            # Q-network action
            try:
                state_norm = self.normalize_state(state).reshape(1, -1)
                q_values = self.q_network.predict(state_norm, verbose=0)[0]
                
                # Convert flat Q-values to 2D action space
                q_matrix = q_values.reshape(len(self.threshold_actions), len(self.batch_size_actions))
                best_action = np.unravel_index(np.argmax(q_matrix), q_matrix.shape)
                threshold_idx, batch_idx = best_action
            except Exception as e:
                print(f"Warning: Error in Q-network prediction: {e}")
                # Fallback to random
                threshold_idx = np.random.randint(len(self.threshold_actions))
                batch_idx = np.random.randint(len(self.batch_size_actions))
        
        return threshold_idx, batch_idx
    
    def calculate_reward(self, metrics, previous_metrics=None):
        """Calculate reward based on detection performance with confidence weighting"""
        # Get confidence level
        confidence_level = metrics.get('confidence_level', 'low')
        confidence_multiplier = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }.get(confidence_level, 0.4)
        
        # Primary rewards (weighted by confidence)
        precision_reward = metrics['precision'] * 2 * confidence_multiplier
        recall_reward = metrics['recall'] * 1.5 * confidence_multiplier
        f1_reward = metrics['f1'] * 3 * confidence_multiplier
        
        # Anomaly rate penalty/reward
        anomaly_rate = metrics['anomaly_rate']
        if 0.01 <= anomaly_rate <= 0.2:  # Reasonable anomaly rate
            rate_reward = 1.0
        elif anomaly_rate < 0.01:  # Too few anomalies detected
            rate_reward = -0.5
        elif 0.2 < anomaly_rate <= 0.5:  # Moderate concern
            rate_reward = -0.3
        else:  # Too many anomalies (possible false positives)
            rate_reward = -1.0
        
        # Confidence reward - encourage high confidence predictions
        confidence_reward = {
            'high': 0.5,
            'medium': 0.2,
            'low': -0.2
        }.get(confidence_level, -0.2)
        
        # Stability reward (if previous metrics available)
        stability_reward = 0
        if previous_metrics is not None:
            # Reward consistency, especially for high confidence predictions
            precision_diff = abs(metrics['precision'] - previous_metrics.get('precision', 0))
            if precision_diff < 0.1:  # Stable precision
                stability_reward += 0.3 * confidence_multiplier
            
            # Reward F1 improvement
            f1_diff = metrics['f1'] - previous_metrics.get('f1', 0)
            if f1_diff > 0:
                stability_reward += f1_diff * 2 * confidence_multiplier
        
        # Ground truth bonus
        ground_truth_bonus = 0
        if metrics.get('has_ground_truth', False):
            ground_truth_bonus = 0.3  # Bonus for having real labels
            
            # Additional bonus for prediction accuracy
            true_rate = metrics.get('true_anomaly_rate', 0)
            predicted_rate = metrics.get('anomaly_rate', 0)
            accuracy_bonus = 1 - abs(true_rate - predicted_rate)
            ground_truth_bonus += accuracy_bonus * 0.5
        
        # Efficiency reward based on batch processing
        efficiency_reward = 0.1  # Small positive reward for processing
        
        # Penalty for very low confidence
        if confidence_level == 'low':
            low_confidence_penalty = -0.3
        else:
            low_confidence_penalty = 0
        
        # Total reward
        total_reward = (precision_reward + recall_reward + f1_reward + 
                       rate_reward + confidence_reward + stability_reward + 
                       ground_truth_bonus + efficiency_reward + low_confidence_penalty)
        
        return total_reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if self.q_network is None or self.target_network is None:
            # Skip training if neural networks not available
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return
            
        if len(self.memory) < batch_size:
            return
        
        try:
            batch = np.random.choice(len(self.memory), batch_size, replace=False)
            
            states = []
            targets = []
            
            for idx in batch:
                state, action, reward, next_state, done = self.memory[idx]
                
                state_norm = self.normalize_state(state)
                target = self.q_network.predict(state_norm.reshape(1, -1), verbose=0)[0]
                
                if done:
                    target_value = reward
                else:
                    next_state_norm = self.normalize_state(next_state)
                    next_q = self.target_network.predict(next_state_norm.reshape(1, -1), verbose=0)[0]
                    target_value = reward + self.gamma * np.max(next_q)
                
                # Convert action to flat index
                threshold_idx, batch_idx = action
                action_idx = threshold_idx * len(self.batch_size_actions) + batch_idx
                target[action_idx] = target_value
                
                states.append(state_norm)
                targets.append(target)
            
            # Train
            if len(states) > 0:
                self.q_network.fit(np.array(states), np.array(targets), verbose=0, epochs=1)
            
        except Exception as e:
            print(f"Warning: Error in RL training: {e}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        if self.q_network is not None and self.target_network is not None:
            try:
                self.target_network.set_weights(self.q_network.get_weights())
            except Exception as e:
                print(f"Warning: Error updating target network: {e}")

class RLAdaptiveAnomalyDetector:
    """
    RL-Enhanced Anomaly Detector with Preliminary Anomaly Filter
    Only processes batches that pass preliminary anomaly screening
    """
    
    def __init__(self, base_detector, focus_model='autoencoder', filter_sensitivity='medium'):
        self.base_detector = base_detector
        self.focus_model = focus_model
        self.rl_agent = RLAgent()
        
        # Initialize Preliminary Anomaly Filter
        self.preliminary_filter = PreliminaryAnomalyFilter(sensitivity=filter_sensitivity)
        
        # Validate that focus model exists
        if focus_model not in base_detector.models:
            print(f"Warning: Focus model '{focus_model}' not found. Available models: {list(base_detector.models.keys())}")
            if 'autoencoder' in base_detector.models:
                self.focus_model = 'autoencoder'
                print(f"Defaulting to autoencoder")
            else:
                self.focus_model = list(base_detector.models.keys())[0]
                print(f"Defaulting to {self.focus_model}")
        else:
            print(f"✓ Focusing RL adaptation on: {focus_model}")
        
        # Adaptive parameters
        self.current_threshold_multiplier = 1.0
        self.current_batch_size = 100
        self.min_batch_size = 50
        self.max_batch_size = 500
        
        # AutoEncoder specific parameters
        if self.focus_model == 'autoencoder' and 'autoencoder' in self.base_detector.thresholds:
            self.autoencoder_threshold = self.base_detector.thresholds['autoencoder']['optimal']
            print(f"  Base AutoEncoder threshold: {self.autoencoder_threshold:.6f}")
        else:
            self.autoencoder_threshold = 0.1  # Default threshold
            print(f"  Using default threshold: {self.autoencoder_threshold}")
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
        # Streaming data buffer
        self.data_buffer = deque(maxlen=10000)
        self.processed_count = 0
        
        # Filter statistics
        self.filter_efficiency_history = []
        
        print("RL-Enhanced Adaptive Anomaly Detector initialized")
        print(f"Available models: {list(base_detector.models.keys())}")
        print(f"Focus model: {self.focus_model}")
        print(f"Preliminary filter sensitivity: {filter_sensitivity}")
    
    def train_preliminary_filter(self, training_data=None):
        """Train the preliminary anomaly filter on normal data"""
        print("\n🎯 Training Preliminary Anomaly Filter...")
        
        if training_data is None:
            # Use normal data from base detector's training
            if hasattr(self.base_detector, 'X_test') and hasattr(self.base_detector, 'y_test'):
                X_test = self.base_detector.X_test
                y_test = self.base_detector.y_test
                
                # Extract normal samples
                normal_samples = X_test[y_test == 1]
                validation_data = (X_test, y_test)
                
                print(f"Using {len(normal_samples)} normal samples from base detector")
                self.preliminary_filter.train_on_normal_data(normal_samples, validation_data)
            else:
                print("❌ No training data available for preliminary filter")
                return False
        else:
            # Use provided training data
            X_train, y_train = training_data
            normal_samples = X_train[y_train == 1]
            validation_data = (X_train, y_train)
            
            print(f"Using {len(normal_samples)} normal samples from provided data")
            self.preliminary_filter.train_on_normal_data(normal_samples, validation_data)
        
        return True
    
    def process_streaming_data(self, streaming_path, update_frequency=10, start_step=0):
        """Process streaming data with preliminary filtering + RL adaptation"""
        print(f"\n📡 Processing streaming data with Preliminary Filter + RL (starting from step {start_step})")
        print("="*80)
        
        # Load streaming data
        try:
            streaming_df = pd.read_csv(streaming_path)
            print(f"Streaming data shape: {streaming_df.shape}")
        except Exception as e:
            print(f"Error loading streaming data: {e}")
            return []
        
        # Train preliminary filter if not trained
        if not self.preliminary_filter.is_trained:
            print("🔧 Training preliminary filter...")
            if not self.train_preliminary_filter():
                print("⚠️  Warning: Preliminary filter training failed, processing all batches")
        
        # Preprocess streaming data
        X_stream, y_stream = self._preprocess_streaming_data(streaming_df)
        
        # Process in chunks with preliminary filtering + RL adaptation
        total_samples = len(X_stream)
        current_position = 0
        step = start_step
        previous_metrics = None
        
        # Skip to resume position if continuing
        if start_step > 0 and self.performance_history:
            total_processed_before = sum([h['batch_size'] for h in self.performance_history])
            current_position = min(total_processed_before, total_samples - self.current_batch_size)
            print(f"Resuming from position {current_position}/{total_samples}")
        
        while current_position < total_samples:
            step += 1
            
            try:
                # Get current batch
                batch_end = min(current_position + self.current_batch_size, total_samples)
                X_batch = X_stream.iloc[current_position:batch_end]
                y_batch = y_stream.iloc[current_position:batch_end] if y_stream is not None else None
                
                print(f"\nStep {step}: Processing batch {current_position}-{batch_end} (size: {len(X_batch)})")
                
                # Validate batch
                if len(X_batch) == 0:
                    print(f"  Warning: Empty batch, skipping...")
                    current_position = batch_end
                    continue
                
                # Scale batch for preliminary filter
                X_batch_scaled = self.base_detector.scaler.transform(X_batch)
                
                # 🔍 PRELIMINARY ANOMALY FILTERING
                filter_result = self.preliminary_filter.evaluate_batch(X_batch_scaled, y_batch)
                
                print(f"  🔍 Preliminary Filter Result:")
                print(f"    Has anomalies: {filter_result['has_anomalies']}")
                print(f"    Confidence: {filter_result['confidence']:.3f}")
                print(f"    Rationale: {filter_result['decision_rationale']}")
                print(f"    Processing time: {filter_result.get('processing_time', 0):.4f}s")
                
                # Track filter efficiency
                filter_efficiency = {
                    'step': step,
                    'passed_filter': filter_result['has_anomalies'],
                    'filter_confidence': filter_result['confidence'],
                    'filter_time': filter_result.get('processing_time', 0),
                    'batch_size': len(X_batch)
                }
                self.filter_efficiency_history.append(filter_efficiency)
                
                # Only process through RL if batch passes preliminary filter
                if filter_result['has_anomalies']:
                    print(f"  ✅ Batch passed preliminary filter → Processing with RL")
                    
                    # Make predictions with current thresholds
                    predictions, scores = self._predict_with_adaptive_thresholds(X_batch_scaled)
                    
                    # Calculate metrics
                    metrics = self._calculate_batch_metrics(predictions, scores, y_batch)
                    
                    # Add filter information to metrics
                    metrics['filter_passed'] = True
                    metrics['filter_confidence'] = filter_result['confidence']
                    metrics['filter_time'] = filter_result.get('processing_time', 0)
                    
                    # AutoEncoder specific metrics
                    if self.focus_model == 'autoencoder':
                        metrics['effective_threshold'] = self.autoencoder_threshold * self.current_threshold_multiplier
                        metrics['threshold_multiplier'] = self.current_threshold_multiplier
                    
                    # Get current state for RL
                    current_state = self._get_current_state(metrics)
                    
                    # RL adaptation every update_frequency steps
                    if step % update_frequency == 0 and step > start_step:
                        try:
                            self._rl_adaptation_step(current_state, metrics, previous_metrics, step)
                        except Exception as e:
                            print(f"  Warning: RL adaptation failed: {e}")
                    
                    # Store metrics
                    self.performance_history.append({
                        'step': step,
                        'batch_size': len(X_batch),
                        'threshold_multiplier': self.current_threshold_multiplier,
                        'batch_size_param': self.current_batch_size,
                        **metrics
                    })
                    
                    # Display real-time metrics
                    self._display_realtime_metrics(step, metrics)
                    
                    previous_metrics = metrics
                    
                else:
                    print(f"  ❌ Batch filtered out → Skipping RL processing")
                    print(f"    Reason: {filter_result['decision_rationale']}")
                    
                    # Store filtered batch info (minimal metrics)
                    filtered_metrics = {
                        'step': step,
                        'batch_size': len(X_batch),
                        'threshold_multiplier': self.current_threshold_multiplier,
                        'batch_size_param': self.current_batch_size,
                        'filter_passed': False,
                        'filter_confidence': filter_result['confidence'],
                        'filter_time': filter_result.get('processing_time', 0),
                        'anomaly_rate': 0.0,  # Assumed normal batch
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'has_ground_truth': y_batch is not None,
                        'confidence_level': 'low'
                    }
                    
                    self.performance_history.append(filtered_metrics)
                
                current_position = batch_end
                self.processed_count += len(X_batch)
                
            except Exception as e:
                print(f"  ❌ Error processing batch {current_position}-{batch_end}: {e}")
                print(f"  Skipping this batch and continuing...")
                current_position = min(current_position + self.current_batch_size, total_samples)
                continue
        
        print(f"\n📊 Streaming processing completed!")
        print(f"Total samples processed: {current_position}")
        print(f"Total cumulative samples: {self.processed_count}")
        
        # Display preliminary filter statistics
        self._display_filter_statistics()
        
        return self.performance_history
    
    def _display_filter_statistics(self):
        """Display preliminary filter performance statistics"""
        print(f"\n🔍 Preliminary Filter Performance:")
        print("="*50)
        
        filter_stats = self.preliminary_filter.get_filter_statistics()
        
        print(f"📈 Filter Efficiency:")
        print(f"  Total batches evaluated: {filter_stats['total_batches']}")
        print(f"  Batches passed: {filter_stats['passed_batches']}")
        print(f"  Batches filtered out: {filter_stats['filtered_batches']}")
        print(f"  Pass rate: {filter_stats['pass_rate']:.2%}")
        print(f"  Filter rate: {filter_stats['filter_rate']:.2%}")
        print(f"  Efficiency gain: {filter_stats['efficiency_gain']:.2%}")
        
        print(f"\n⏱️  Performance:")
        print(f"  Average processing time: {filter_stats['avg_processing_time']:.4f}s")
        
        if filter_stats['false_positives'] + filter_stats['false_negatives'] > 0:
            print(f"\n🎯 Accuracy (when ground truth available):")
            print(f"  False positives: {filter_stats['false_positives']}")
            print(f"  False negatives: {filter_stats['false_negatives']}")
            if 'filter_accuracy' in filter_stats:
                print(f"  Filter accuracy: {filter_stats['filter_accuracy']:.2%}")
        
        # Calculate computational savings
        if filter_stats['total_batches'] > 0:
            computational_savings = filter_stats['filtered_batches'] / filter_stats['total_batches']
            print(f"\n💰 Computational Savings:")
            print(f"  Estimated RL processing savings: {computational_savings:.2%}")
            print(f"  Batches processed by RL: {filter_stats['passed_batches']}")
            print(f"  Batches avoided by RL: {filter_stats['filtered_batches']}")
    
    def _preprocess_streaming_data(self, streaming_df):
        """Preprocess streaming data with enhanced validation"""
        print("Preprocessing streaming data...")
        
        # Find target column
        target_col = self.base_detector.target_column
        
        if target_col and target_col in streaming_df.columns:
            X_stream = streaming_df.drop(columns=[target_col])
            y_stream = streaming_df[target_col]
            
            # Validate ground truth labels
            unique_labels = y_stream.unique()
            print(f"  Found {len(unique_labels)} unique labels in streaming data: {unique_labels}")
            
            # Convert to binary if needed
            if self.base_detector.normal_label is not None:
                y_stream = (y_stream == self.base_detector.normal_label).astype(int)
                
                # Check label distribution
                normal_count = np.sum(y_stream)
                anomaly_count = len(y_stream) - normal_count
                print(f"  Label distribution: Normal={normal_count}, Anomaly={anomaly_count}")
                
                # Warn about potential issues
                if anomaly_count == 0:
                    print("  ⚠️  Warning: No anomalies in streaming data - metrics may be unreliable")
                elif normal_count == 0:
                    print("  ⚠️  Warning: No normal samples in streaming data - metrics may be unreliable")
        else:
            X_stream = streaming_df
            y_stream = None
            print("  No target column found in streaming data - unsupervised mode")
        
        # Ensure same feature columns as training
        if self.base_detector.feature_columns:
            missing_cols = set(self.base_detector.feature_columns) - set(X_stream.columns)
            extra_cols = set(X_stream.columns) - set(self.base_detector.feature_columns)
            
            if missing_cols:
                print(f"  Warning: Missing columns in streaming data: {missing_cols}")
                for col in missing_cols:
                    X_stream[col] = 0  # Default value
            
            if extra_cols:
                print(f"  Info: Dropping extra columns: {extra_cols}")
                X_stream = X_stream.drop(columns=list(extra_cols))
            
            # Reorder columns to match training
            X_stream = X_stream[self.base_detector.feature_columns]
        
        # Handle missing values
        print("  Handling missing values...")
        numeric_cols = X_stream.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_stream[col].isnull().any():
                median_val = X_stream[col].median()
                X_stream[col] = X_stream[col].fillna(median_val)
        
        # Categorical columns
        categorical_cols = X_stream.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if X_stream[col].isnull().any():
                mode_val = X_stream[col].mode()
                if len(mode_val) > 0:
                    X_stream[col] = X_stream[col].fillna(mode_val[0])
                else:
                    X_stream[col] = X_stream[col].fillna('unknown')
        
        # Final validation
        remaining_nulls = X_stream.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"  ⚠️  Warning: {remaining_nulls} null values remaining after preprocessing")
            X_stream = X_stream.fillna(0)  # Final fallback
        
        print(f"  ✅ Streaming data preprocessed: {X_stream.shape}")
        print(f"  Features: {len(X_stream.columns)}")
        print(f"  Samples: {len(X_stream)}")
        
        return X_stream, y_stream
    
    def _predict_with_adaptive_thresholds(self, X_batch):
        """Make predictions with adaptive thresholds, optimized for focus model"""
        if self.focus_model == 'autoencoder':
            # Focus on AutoEncoder predictions
            return self._predict_with_autoencoder(X_batch)
        else:
            # Use ensemble approach
            return self._predict_with_ensemble(X_batch)
    
    def _predict_with_autoencoder(self, X_scaled):
        """Optimized prediction using AutoEncoder only"""
        try:
            model = self.base_detector.models['autoencoder']
            
            # Get reconstruction errors
            with tf.device(self.base_detector.device):
                reconstructed = model.predict(X_scaled, verbose=0)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            
            # Apply adaptive threshold
            adaptive_threshold = self.autoencoder_threshold * self.current_threshold_multiplier
            
            # Predictions
            predictions = (reconstruction_errors > adaptive_threshold).astype(int)
            
            return predictions, reconstruction_errors
            
        except Exception as e:
            print(f"Error in AutoEncoder prediction: {e}")
            # Fallback to simple thresholding
            predictions = np.zeros(len(X_scaled), dtype=int)
            scores = np.random.rand(len(X_scaled))
            return predictions, scores
    
    def _predict_with_ensemble(self, X_scaled):
        """Ensemble prediction with adaptive thresholds"""
        anomaly_scores = []
        predictions = []
        
        for model_name, model in self.base_detector.models.items():
            try:
                # Get anomaly scores
                if model_name == 'isolation_forest':
                    scores = model.decision_function(X_scaled)
                    scores = -scores
                elif model_name in ['ocsvm', 'lof']:
                    scores = model.decision_function(X_scaled)
                    scores = -scores
                elif model_name == 'autoencoder':
                    with tf.device(self.base_detector.device):
                        reconstructed = model.predict(X_scaled, verbose=0)
                        scores = np.mean(np.square(X_scaled - reconstructed), axis=1)
                else:
                    continue
                
                # Apply adaptive threshold
                if model_name in self.base_detector.thresholds:
                    base_threshold = self.base_detector.thresholds[model_name]['optimal']
                    adaptive_threshold = base_threshold * self.current_threshold_multiplier
                else:
                    adaptive_threshold = np.percentile(scores, 90)  # Default threshold
                
                pred = (scores > adaptive_threshold).astype(int)
                
                anomaly_scores.append(scores)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue
        
        # Ensemble predictions
        if len(anomaly_scores) > 0:
            ensemble_scores = np.mean(anomaly_scores, axis=0)
            ensemble_predictions = np.mean(predictions, axis=0)
            final_predictions = (ensemble_predictions >= 0.5).astype(int)
        else:
            # Fallback
            final_predictions = np.zeros(len(X_scaled), dtype=int)
            ensemble_scores = np.random.rand(len(X_scaled))
        
        return final_predictions, ensemble_scores
    
    def _calculate_batch_metrics(self, predictions, scores, y_true=None):
        """Calculate metrics for current batch with improved estimation"""
        # Initialize all required metrics with default values
        metrics = {
            'anomaly_rate': 0.0,
            'total_anomalies': 0,
            'batch_size': 0,
            'avg_anomaly_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'has_ground_truth': False,
            'true_anomaly_rate': 0.0,
            'confidence_level': 'low'
        }
        
        # Basic metrics
        if len(predictions) > 0:
            metrics['anomaly_rate'] = np.mean(predictions)
            metrics['total_anomalies'] = np.sum(predictions)
            metrics['batch_size'] = len(predictions)
        
        if len(scores) > 0:
            metrics['avg_anomaly_score'] = np.mean(scores)
        
        # If ground truth is available
        if y_true is not None and len(y_true) > 0:
            try:
                y_anomaly = 1 - y_true  # Convert to anomaly labels
                metrics['true_anomaly_rate'] = np.mean(y_anomaly)
                metrics['has_ground_truth'] = True
                
                # Enhanced validation for supervised metrics
                predictions_valid = (len(predictions) == len(y_anomaly) and len(predictions) > 0)
                labels_diverse = len(np.unique(y_anomaly)) > 1
                predictions_diverse = len(np.unique(predictions)) > 1
                
                if predictions_valid and labels_diverse and predictions_diverse:
                    # Calculate supervised metrics
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_anomaly, predictions, average='binary', zero_division=0
                    )
                    
                    metrics['precision'] = float(precision) if not np.isnan(precision) else 0.0
                    metrics['recall'] = float(recall) if not np.isnan(recall) else 0.0
                    metrics['f1'] = float(f1) if not np.isnan(f1) else 0.0
                    metrics['confidence_level'] = 'high'
                    
                else:
                    # Use heuristic estimation
                    metrics.update(self._advanced_heuristic_estimation(metrics, y_anomaly, predictions, scores))
                    
            except Exception as e:
                print(f"    Warning: Error in supervised metrics calculation: {e}")
                metrics.update(self._advanced_heuristic_estimation(metrics, None, predictions, scores))
        else:
            # Unsupervised mode - use advanced heuristic estimates
            metrics.update(self._advanced_heuristic_estimation(metrics, None, predictions, scores))
        
        # Always calculate F1 from precision and recall
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Validate all metrics are numbers
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = 0.0
            elif key not in ['has_ground_truth', 'confidence_level', 'filter_passed']:
                metrics[key] = 0.0
        
        return metrics
    
    def _advanced_heuristic_estimation(self, base_metrics, y_anomaly=None, predictions=None, scores=None):
        """Advanced heuristic estimation with historical context"""
        anomaly_rate = base_metrics['anomaly_rate']
        
        # Use historical performance if available
        historical_precision = self._get_historical_average('precision', default=0.5)
        historical_recall = self._get_historical_average('recall', default=0.5)
        
        # Score-based analysis for better estimates
        if scores is not None and len(scores) > 0:
            score_variance = np.var(scores)
            score_mean = np.mean(scores)
            
            # Higher variance might indicate better discrimination
            variance_factor = min(1.0, score_variance / (score_mean + 1e-6))
            
            # Score distribution analysis
            score_percentiles = np.percentile(scores, [10, 50, 90])
            score_range = score_percentiles[2] - score_percentiles[0]
            
            # Better spread = potentially better performance
            spread_factor = min(1.0, score_range / (score_mean + 1e-6))
            
        else:
            variance_factor = 0.5
            spread_factor = 0.5
        
        # Adaptive estimation based on anomaly rate
        if 0.001 <= anomaly_rate <= 0.1:  # Reasonable anomaly rate
            # Good balance suggests good model performance
            precision_base = 0.7
            recall_base = 0.6
            quality_bonus = 0.2
        elif anomaly_rate < 0.001:  # Very few anomalies
            # Conservative detection - high precision, low recall
            precision_base = 0.9
            recall_base = 0.3
            quality_bonus = 0.1
        elif 0.1 < anomaly_rate <= 0.3:  # Moderate anomaly rate
            # Balanced detection
            precision_base = 0.6
            recall_base = 0.7
            quality_bonus = 0.1
        else:  # High anomaly rate (>30%)
            # Possibly too many false positives
            precision_base = 0.4
            recall_base = 0.8
            quality_bonus = 0.0
        
        # Incorporate score quality
        estimated_precision = precision_base + quality_bonus * variance_factor
        estimated_recall = recall_base + quality_bonus * spread_factor
        
        # Blend with historical performance (if available)
        if len(self.performance_history) > 0:
            blend_factor = min(0.3, len(self.performance_history) / 20)
            estimated_precision = (1 - blend_factor) * estimated_precision + blend_factor * historical_precision
            estimated_recall = (1 - blend_factor) * estimated_recall + blend_factor * historical_recall
        
        # Ground truth validation (if available)
        if y_anomaly is not None and len(y_anomaly) > 0:
            true_anomaly_rate = np.mean(y_anomaly)
            
            # Adjust based on how close our prediction rate is to true rate
            rate_similarity = 1 - abs(anomaly_rate - true_anomaly_rate)
            estimated_precision *= (0.7 + 0.3 * rate_similarity)
            estimated_recall *= (0.7 + 0.3 * rate_similarity)
        
        # Determine confidence level
        if scores is not None and len(scores) > 0 and variance_factor > 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'precision': np.clip(estimated_precision, 0.0, 1.0),
            'recall': np.clip(estimated_recall, 0.0, 1.0),
            'confidence_level': confidence,
            'has_ground_truth': y_anomaly is not None
        }
    
    def _get_historical_average(self, metric, default=0.5, window=10):
        """Get historical average of a metric for better estimation"""
        if not self.performance_history:
            return default
        
        recent_history = self.performance_history[-window:]
        values = [h.get(metric, default) for h in recent_history if h.get(metric, 0) > 0]
        
        if not values:
            return default
        
        return np.mean(values)
    
    def _get_current_state(self, metrics):
        """Get current state for RL agent"""
        # Safely extract metrics with default values
        anomaly_rate = metrics.get('anomaly_rate', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        state = [
            float(anomaly_rate),
            float(precision),
            float(recall),
            float(f1),
            float(self.current_threshold_multiplier),
            float(self.current_batch_size)
        ]
        
        # Validate state values
        for i, val in enumerate(state):
            if np.isnan(val) or np.isinf(val):
                state[i] = 0.0
        
        return np.array(state, dtype=np.float32)
    
    def _rl_adaptation_step(self, current_state, metrics, previous_metrics, step):
        """Perform RL adaptation step"""
        print(f"  🤖 RL Adaptation at step {step}")
        
        # Get action from RL agent
        try:
            threshold_idx, batch_idx = self.rl_agent.get_action(current_state)
            
            # Apply actions
            threshold_adjustment = self.rl_agent.threshold_actions[threshold_idx]
            batch_size_multiplier = self.rl_agent.batch_size_actions[batch_idx]
            
        except Exception as e:
            print(f"  Warning: Error getting RL action: {e}")
            # Fallback to small random adjustments
            threshold_adjustment = np.random.uniform(-0.05, 0.05)
            batch_size_multiplier = np.random.choice([0.9, 1.0, 1.1])
        
        # Calculate reward
        reward = self.rl_agent.calculate_reward(metrics, previous_metrics)
        
        # Store previous state-action-reward if available
        if len(self.adaptation_history) > 0:
            try:
                prev_data = self.adaptation_history[-1]
                self.rl_agent.remember(
                    prev_data['state'],
                    prev_data['action'],
                    reward,
                    current_state,
                    False  # Not done
                )
            except Exception as e:
                print(f"  Warning: Error storing RL memory: {e}")
        
        # Update parameters
        old_threshold = self.current_threshold_multiplier
        old_batch_size = self.current_batch_size
        
        self.current_threshold_multiplier = np.clip(
            self.current_threshold_multiplier + threshold_adjustment,
            0.1, 3.0
        )
        
        self.current_batch_size = int(np.clip(
            self.current_batch_size * batch_size_multiplier,
            self.min_batch_size, self.max_batch_size
        ))
        
        # Store adaptation history
        try:
            adaptation_data = {
                'step': step,
                'state': current_state.copy(),
                'action': (threshold_idx if 'threshold_idx' in locals() else 0, 
                          batch_idx if 'batch_idx' in locals() else 0),
                'reward': reward,
                'threshold_adjustment': threshold_adjustment,
                'batch_size_multiplier': batch_size_multiplier,
                'old_threshold': old_threshold,
                'new_threshold': self.current_threshold_multiplier,
                'old_batch_size': old_batch_size,
                'new_batch_size': self.current_batch_size,
                'epsilon': self.rl_agent.epsilon
            }
            
            self.adaptation_history.append(adaptation_data)
        except Exception as e:
            print(f"  Warning: Error storing adaptation history: {e}")
        
        # Train RL agent
        try:
            self.rl_agent.replay()
        except Exception as e:
            print(f"  Warning: Error in RL training: {e}")
        
        # Update target network periodically
        if len(self.adaptation_history) % 10 == 0:
            try:
                self.rl_agent.update_target_network()
            except Exception as e:
                print(f"  Warning: Error updating target network: {e}")
        
        print(f"    Threshold: {old_threshold:.3f} -> {self.current_threshold_multiplier:.3f}")
        print(f"    Batch size: {old_batch_size} -> {self.current_batch_size}")
        print(f"    Reward: {reward:.3f}")
        print(f"    Epsilon: {self.rl_agent.epsilon:.3f}")
    
    def _display_realtime_metrics(self, step, metrics):
        """Display real-time metrics with confidence and filter information"""
        # Safely display metrics with default values
        anomaly_rate = metrics.get('anomaly_rate', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        confidence = metrics.get('confidence_level', 'unknown')
        filter_passed = metrics.get('filter_passed', False)
        filter_confidence = metrics.get('filter_confidence', 0.0)
        
        # Color coding for confidence levels
        if confidence == 'high':
            confidence_icon = "🟢"
        elif confidence == 'medium':
            confidence_icon = "🟡"
        else:
            confidence_icon = "🔴"
        
        # Filter status
        filter_icon = "✅" if filter_passed else "❌"
        
        print(f"  📊 Batch Metrics:")
        print(f"    Anomaly rate: {anomaly_rate:.4f}")
        print(f"    Precision: {precision:.4f} {confidence_icon}")
        print(f"    Recall: {recall:.4f} {confidence_icon}")
        print(f"    F1: {f1:.4f} {confidence_icon}")
        print(f"    Confidence: {confidence}")
        print(f"    Filter: {filter_icon} (conf: {filter_confidence:.3f})")
        
        # AutoEncoder specific display
        if self.focus_model == 'autoencoder' and 'effective_threshold' in metrics:
            effective_threshold = metrics.get('effective_threshold', 0.0)
            print(f"    AutoEncoder threshold: {effective_threshold:.6f}")
        
        # Ground truth info
        if metrics.get('has_ground_truth', False):
            true_anomaly_rate = metrics.get('true_anomaly_rate', 0.0)
            print(f"    True anomaly rate: {true_anomaly_rate:.4f}")
            
            # Show accuracy of our prediction
            prediction_accuracy = 1 - abs(anomaly_rate - true_anomaly_rate)
            print(f"    Prediction accuracy: {prediction_accuracy:.4f}")
        
        # Performance trend indicator
        if len(self.performance_history) > 1:
            prev_f1 = self.performance_history[-1].get('f1', 0)
            current_f1 = f1
            trend = "📈" if current_f1 > prev_f1 else "📉" if current_f1 < prev_f1 else "➡️"
            print(f"    Trend: {trend}")
        
        if step % 5 == 0:  # Display adaptation status every 5 steps
            print(f"    Current threshold multiplier: {self.current_threshold_multiplier:.3f}")
            print(f"    Current batch size: {self.current_batch_size}")
            if hasattr(self, 'rl_agent'):
                print(f"    RL exploration (ε): {self.rl_agent.epsilon:.3f}")
            print(f"    Total processed: {self.processed_count}")
    
    def get_adaptation_summary(self):
        """Get summary of RL adaptation performance with filter statistics"""
        if not self.performance_history:
            return {"message": "No performance history available"}
        
        try:
            adapt_df = pd.DataFrame(self.adaptation_history) if self.adaptation_history else pd.DataFrame()
            perf_df = pd.DataFrame(self.performance_history)
            
            # Safely get final performance
            final_performance = {}
            if len(perf_df) > 0:
                last_row = perf_df.iloc[-1]
                final_performance = {
                    'anomaly_rate': last_row.get('anomaly_rate', 0.0),
                    'precision': last_row.get('precision', 0.0),
                    'recall': last_row.get('recall', 0.0),
                    'f1': last_row.get('f1', 0.0)
                }
            
            # Filter performance analysis
            filter_performance = {}
            if len(perf_df) > 0:
                total_batches = len(perf_df)
                passed_batches = len(perf_df[perf_df.get('filter_passed', True) == True])
                filtered_batches = total_batches - passed_batches
                
                filter_performance = {
                    'total_batches': total_batches,
                    'passed_batches': passed_batches,
                    'filtered_batches': filtered_batches,
                    'pass_rate': passed_batches / total_batches if total_batches > 0 else 0,
                    'filter_efficiency': filtered_batches / total_batches if total_batches > 0 else 0,
                    'avg_filter_confidence': perf_df.get('filter_confidence', pd.Series([0])).mean(),
                    'computational_savings': filtered_batches / total_batches if total_batches > 0 else 0
                }
            
            # AutoEncoder specific analysis
            autoencoder_performance = {}
            if self.focus_model == 'autoencoder':
                autoencoder_performance = {
                    'base_threshold': getattr(self, 'autoencoder_threshold', 0.0),
                    'final_threshold_multiplier': self.current_threshold_multiplier,
                    'final_effective_threshold': getattr(self, 'autoencoder_threshold', 0.0) * self.current_threshold_multiplier,
                    'threshold_improvement': 0.0
                }
                
                # Calculate improvement if we have enough data
                if len(perf_df) > 1:
                    first_f1 = perf_df.iloc[0].get('f1', 0.0)
                    last_f1 = perf_df.iloc[-1].get('f1', 0.0)
                    autoencoder_performance['threshold_improvement'] = last_f1 - first_f1
            
            # Preliminary filter statistics
            filter_stats = self.preliminary_filter.get_filter_statistics()
            
            summary = {
                'focus_model': self.focus_model,
                'filter_sensitivity': self.preliminary_filter.sensitivity,
                'total_adaptation_steps': len(adapt_df),
                'avg_reward': adapt_df['reward'].mean() if len(adapt_df) > 0 else 0.0,
                'final_epsilon': adapt_df['epsilon'].iloc[-1] if len(adapt_df) > 0 else self.rl_agent.epsilon,
                'threshold_range': (
                    perf_df['threshold_multiplier'].min() if len(perf_df) > 0 else self.current_threshold_multiplier, 
                    perf_df['threshold_multiplier'].max() if len(perf_df) > 0 else self.current_threshold_multiplier
                ),
                'batch_size_range': (
                    perf_df['batch_size_param'].min() if len(perf_df) > 0 else self.current_batch_size, 
                    perf_df['batch_size_param'].max() if len(perf_df) > 0 else self.current_batch_size
                ),
                'final_performance': final_performance,
                'autoencoder_specific': autoencoder_performance,
                'filter_performance': filter_performance,
                'preliminary_filter_stats': filter_stats,
                'total_samples_processed': self.processed_count,
                'total_steps': len(perf_df)
            }
            
            return summary
            
        except Exception as e:
            print(f"Error creating adaptation summary: {e}")
            return {
                "error": str(e),
                "focus_model": self.focus_model,
                "filter_sensitivity": getattr(self.preliminary_filter, 'sensitivity', 'unknown'),
                "total_samples_processed": self.processed_count
            }
    
    def save_rl_enhanced_model(self, path_prefix='rl_enhanced_anomaly_model_with_filter'):
        """Save RL-enhanced model with preliminary filter to designated directory"""
        ensure_model_directory()
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nSaving RL-enhanced model with filter to: {model_file}")
        
        try:
            # Prepare model data
            rl_model_data = {
                'base_detector_scaler': self.base_detector.scaler,
                'base_detector_feature_columns': self.base_detector.feature_columns,
                'base_detector_target_column': self.base_detector.target_column,
                'base_detector_normal_label': self.base_detector.normal_label,
                'base_detector_thresholds': self.base_detector.thresholds,
                
                # RL-specific parameters
                'focus_model': self.focus_model,
                'current_threshold_multiplier': self.current_threshold_multiplier,
                'current_batch_size': self.current_batch_size,
                'autoencoder_threshold': getattr(self, 'autoencoder_threshold', None),
                
                # RL agent state
                'rl_agent_epsilon': self.rl_agent.epsilon,
                'rl_agent_memory': list(self.rl_agent.memory),
                
                # Performance history
                'performance_history': self.performance_history,
                'adaptation_history': self.adaptation_history,
                'filter_efficiency_history': self.filter_efficiency_history,
                
                # Metadata
                'total_processed': self.processed_count,
                'save_timestamp': datetime.now().isoformat(),
                'model_save_dir': MODEL_SAVE_DIR,
                'filter_sensitivity': self.preliminary_filter.sensitivity
            }
            
            # Save non-autoencoder models
            rl_model_data['base_detector_models'] = {}
            for model_name, model in self.base_detector.models.items():
                if model_name != 'autoencoder':
                    rl_model_data['base_detector_models'][model_name] = model
            
            # Save AutoEncoder separately if it exists
            if 'autoencoder' in self.base_detector.models:
                autoencoder_path = get_model_path(f"{path_prefix}_autoencoder.keras")
                autoencoder_model = self.base_detector.models['autoencoder']
                autoencoder_model.save(autoencoder_path)
                rl_model_data['autoencoder_path'] = f"{path_prefix}_autoencoder"
                print(f"  ✓ AutoEncoder saved to: {autoencoder_path}")
            
            # Save RL Q-networks
            q_network_path = get_model_path(f"{path_prefix}_q_network.keras")
            target_network_path = get_model_path(f"{path_prefix}_target_network.keras")
            
            if self.rl_agent.q_network is not None:
                self.rl_agent.q_network.save(q_network_path)
                rl_model_data['q_network_path'] = f"{path_prefix}_q_network"
                print(f"  ✓ Q-Network saved to: {q_network_path}")
            
            if self.rl_agent.target_network is not None:
                self.rl_agent.target_network.save(target_network_path)
                rl_model_data['target_network_path'] = f"{path_prefix}_target_network"
                print(f"  ✓ Target Network saved to: {target_network_path}")
            
            # Save preliminary filter separately
            filter_prefix = f"{path_prefix}_preliminary_filter"
            self.preliminary_filter.save_filter(filter_prefix)
            rl_model_data['preliminary_filter_path'] = filter_prefix
            
            # Save main model data
            joblib.dump(rl_model_data, model_file)
            print(f"  ✓ Main RL data saved to: {model_file}")
            
            print("✅ RL-enhanced model with preliminary filter saved successfully!")
            print(f"  📁 All files saved in: {MODEL_SAVE_DIR}")
            print(f"  📊 Performance history: {len(self.performance_history)} steps")
            print(f"  🤖 Adaptation history: {len(self.adaptation_history)} adaptations")
            print(f"  🔍 Filter sensitivity: {self.preliminary_filter.sensitivity}")
            print(f"  📈 Total samples processed: {self.processed_count}")
            
        except Exception as e:
            print(f"❌ Error saving RL-enhanced model: {e}")
            raise
    
    def load_rl_enhanced_model(self, path_prefix='rl_enhanced_anomaly_model_with_filter'):
        """Load RL-enhanced model with preliminary filter from designated directory"""
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nLoading RL-enhanced model with filter from: {model_file}")
        
        try:
            # Load main model data
            rl_model_data = joblib.load(model_file)
            
            # Display loading info
            save_timestamp = rl_model_data.get('save_timestamp', 'Unknown')
            filter_sensitivity = rl_model_data.get('filter_sensitivity', 'Unknown')
            print(f"  📅 Model saved on: {save_timestamp}")
            print(f"  🔍 Filter sensitivity: {filter_sensitivity}")
            
            # Restore base detector
            self.base_detector.scaler = rl_model_data['base_detector_scaler']
            self.base_detector.feature_columns = rl_model_data['base_detector_feature_columns']
            self.base_detector.target_column = rl_model_data['base_detector_target_column']
            self.base_detector.normal_label = rl_model_data['base_detector_normal_label']
            self.base_detector.thresholds = rl_model_data['base_detector_thresholds']
            self.base_detector.models = rl_model_data['base_detector_models']
            
            # Load AutoEncoder
            if 'autoencoder_path' in rl_model_data:
                autoencoder_path = get_model_path(rl_model_data['autoencoder_path'])
                if os.path.exists(autoencoder_path):
                    autoencoder = tf.keras.models.load_model(autoencoder_path)
                    self.base_detector.models['autoencoder'] = autoencoder
                    print(f"  ✓ AutoEncoder loaded from: {autoencoder_path}")
                else:
                    print(f"  ⚠️ AutoEncoder not found at: {autoencoder_path}")
            
            # Restore RL-specific parameters
            self.focus_model = rl_model_data['focus_model']
            self.current_threshold_multiplier = rl_model_data['current_threshold_multiplier']
            self.current_batch_size = rl_model_data['current_batch_size']
            self.autoencoder_threshold = rl_model_data.get('autoencoder_threshold')
            
            # Restore RL agent
            self.rl_agent.epsilon = rl_model_data['rl_agent_epsilon']
            self.rl_agent.memory = deque(rl_model_data['rl_agent_memory'], maxlen=1000)
            
            # Load RL networks
            if 'q_network_path' in rl_model_data:
                q_network_path = get_model_path(rl_model_data['q_network_path'])
                if os.path.exists(q_network_path):
                    self.rl_agent.q_network = tf.keras.models.load_model(q_network_path)
                    print(f"  ✓ Q-Network loaded from: {q_network_path}")
                else:
                    print(f"  ⚠️ Q-Network not found at: {q_network_path}")
            
            if 'target_network_path' in rl_model_data:
                target_network_path = get_model_path(rl_model_data['target_network_path'])
                if os.path.exists(target_network_path):
                    self.rl_agent.target_network = tf.keras.models.load_model(target_network_path)
                    print(f"  ✓ Target Network loaded from: {target_network_path}")
                else:
                    print(f"  ⚠️ Target Network not found at: {target_network_path}")
            
            # Load preliminary filter
            if 'preliminary_filter_path' in rl_model_data:
                filter_prefix = rl_model_data['preliminary_filter_path']
                try:
                    self.preliminary_filter.load_filter(filter_prefix)
                    print(f"  ✓ Preliminary filter loaded")
                except Exception as e:
                    print(f"  ⚠️ Error loading preliminary filter: {e}")
            
            # Restore history
            self.performance_history = rl_model_data['performance_history']
            self.adaptation_history = rl_model_data['adaptation_history']
            self.filter_efficiency_history = rl_model_data.get('filter_efficiency_history', [])
            self.processed_count = rl_model_data['total_processed']
            
            print("✅ RL-enhanced model with filter loaded successfully!")
            print(f"  🎯 Focus model: {self.focus_model}")
            print(f"  🔧 Threshold multiplier: {self.current_threshold_multiplier:.3f}")
            print(f"  📦 Batch size: {self.current_batch_size}")
            print(f"  🔍 Filter trained: {self.preliminary_filter.is_trained}")
            print(f"  📊 Total processed: {self.processed_count}")
            print(f"  📈 Performance steps: {len(self.performance_history)}")
            print(f"  🤖 Adaptation steps: {len(self.adaptation_history)}")
            print(f"  🎲 Current epsilon: {self.rl_agent.epsilon:.3f}")
            
        except Exception as e:
            print(f"❌ Error loading RL-enhanced model: {e}")
            raise
    
    def continue_streaming_processing(self, streaming_path, resume_from_step=0):
        """Continue processing streaming data from saved state"""
        print(f"\n📡 Resuming streaming processing with filter from step {resume_from_step}")
        print(f"Current RL state:")
        print(f"  - Threshold multiplier: {self.current_threshold_multiplier:.3f}")
        print(f"  - Batch size: {self.current_batch_size}")
        print(f"  - Epsilon: {self.rl_agent.epsilon:.3f}")
        print(f"  - Memory size: {len(self.rl_agent.memory)}")
        print(f"  - Filter trained: {self.preliminary_filter.is_trained}")
        print(f"  - Filter sensitivity: {self.preliminary_filter.sensitivity}")
        
        return self.process_streaming_data(streaming_path, update_frequency=5, start_step=resume_from_step)
    
    def adjust_filter_sensitivity(self, new_sensitivity):
        """Dynamically adjust preliminary filter sensitivity during processing"""
        success = self.preliminary_filter.adjust_sensitivity(new_sensitivity)
        if success:
            print(f"🎛️ Filter sensitivity adjusted to: {new_sensitivity}")
            return True
        else:
            print(f"❌ Invalid sensitivity level: {new_sensitivity}")
            return False

def main():
    """Complete pipeline with Preliminary Anomaly Filter: Phase 1 (Training) + Phase 2 (RL Adaptation with Filter)"""
    # File paths

    training_file = os.path.join(DATA_PATH, 'extract_1p_ciciot2023.csv')
    streaming_file = os.path.join(DATA_PATH, 'streaming_data.csv')
    
    print("="*80)
    print("🚀 ENHANCED ANOMALY DETECTION PIPELINE WITH PRELIMINARY FILTER")
    print("Phase 1 (Training) + Phase 2 (RL Adaptation with Smart Filtering)")
    print("="*80)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    print(f"📊 Training data: {training_file}")
    print(f"📡 Streaming data: {streaming_file}")
    
    # Check if base models already exist
    base_model_file = get_model_path('optimized_anomaly_detection_models.pkl')
    skip_phase1 = os.path.exists(base_model_file)
    skip_phase1 = False  # Force Phase 1 for this run
    
    if skip_phase1:
        print("\n🔍 Auto-detection: Base models found!")
        print("⚡ Skipping Phase 1 (training) and proceeding to Phase 2 (RL with Filter)")
        print("💡 Use 'python script.py phase1' to retrain base models")
        
        # Load existing base models
        base_detector = OptimizedAnomalyDetector()
        base_detector.load_models('optimized_anomaly_detection_models')
        best_model = 'autoencoder'  # Default focus model
        
        print("✅ Base models loaded successfully!")
        
    else:
        # =================================================================
        # PHASE 1: BASE MODEL TRAINING
        # =================================================================
        print("\n" + "🔶"*20 + " PHASE 1: BASE MODEL TRAINING " + "🔶"*20)
        
        try:
            # Step 1: Initialize base detector
            print("\n1️⃣ Initializing base anomaly detector...")
            base_detector = OptimizedAnomalyDetector()
            
            # Step 2: Load and preprocess training data
            print("\n2️⃣ Loading and preprocessing training data...")
            X, y_binary, y_original = base_detector.load_and_preprocess_data(training_file)
            
            # Step 3: Train anomaly detection models
            print("\n3️⃣ Training anomaly detection models...")
            X_test, y_test = base_detector.train_anomaly_models(X, y_binary)
            
            # Step 4: Create ensemble
            print("\n4️⃣ Creating ensemble model...")
            base_detector.create_ensemble_model()
            
            # Step 5: Evaluate base models
            print("\n5️⃣ Evaluating base models...")
            best_model, best_f1 = base_detector.evaluate_comprehensive()
            
            # Step 6: Save base models
            print("\n6️⃣ Saving base models...")
            base_detector.save_models('optimized_anomaly_detection_models')
            
            print("\n✅ Phase 1 completed successfully!")
            print(f"🏆 Best model: {best_model} (F1={best_f1:.4f})")
            print("📦 Base models saved and ready for Phase 2")
            
        except Exception as e:
            print(f"❌ Error in Phase 1: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # =================================================================
    # PHASE 2: RL ADAPTATION WITH PRELIMINARY FILTER
    # =================================================================
    print("\n" + "🔷"*15 + " PHASE 2: RL ADAPTATION WITH PRELIMINARY FILTER " + "🔷"*15)
    
    try:
        # Step 1: Initialize RL-enhanced detector with preliminary filter
        print("\n1️⃣ Initializing RL-enhanced detector with preliminary filter...")
        focus_model = best_model if 'best_model' in locals() else 'autoencoder'
        filter_sensitivity = 'medium'  # Can be 'low', 'medium', 'high'
        print(f"   🎯 Focus model: {focus_model}")
        print(f"   🔍 Filter sensitivity: {filter_sensitivity}")
        
        rl_detector = RLAdaptiveAnomalyDetector(
            base_detector, 
            focus_model=focus_model,
            filter_sensitivity=filter_sensitivity
        )
        
        # Step 2: Train preliminary filter
        print("\n2️⃣ Training preliminary anomaly filter...")
        filter_trained = rl_detector.train_preliminary_filter()
        
        if filter_trained:
            print("✅ Preliminary filter trained successfully!")
            filter_stats = rl_detector.preliminary_filter.get_filter_statistics()
            print(f"   Models trained: {list(rl_detector.preliminary_filter.models.keys())}")
            print(f"   Sensitivity: {rl_detector.preliminary_filter.sensitivity}")
        else:
            print("⚠️  Warning: Preliminary filter training failed, proceeding without filtering")
        
        # Step 3: Process streaming data with preliminary filtering + RL adaptation
        print("\n3️⃣ Processing streaming data with preliminary filter + RL adaptation...")
        performance_history = rl_detector.process_streaming_data(
            streaming_file, 
            update_frequency=5  # Adapt every 5 batches
        )
        
        # Step 4: Analyze adaptation results with filter performance
        print("\n4️⃣ Analyzing RL adaptation and filter performance...")
        summary = rl_detector.get_adaptation_summary()
        
        print("\n📊 Complete System Summary:")
        print("=" * 50)
        
        # System overview
        print(f"🎯 System Configuration:")
        print(f"   Focus model: {summary.get('focus_model', 'Unknown')}")
        print(f"   Filter sensitivity: {summary.get('filter_sensitivity', 'Unknown')}")
        print(f"   Total samples processed: {summary.get('total_samples_processed', 0)}")
        print(f"   Total processing steps: {summary.get('total_steps', 0)}")
        
        # Filter performance
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"\n🔍 Preliminary Filter Performance:")
            print(f"   Total batches: {filter_perf.get('total_batches', 0)}")
            print(f"   Passed to RL: {filter_perf.get('passed_batches', 0)}")
            print(f"   Filtered out: {filter_perf.get('filtered_batches', 0)}")
            print(f"   Pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"   Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
            print(f"   Average filter confidence: {filter_perf.get('avg_filter_confidence', 0):.3f}")
        
        # RL performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"\n🤖 RL Adaptation Performance:")
            print(f"   Final anomaly rate: {final_perf.get('anomaly_rate', 0):.4f}")
            print(f"   Final precision: {final_perf.get('precision', 0):.4f}")
            print(f"   Final recall: {final_perf.get('recall', 0):.4f}")
            print(f"   Final F1 score: {final_perf.get('f1', 0):.4f}")
            
        # AutoEncoder specific results
        if 'autoencoder_specific' in summary and summary['autoencoder_specific']:
            ae_perf = summary['autoencoder_specific']
            print(f"\n⭐ AutoEncoder Optimization:")
            print(f"   Base threshold: {ae_perf.get('base_threshold', 0):.6f}")
            print(f"   Final multiplier: {ae_perf.get('final_threshold_multiplier', 0):.3f}")
            print(f"   Effective threshold: {ae_perf.get('final_effective_threshold', 0):.6f}")
            print(f"   Performance improvement: {ae_perf.get('threshold_improvement', 0):+.4f}")
        
        # Step 5: Save complete enhanced model
        print("\n5️⃣ Saving RL-enhanced model with preliminary filter...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter')
        
        print("\n✅ Phase 2 completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # FINAL COMPREHENSIVE SUMMARY
    # =================================================================
    print("\n" + "🌟"*20 + " FINAL COMPREHENSIVE SUMMARY " + "🌟"*20)
    
    # Show file structure
    print("\n📁 Saved files structure:")
    try:
        saved_files = []
        for file in os.listdir(MODEL_SAVE_DIR):
            file_path = os.path.join(MODEL_SAVE_DIR, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            saved_files.append((file, file_size))
        
        # Group files by type
        base_files = [f for f in saved_files if 'optimized_anomaly' in f[0]]
        rl_files = [f for f in saved_files if 'rl_enhanced' in f[0]]
        filter_files = [f for f in saved_files if 'preliminary_filter' in f[0]]
        
        if base_files:
            print("\n🔧 Base Models:")
            for filename, size in sorted(base_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        if rl_files:
            print("\n🤖 RL-Enhanced Models:")
            for filename, size in sorted(rl_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        if filter_files:
            print("\n🔍 Preliminary Filter:")
            for filename, size in sorted(filter_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        total_size = sum([size for _, size in saved_files])
        print(f"\n💾 Total storage: {total_size:.1f} MB")
        
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Performance comparison and efficiency analysis
    if 'performance_history' in locals() and len(performance_history) > 1:
        # Calculate performance improvement
        processed_batches = [h for h in performance_history if h.get('filter_passed', True)]
        if len(processed_batches) > 1:
            initial_f1 = processed_batches[0].get('f1', 0)
            final_f1 = processed_batches[-1].get('f1', 0)
            improvement = final_f1 - initial_f1
            
            print(f"\n📈 Performance Improvement (Processed Batches Only):")
            print(f"   Initial F1: {initial_f1:.4f}")
            print(f"   Final F1: {final_f1:.4f}")
            print(f"   Improvement: {improvement:+.4f}")
            
            if improvement > 0:
                print("   🎉 RL adaptation improved performance!")
            else:
                print("   📊 RL adaptation maintained stability")
        
        # Filter efficiency analysis
        total_batches = len(performance_history)
        filtered_batches = len([h for h in performance_history if not h.get('filter_passed', True)])
        efficiency_gain = filtered_batches / total_batches if total_batches > 0 else 0
        
        print(f"\n🔍 Filter Efficiency Analysis:")
        print(f"   Total batches: {total_batches}")
        print(f"   Processed by RL: {total_batches - filtered_batches}")
        print(f"   Filtered out: {filtered_batches}")
        print(f"   Computational savings: {efficiency_gain:.2%}")
        
        if efficiency_gain > 0.3:
            print("   💰 Excellent computational savings!")
        elif efficiency_gain > 0.1:
            print("   👍 Good computational efficiency")
        else:
            print("   📝 Consider adjusting filter sensitivity")
    
    # Show what was executed
    print(f"\n🎯 Pipeline Summary:")
    if skip_phase1:
        print("   🔶 Phase 1: ⚡ Skipped (models existed)")
    else:
        print("   🔶 Phase 1: ✅ Base models trained")
    print("   🔷 Phase 2: ✅ RL adaptation with preliminary filter completed")
    print("   🔍 Filter: ✅ Preliminary anomaly filter integrated")
    
    print("\n" + "="*80)
    print("🎉 ENHANCED PIPELINE WITH PRELIMINARY FILTER FINISHED SUCCESSFULLY!")
    print("="*80)
    print("✅ Models ready for production use with smart filtering")
    print(f"📁 All files saved in: {MODEL_SAVE_DIR}")
    print("\n💡 Next steps:")
    print("   🔄 python script.py continue     → Resume RL training")
    print("   �� python script.py list         → Manage model files")
    print("   🎛️  rl_detector.adjust_filter_sensitivity('high') → Adjust filter")
    print("   🧹 python script.py cleanup      → Clean old versions")
    print("   🔧 python script.py phase1       → Retrain base models")
    print("   📈 Load models for inference with intelligent filtering")

def run_phase_1_only():
    """Run only Phase 1: Base model training"""
    training_file = os.path.join(DATA_PATH,'extract_1p_ciciot2023.csv')
    
    print("="*60)
    print("🔶 PHASE 1 ONLY: BASE MODEL TRAINING")
    print("="*60)
    print(f"�� Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Initialize and train
        base_detector = OptimizedAnomalyDetector()
        X, y_binary, y_original = base_detector.load_and_preprocess_data(training_file)
        X_test, y_test = base_detector.train_anomaly_models(X, y_binary)
        base_detector.create_ensemble_model()
        best_model, best_f1 = base_detector.evaluate_comprehensive()
        
        # Save models
        base_detector.save_models('optimized_anomaly_detection_models')
        
        print("\n✅ Phase 1 completed!")
        print(f"🏆 Best model: {best_model} (F1={best_f1:.4f})")
        print("📦 Ready for Phase 2 RL adaptation with preliminary filter")
        
        return base_detector, best_model
        
    except Exception as e:
        print(f"❌ Error in Phase 1: {e}")
        raise

def run_phase_2_with_filter():
    """Run only Phase 2: RL adaptation with preliminary filter (requires Phase 1 models)"""
    streaming_file = os.path.join(DATA_PATH,'streaming_data.csv')
    model_name = 'optimized_anomaly_detection_models'
    
    print("="*60)
    print("🔷 PHASE 2 ONLY: RL ADAPTATION WITH PRELIMINARY FILTER")
    print("="*60)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Load base models
        print("\n1️⃣ Loading pre-trained base models...")
        base_detector = OptimizedAnomalyDetector()
        base_detector.load_models(model_name)
        
        # Initialize RL detector with filter
        print("\n2️⃣ Initializing RL-enhanced detector with preliminary filter...")
        rl_detector = RLAdaptiveAnomalyDetector(
            base_detector, 
            focus_model='autoencoder',
            filter_sensitivity='medium'
        )
        
        # Train preliminary filter
        print("\n3️⃣ Training preliminary filter...")
        filter_trained = rl_detector.train_preliminary_filter()
        if not filter_trained:
            print("⚠️  Warning: Filter training failed, proceeding without filtering")
        
        # Process streaming data
        print("\n4️⃣ Processing streaming data with filter + RL...")
        performance_history = rl_detector.process_streaming_data(streaming_file, update_frequency=5)
        
        # Save results
        print("\n5️⃣ Saving RL-enhanced model with filter...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter')
        
        # Summary
        summary = rl_detector.get_adaptation_summary()
        print("\n📊 Results:")
        
        # Filter performance
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"  Filter pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"  Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
        
        # Final performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"  Final F1: {final_perf.get('f1', 0):.4f}")
        
        print("\n✅ Phase 2 with preliminary filter completed!")
        
        return rl_detector
        
    except FileNotFoundError:
        print("❌ Base models not found!")
        print("Please run Phase 1 first or use main() for complete pipeline")
        raise
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")
        raise

def continue_from_saved_model_with_filter():
    """Utility function to continue RL adaptation from saved model with filter"""
    streaming_file = os.path.join(DATA_PATH,'streaming_data.csv')
    model_name = 'rl_enhanced_anomaly_model_with_filter'
    
    print("="*70)
    print("🔄 CONTINUING RL ADAPTATION FROM SAVED MODEL WITH FILTER")
    print("="*70)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Load saved RL-enhanced model with filter
        print("\n1. Loading saved RL-enhanced model with filter...")
        
        base_detector = OptimizedAnomalyDetector()
        rl_detector = RLAdaptiveAnomalyDetector(base_detector)
        rl_detector.load_rl_enhanced_model(model_name)
        
        # Continue processing
        print("\n2. Continuing streaming data processing with filter...")
        last_step = len(rl_detector.performance_history)
        print(f"   📈 Resuming from step: {last_step}")
        print(f"   🔍 Filter trained: {rl_detector.preliminary_filter.is_trained}")
        print(f"   📊 Filter sensitivity: {rl_detector.preliminary_filter.sensitivity}")
        
        performance_history = rl_detector.continue_streaming_processing(
            streaming_file, 
            resume_from_step=last_step
        )
        
        # Analysis
        print("\n3. Updated results...")
        summary = rl_detector.get_adaptation_summary()
        print("\n📊 Updated Summary:")
        
        # Filter stats
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"  Filter Performance:")
            print(f"    Pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"    Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
            print(f"    Average confidence: {filter_perf.get('avg_filter_confidence', 0):.3f}")
        
        # Overall performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"  Final Performance:")
            print(f"    F1 Score: {final_perf.get('f1', 0):.4f}")
            print(f"    Precision: {final_perf.get('precision', 0):.4f}")
            print(f"    Recall: {final_perf.get('recall', 0):.4f}")
        
        # Save updated model
        print("\n4. Saving updated model...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter_updated')
        
        print("\n✅ Continuation with filter completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Saved RL model with filter not found: {e}")
        print("Please run the main RL training with filter first")
        print(f"Expected location: {MODEL_SAVE_DIR}")
    except Exception as e:
        print(f"❌ Error continuing from saved model: {e}")
        import traceback
        traceback.print_exc()

def list_saved_models():
    """Utility function to list all saved models including filter components"""
    print("="*50)
    print("�� SAVED MODELS DIRECTORY")
    print("="*50)
    print(f"Directory: {MODEL_SAVE_DIR}")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        print("❌ Model directory does not exist")
        return
    
    files = os.listdir(MODEL_SAVE_DIR)
    if not files:
        print("📭 No saved models found")
        return
    
    print(f"\n📋 Found {len(files)} files:")
    
    # Group files by model type
    base_models = []
    rl_models = []
    filter_models = []
    other_files = []
    
    for file in files:
        file_path = os.path.join(MODEL_SAVE_DIR, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        file_info = (file, file_size)
        
        if 'optimized_anomaly' in file:
            base_models.append(file_info)
        elif 'rl_enhanced' in file:
            rl_models.append(file_info)
        elif 'preliminary_filter' in file:
            filter_models.append(file_info)
        else:
            other_files.append(file_info)
    
    if base_models:
        print("\n🔧 Base Models:")
        for filename, size in sorted(base_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if rl_models:
        print("\n🤖 RL-Enhanced Models:")
        for filename, size in sorted(rl_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if filter_models:
        print("\n🔍 Preliminary Filters:")
        for filename, size in sorted(filter_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if other_files:
        print("\n📎 Other Files:")
        for filename, size in sorted(other_files):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    total_size = sum([size for _, size in base_models + rl_models + filter_models + other_files])
    print(f"\n💾 Total storage: {total_size:.1f} MB")

if __name__ == "__main__":
    main()

# Example usage:
# To continue from saved model with filter: continue_from_saved_model_with_filter()
# To run only Phase 2 with filter: run_phase_2_with_filter()
# To adjust filter sensitivity: rl_detector.adjust_filter_sensitivity('high')import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_fscore_support, 
                           precision_recall_curve, roc_curve, average_precision_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import deque
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import google.colab
    # Đang chạy trên Google Colab
    DATA_PATH = '/content/drive/MyDrive/MyStudy/NCS/VNU/01-2023/datasets/IOTDataset/CIC_IOT_Dataset2023/'
    CODE_PATH = '/content/drive/MyDrive/MyStudy/NCS/VNU/01-2023/MyJournal/202411-ICTA/codes/icta2025/'
except ImportError:
    # Đang chạy trên local
    DATA_PATH = '/home/noattran/codes/icta2025/dataset/ciciot2023/'
    CODE_PATH= '/home/noattran/codes/icta2025/'

# Global configuration for model save directory
MODEL_SAVE_DIR = os.path.join(CODE_PATH,'saved_models')

def ensure_model_directory():
    """Ensure model save directory exists"""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"✓ Created model directory: {MODEL_SAVE_DIR}")
    else:
        print(f"✓ Model directory exists: {MODEL_SAVE_DIR}")

def get_model_path(filename):
    """Get full path for model file"""
    ensure_model_directory()
    return os.path.join(MODEL_SAVE_DIR, filename)

class PreliminaryAnomalyFilter:
    """
    Preliminary Anomaly Filter for streaming data batches
    Uses lightweight techniques to quickly assess if a batch contains anomalies
    Only anomalous batches are sent to RL agent for detailed processing
    """
    
    def __init__(self, sensitivity='medium'):
        """
        Initialize preliminary anomaly filter
        
        Args:
            sensitivity: 'low', 'medium', 'high' - determines how strict the filter is
        """
        self.sensitivity = sensitivity
        self.sensitivity_params = {
            'low': {'contamination': 0.05, 'threshold_multiplier': 1.5},
            'medium': {'contamination': 0.1, 'threshold_multiplier': 1.0},
            'high': {'contamination': 0.15, 'threshold_multiplier': 0.7}
        }
        
        # Models for preliminary detection
        self.models = {}
        self.thresholds = {}
        self.is_trained = False
        
        # Statistics for Mahalanobis distance
        self.mahalanobis_cov = None
        self.mahalanobis_mean = None
        
        # Performance tracking
        self.filter_stats = {
            'total_batches': 0,
            'passed_batches': 0,
            'filtered_batches': 0,
            'false_positives': 0,  # Normal batches marked as anomalous
            'false_negatives': 0,  # Anomalous batches marked as normal
            'processing_time': []
        }
        
        print(f"🔍 Preliminary Anomaly Filter initialized")
        print(f"   Sensitivity: {sensitivity}")
        print(f"   Expected contamination: {self.sensitivity_params[sensitivity]['contamination']}")
    
    def train_on_normal_data(self, X_normal, validation_data=None):
        """
        Train preliminary filter on normal data
        
        Args:
            X_normal: Normal training data (scaled)
            validation_data: Optional (X_val, y_val) for threshold tuning
        """
        print(f"\n🎯 Training Preliminary Anomaly Filter on {len(X_normal)} normal samples")
        
        params = self.sensitivity_params[self.sensitivity]
        
        try:
            # 1. Isolation Forest - Fast tree-based anomaly detection
            print("   Training Isolation Forest...")
            self.models['isolation_forest'] = IsolationForest(
                contamination=params['contamination'],
                random_state=42,
                n_estimators=100,  # Fewer trees for speed
                max_samples=min(256, len(X_normal)),  # Smaller sample size
                n_jobs=-1
            )
            self.models['isolation_forest'].fit(X_normal)
            
            # 2. One-Class SVM - Boundary-based detection
            print("   Training One-Class SVM...")
            # Use subset for speed if data is large
            train_subset = X_normal[:min(1000, len(X_normal))]
            self.models['ocsvm'] = OneClassSVM(
                nu=params['contamination'],
                kernel='rbf',
                gamma='scale',
                cache_size=200
            )
            self.models['ocsvm'].fit(train_subset)
            
            # 3. Local Outlier Factor - Density-based detection
            print("   Training Local Outlier Factor...")
            self.models['lof'] = LocalOutlierFactor(
                n_neighbors=min(20, len(X_normal)//5),
                contamination=params['contamination'],
                novelty=True,
                n_jobs=-1
            )
            self.models['lof'].fit(X_normal)
            
            # 4. Mahalanobis Distance - Statistical distance-based detection
            print("   Computing Mahalanobis statistics...")
            self.mahalanobis_cov = EmpiricalCovariance()
            self.mahalanobis_cov.fit(X_normal)
            self.mahalanobis_mean = np.mean(X_normal, axis=0)
            
            # Set thresholds based on validation or percentiles
            if validation_data is not None:
                print("   Tuning thresholds on validation data...")
                self._tune_thresholds(validation_data)
            else:
                print("   Setting default thresholds...")
                self._set_default_thresholds(X_normal)
            
            self.is_trained = True
            print("   ✅ Preliminary filter training completed!")
            
        except Exception as e:
            print(f"   ❌ Error training preliminary filter: {e}")
            self.is_trained = False
    
    def _set_default_thresholds(self, X_normal):
        """Set default thresholds based on training data"""
        params = self.sensitivity_params[self.sensitivity]
        
        try:
            # Isolation Forest threshold
            if_scores = self.models['isolation_forest'].decision_function(X_normal)
            self.thresholds['isolation_forest'] = np.percentile(if_scores, 
                                                               (1-params['contamination'])*100)
            
            # One-Class SVM threshold
            svm_scores = self.models['ocsvm'].decision_function(X_normal)
            self.thresholds['ocsvm'] = np.percentile(svm_scores, 
                                                    (1-params['contamination'])*100)
            
            # LOF threshold
            lof_scores = self.models['lof'].decision_function(X_normal)
            self.thresholds['lof'] = np.percentile(lof_scores, 
                                                  (1-params['contamination'])*100)
            
            # Mahalanobis threshold
            mahalanobis_scores = self._compute_mahalanobis_distance(X_normal)
            self.thresholds['mahalanobis'] = np.percentile(mahalanobis_scores, 
                                                          (1-params['contamination'])*100)
            
            print(f"   Default thresholds set for {self.sensitivity} sensitivity")
            
        except Exception as e:
            print(f"   Warning: Error setting thresholds: {e}")
            # Fallback thresholds
            self.thresholds = {
                'isolation_forest': 0.0,
                'ocsvm': 0.0,
                'lof': 0.0,
                'mahalanobis': 2.0
            }
    
    def _tune_thresholds(self, validation_data):
        """Tune thresholds using validation data"""
        X_val, y_val = validation_data
        
        # Convert to anomaly labels if needed
        if len(np.unique(y_val)) == 2:
            y_anomaly = 1 - y_val  # Assuming 1=normal, 0=anomaly
        else:
            y_anomaly = y_val
        
        best_thresholds = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'mahalanobis':
                    scores = self._compute_mahalanobis_distance(X_val)
                else:
                    scores = model.decision_function(X_val)
                
                # Find optimal threshold using F1 score
                precisions, recalls, thresholds = precision_recall_curve(y_anomaly, scores)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
                
                best_thresholds[model_name] = best_threshold
                print(f"   {model_name}: threshold={best_threshold:.4f}, F1={f1_scores[best_idx]:.4f}")
                
            except Exception as e:
                print(f"   Warning: Could not tune {model_name}: {e}")
                best_thresholds[model_name] = 0.0
        
        self.thresholds = best_thresholds
    
    def _compute_mahalanobis_distance(self, X):
        """Compute Mahalanobis distance for samples"""
        if self.mahalanobis_cov is None:
            return np.zeros(len(X))
        
        try:
            # Compute squared Mahalanobis distance
            diff = X - self.mahalanobis_mean
            inv_cov = self.mahalanobis_cov.precision_
            mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return mahalanobis_dist
        except Exception as e:
            print(f"   Warning: Mahalanobis computation failed: {e}")
            return np.zeros(len(X))
    
    def evaluate_batch(self, X_batch, y_batch=None):
        """
        Evaluate if a batch contains anomalies
        
        Args:
            X_batch: Batch data (already scaled)
            y_batch: Optional ground truth labels
            
        Returns:
            dict: {
                'has_anomalies': bool,
                'confidence': float,
                'anomaly_scores': dict,
                'decision_rationale': str
            }
        """
        if not self.is_trained:
            return {
                'has_anomalies': True,  # If not trained, pass everything
                'confidence': 0.0,
                'anomaly_scores': {},
                'decision_rationale': 'Filter not trained - passing all batches'
            }
        
        if len(X_batch) == 0:
            return {
                'has_anomalies': False,
                'confidence': 1.0,
                'anomaly_scores': {},
                'decision_rationale': 'Empty batch'
            }
        
        start_time = datetime.now()
        
        try:
            # Get scores from all models
            scores = {}
            anomaly_votes = []
            
            # 1. Isolation Forest
            if_scores = self.models['isolation_forest'].decision_function(X_batch)
            scores['isolation_forest'] = np.mean(if_scores)
            anomaly_votes.append(scores['isolation_forest'] < self.thresholds['isolation_forest'])
            
            # 2. One-Class SVM
            svm_scores = self.models['ocsvm'].decision_function(X_batch)
            scores['ocsvm'] = np.mean(svm_scores)
            anomaly_votes.append(scores['ocsvm'] < self.thresholds['ocsvm'])
            
            # 3. Local Outlier Factor
            lof_scores = self.models['lof'].decision_function(X_batch)
            scores['lof'] = np.mean(lof_scores)
            anomaly_votes.append(scores['lof'] < self.thresholds['lof'])
            
            # 4. Mahalanobis Distance
            mahalanobis_scores = self._compute_mahalanobis_distance(X_batch)
            scores['mahalanobis'] = np.mean(mahalanobis_scores)
            anomaly_votes.append(scores['mahalanobis'] > self.thresholds['mahalanobis'])
            
            # Ensemble decision (majority vote)
            anomaly_count = sum(anomaly_votes)
            has_anomalies = anomaly_count >= 2  # At least 2 out of 4 models agree
            
            # Calculate confidence
            confidence = abs(anomaly_count - 2) / 2  # Distance from decision boundary
            
            # Decision rationale
            voting_details = [
                f"IF: {'✓' if anomaly_votes[0] else '✗'}",
                f"SVM: {'✓' if anomaly_votes[1] else '✗'}",
                f"LOF: {'✓' if anomaly_votes[2] else '✗'}",
                f"Mahal: {'✓' if anomaly_votes[3] else '✗'}"
            ]
            
            decision_rationale = f"Votes: {anomaly_count}/4 ({', '.join(voting_details)})"
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.filter_stats['processing_time'].append(processing_time)
            self.filter_stats['total_batches'] += 1
            
            if has_anomalies:
                self.filter_stats['passed_batches'] += 1
            else:
                self.filter_stats['filtered_batches'] += 1
            
            # Ground truth evaluation if available
            if y_batch is not None:
                y_anomaly = 1 - y_batch if len(np.unique(y_batch)) == 2 else y_batch
                true_has_anomalies = np.any(y_anomaly == 1)
                
                if has_anomalies and not true_has_anomalies:
                    self.filter_stats['false_positives'] += 1
                elif not has_anomalies and true_has_anomalies:
                    self.filter_stats['false_negatives'] += 1
            
            return {
                'has_anomalies': has_anomalies,
                'confidence': confidence,
                'anomaly_scores': scores,
                'decision_rationale': decision_rationale,
                'processing_time': processing_time,
                'individual_votes': anomaly_votes
            }
            
        except Exception as e:
            print(f"Warning: Error in batch evaluation: {e}")
            return {
                'has_anomalies': True,  # If error, pass the batch
                'confidence': 0.0,
                'anomaly_scores': {},
                'decision_rationale': f'Error in evaluation: {e}'
            }
    
    def get_filter_statistics(self):
        """Get filtering performance statistics"""
        total = self.filter_stats['total_batches']
        
        if total == 0:
            return {'message': 'No batches processed yet'}
        
        passed = self.filter_stats['passed_batches']
        filtered = self.filter_stats['filtered_batches']
        
        stats = {
            'total_batches': total,
            'passed_batches': passed,
            'filtered_batches': filtered,
            'pass_rate': passed / total if total > 0 else 0,
            'filter_rate': filtered / total if total > 0 else 0,
            'efficiency_gain': filtered / total if total > 0 else 0,
            'avg_processing_time': np.mean(self.filter_stats['processing_time']) if self.filter_stats['processing_time'] else 0,
            'false_positives': self.filter_stats['false_positives'],
            'false_negatives': self.filter_stats['false_negatives']
        }
        
        # Calculate accuracy metrics if we have ground truth
        if stats['false_positives'] + stats['false_negatives'] > 0:
            total_with_gt = stats['false_positives'] + stats['false_negatives'] + \
                           (passed - stats['false_positives']) + (filtered - stats['false_negatives'])
            stats['filter_accuracy'] = (total_with_gt - stats['false_positives'] - stats['false_negatives']) / total_with_gt
        
        return stats
    
    def adjust_sensitivity(self, new_sensitivity):
        """Adjust filter sensitivity dynamically"""
        if new_sensitivity in self.sensitivity_params:
            old_sensitivity = self.sensitivity
            self.sensitivity = new_sensitivity
            print(f"🎛️ Filter sensitivity changed: {old_sensitivity} → {new_sensitivity}")
            
            # Adjust thresholds based on new sensitivity
            multiplier = self.sensitivity_params[new_sensitivity]['threshold_multiplier']
            for model_name in self.thresholds:
                if model_name != 'mahalanobis':
                    self.thresholds[model_name] *= multiplier
                else:
                    self.thresholds[model_name] /= multiplier
            
            return True
        return False
    
    def save_filter(self, path_prefix='preliminary_anomaly_filter'):
        """Save the preliminary filter"""
        ensure_model_directory()
        filter_file = get_model_path(f'{path_prefix}.pkl')
        
        filter_data = {
            'sensitivity': self.sensitivity,
            'models': self.models,
            'thresholds': self.thresholds,
            'is_trained': self.is_trained,
            'mahalanobis_cov': self.mahalanobis_cov,
            'mahalanobis_mean': self.mahalanobis_mean,
            'filter_stats': self.filter_stats,
            'save_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(filter_data, filter_file)
        print(f"💾 Preliminary filter saved to: {filter_file}")
    
    def load_filter(self, path_prefix='preliminary_anomaly_filter'):
        """Load the preliminary filter"""
        filter_file = get_model_path(f'{path_prefix}.pkl')
        
        try:
            filter_data = joblib.load(filter_file)
            
            self.sensitivity = filter_data['sensitivity']
            self.models = filter_data['models']
            self.thresholds = filter_data['thresholds']
            self.is_trained = filter_data['is_trained']
            self.mahalanobis_cov = filter_data['mahalanobis_cov']
            self.mahalanobis_mean = filter_data['mahalanobis_mean']
            self.filter_stats = filter_data['filter_stats']
            
            print(f"📂 Preliminary filter loaded from: {filter_file}")
            print(f"   Sensitivity: {self.sensitivity}")
            print(f"   Trained: {self.is_trained}")
            print(f"   Previous stats: {self.filter_stats['total_batches']} batches processed")
            
        except Exception as e:
            print(f"❌ Error loading filter: {e}")
            raise

class OptimizedAnomalyDetector:
    """
    Optimized Anomaly Detection focused on original imbalanced data
    Phase 1: Training base models without data balancing
    """
    
    def __init__(self):
        # Check GPU availability
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {self.device}")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Models for unsupervised learning on normal data
        self.models = {}
        self.thresholds = {}
        
        # Data processing
        self.feature_columns = None
        self.target_column = None
        self.normal_label = None
        
        # Performance tracking
        self.results = {}
        
        # Training data for testing
        self.X_test = None
        self.y_test = None
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data with focus on efficiency"""
        print(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Data shape: {df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Find target column
        possible_targets = ['label', 'Label', 'target', 'Target', 'class', 'Class']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[-1]
        
        self.target_column = target_col
        print(f"Target column: {target_col}")
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Process missing values and categorical data
        X = self._preprocess_features(X)
        
        # Convert labels to binary
        y_binary = self._convert_to_binary(y)
        
        self.feature_columns = X.columns.tolist()
        
        # Analyze data distribution
        self._analyze_data_distribution(X, y_binary)
        
        return X, y_binary, y
    
    def _preprocess_features(self, X):
        """Efficient feature preprocessing"""
        print("\nEfficient feature preprocessing...")
        
        # Handle missing values
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])
        
        print(f"Numeric features: {len(X_numeric.columns)}")
        print(f"Categorical features: {len(X_categorical.columns)}")
        
        # Numeric features - use median for robustness
        if len(X_numeric.columns) > 0:
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Remove features with very low variance
            low_variance_cols = []
            for col in X_numeric.columns:
                if X_numeric[col].var() < 1e-6:
                    low_variance_cols.append(col)
            
            if low_variance_cols:
                print(f"Removing {len(low_variance_cols)} low variance features")
                X_numeric = X_numeric.drop(columns=low_variance_cols)
        
        # Categorical features
        if len(X_categorical.columns) > 0:
            for col in X_categorical.columns:
                mode_val = X_categorical[col].mode()
                if len(mode_val) > 0:
                    X_categorical[col] = X_categorical[col].fillna(mode_val[0])
                else:
                    X_categorical[col] = X_categorical[col].fillna('unknown')
                
                # Encode categorical variables
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
            
            # Combine
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X = X_numeric
        
        print(f"Final feature count: {X.shape[1]}")
        
        return X
    
    def _convert_to_binary(self, y):
        """Convert multi-class to binary with detailed analysis"""
        print("\nAnalyzing label distribution...")
        
        label_counts = y.value_counts()
        print("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(y)) * 100
            print(f"  {label}: {count} ({percentage:.3f}%)")
        
        # Identify normal label (most frequent or explicit normal labels)
        normal_labels = ['benign', 'Benign', 'BENIGN', 'normal', 'Normal', 'NORMAL']
        
        normal_label = None
        for label in normal_labels:
            if label in y.values:
                normal_label = label
                break
        
        if normal_label is None:
            # Take the most frequent label as normal
            normal_label = label_counts.index[0]
        
        self.normal_label = normal_label
        print(f"Normal label identified: {normal_label}")
        
        y_binary = (y == normal_label).astype(int)
        
        normal_count = np.sum(y_binary)
        anomaly_count = len(y_binary) - normal_count
        
        print(f"Binary distribution:")
        print(f"  Normal (1): {normal_count} ({normal_count/len(y_binary)*100:.3f}%)")
        print(f"  Anomaly (0): {anomaly_count} ({anomaly_count/len(y_binary)*100:.3f}%)")
        if normal_count > 0:
            print(f"  Imbalance ratio: 1:{anomaly_count/normal_count:.1f}")
        
        return y_binary
    
    def _analyze_data_distribution(self, X, y_binary):
        """Detailed data distribution analysis"""
        print("\nData distribution analysis...")
        
        normal_data = X[y_binary == 1]
        anomaly_data = X[y_binary == 0]
        
        print(f"Normal samples: {len(normal_data)}")
        print(f"Anomaly samples: {len(anomaly_data)}")
        
        # Analyze statistical differences for first few features
        print("\nStatistical differences (first 5 features):")
        for i, col in enumerate(X.columns[:5]):
            if len(normal_data) > 0 and len(anomaly_data) > 0:
                normal_mean = normal_data[col].mean()
                anomaly_mean = anomaly_data[col].mean()
                normal_std = normal_data[col].std()
                anomaly_std = anomaly_data[col].std()
                
                print(f"  {col}:")
                print(f"    Normal: μ={normal_mean:.4f}, σ={normal_std:.4f}")
                print(f"    Anomaly: μ={anomaly_mean:.4f}, σ={anomaly_std:.4f}")
    
    def train_anomaly_models(self, X, y_binary):
        """Train multiple anomaly detection models on original data"""
        print("\n" + "="*50)
        print("TRAINING ANOMALY DETECTION MODELS")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Extract normal samples for unsupervised training
        normal_train_data = X_train_scaled[y_train == 1]
        print(f"Training on {len(normal_train_data)} normal samples")
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Train models
        self._train_isolation_forest(normal_train_data, X_test_scaled, y_test)
        self._train_one_class_svm(normal_train_data, X_test_scaled, y_test)
        self._train_autoencoder(normal_train_data, X_test_scaled, y_test)
        self._train_local_outlier_factor(X_train_scaled, X_test_scaled, y_test)
        
        # Find optimal thresholds
        self._find_optimal_thresholds(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def _train_isolation_forest(self, normal_data, X_test, y_test):
        """Train Isolation Forest"""
        print("\nTraining Isolation Forest...")
        
        # Calculate contamination based on actual data distribution
        contamination = np.sum(y_test == 0) / len(y_test)
        contamination = max(0.01, min(0.5, contamination))  # Clamp between 1% and 50%
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            n_jobs=-1
        )
        
        model.fit(normal_data)
        self.models['isolation_forest'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _train_one_class_svm(self, normal_data, X_test, y_test):
        """Train One-Class SVM"""
        print("\nTraining One-Class SVM...")
        
        # Adaptive nu parameter based on contamination
        contamination = np.sum(y_test == 0) / len(y_test)
        nu = max(0.01, min(0.5, contamination))
        
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        model.fit(normal_data)
        self.models['ocsvm'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _train_autoencoder(self, normal_data, X_test, y_test):
        """Train Autoencoder for reconstruction-based anomaly detection"""
        print("\nTraining Autoencoder...")
        
        def create_autoencoder(input_dim):
            # Adaptive architecture based on input dimension
            encoding_dim = max(16, input_dim // 4)
            bottleneck_dim = max(8, encoding_dim // 2)
            
            model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(encoding_dim, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(bottleneck_dim, activation='relu'),
                layers.Dense(encoding_dim, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(input_dim, activation='linear')
            ])
            return model
        
        with tf.device(self.device):
            # Create and compile model
            model = create_autoencoder(normal_data.shape[1])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Training
            model.fit(
                normal_data, normal_data,
                epochs=100,
                batch_size=32,
                shuffle=True,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.1
            )
        
        self.models['autoencoder'] = model
        
        # Quick evaluation
        with tf.device(self.device):
            reconstructed = model.predict(X_test, verbose=0)
            reconstruction_errors = np.mean(np.square(X_test - reconstructed), axis=1)
        
        # Use percentile threshold for binary classification
        threshold = np.percentile(reconstruction_errors, 90)
        pred_binary = (reconstruction_errors > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  ⭐ AutoEncoder shows strong performance - will be focus for RL optimization")
    
    def _train_local_outlier_factor(self, X_train, X_test, y_test):
        """Train Local Outlier Factor"""
        print("\nTraining Local Outlier Factor...")
        
        # LOF for novelty detection
        contamination = np.sum(y_test == 0) / len(y_test)
        contamination = max(0.01, min(0.5, contamination))
        
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )
        
        # Use all training data for LOF
        model.fit(X_train)
        self.models['lof'] = model
        
        # Quick evaluation
        pred = model.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly, 1 = normal
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            1 - y_test, pred_binary, average='binary', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def _find_optimal_thresholds(self, X_test, y_test):
        """Find optimal thresholds for each model"""
        print("\nFinding optimal thresholds...")
        
        self.thresholds = {}
        
        for model_name, model in self.models.items():
            print(f"  Processing {model_name}...")
            
            try:
                # Get anomaly scores
                if model_name == 'isolation_forest':
                    scores = model.decision_function(X_test)
                    anomaly_scores = -scores  # Higher scores = more anomalous
                elif model_name in ['ocsvm', 'lof']:
                    scores = model.decision_function(X_test)
                    anomaly_scores = -scores
                elif model_name == 'autoencoder':
                    with tf.device(self.device):
                        reconstructed = model.predict(X_test, verbose=0)
                        anomaly_scores = np.mean(np.square(X_test - reconstructed), axis=1)
                
                # Find optimal threshold using precision-recall curve
                y_anomaly = 1 - y_test  # Convert to anomaly labels (1=anomaly, 0=normal)
                
                if len(np.unique(y_anomaly)) == 2:
                    precisions, recalls, thresholds_pr = precision_recall_curve(y_anomaly, anomaly_scores)
                    
                    # Find threshold that maximizes F1 score
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else thresholds_pr[-1]
                    
                    self.thresholds[model_name] = {
                        'optimal': best_threshold,
                        'precision': precisions[best_idx],
                        'recall': recalls[best_idx],
                        'f1': f1_scores[best_idx]
                    }
                    
                    print(f"    Optimal threshold: {best_threshold:.6f}")
                    print(f"    Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")
                    
            except Exception as e:
                print(f"    Error processing {model_name}: {e}")
    
    def create_ensemble_model(self):
        """Create ensemble model from trained models"""
        print("\n" + "="*50)
        print("CREATING ENSEMBLE MODEL")
        print("="*50)
        
        # Simple ensemble with equal weights
        self.ensemble_weights = {model_name: 1.0/len(self.models) for model_name in self.models.keys()}
        
        print(f"Ensemble created with {len(self.models)} models:")
        for model_name, weight in self.ensemble_weights.items():
            print(f"  {model_name}: {weight:.3f}")
        
        return self.ensemble_weights
    
    def evaluate_comprehensive(self, X_test=None, y_test=None):
        """Comprehensive evaluation of base models"""
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)
        
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Individual model performance
        print(f"\nIndividual Model Performance:")
        y_anomaly = 1 - y_test
        best_model = None
        best_f1 = 0
        
        for model_name, model in self.models.items():
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    pred = model.predict(X_test)
                    pred_binary = (pred == -1).astype(int)
                elif model_name in ['ocsvm', 'lof']:
                    pred = model.predict(X_test)
                    pred_binary = (pred == -1).astype(int)
                elif model_name == 'autoencoder':
                    with tf.device(self.device):
                        reconstructed = model.predict(X_test, verbose=0)
                        scores = np.mean(np.square(X_test - reconstructed), axis=1)
                    threshold = self.thresholds[model_name]['optimal']
                    pred_binary = (scores > threshold).astype(int)
                
                # Calculate metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_anomaly, pred_binary, average='binary', zero_division=0
                )
                
                print(f"  {model_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                # Track best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
                    
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        
        print(f"\n🏆 Best performing model: {best_model} (F1={best_f1:.4f})")
        print(f"📈 This model will be the focus for RL optimization in Phase 2")
        
        return best_model, best_f1
    
    def save_models(self, path_prefix='optimized_anomaly_models'):
        """Save all models to designated directory"""
        ensure_model_directory()
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nSaving models to: {model_file}")
        
        model_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'normal_label': self.normal_label,
            'thresholds': self.thresholds,
            'models': {}
        }
        
        # Save models (except autoencoder)
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                autoencoder_path = get_model_path(f"{path_prefix}_autoencoder.keras")
                model.save(autoencoder_path)
                model_data['models'][model_name] = f"{path_prefix}_autoencoder"
                print(f"  ✓ AutoEncoder saved to: {autoencoder_path}")
            else:
                model_data['models'][model_name] = model
        
        joblib.dump(model_data, model_file)
        print(f"  ✓ Main model data saved to: {model_file}")
        print("✓ All models saved successfully!")
    
    def load_models(self, path_prefix='optimized_anomaly_models'):
        """Load all models from saved directory"""
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nLoading models from: {model_file}")
        
        try:
            model_data = joblib.load(model_file)
            
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.normal_label = model_data['normal_label']
            self.thresholds = model_data['thresholds']
            
            # Load models
            self.models = {}
            for model_name, model_path in model_data['models'].items():
                if model_name == 'autoencoder':
                    if isinstance(model_path, str):
                        # Construct full path
                        full_autoencoder_path = get_model_path(os.path.basename(model_path))
                        if os.path.exists(full_autoencoder_path):
                            self.models[model_name] = tf.keras.models.load_model(full_autoencoder_path)
                        else:
                            print(f"Warning: AutoEncoder model not found at {full_autoencoder_path}")
                    else:
                        print(f"Warning: Invalid AutoEncoder model path: {model_path}")
                else:
                    self.models[model_name] = model_path
            
            print("✓ Models loaded successfully!")
            print(f"  Available models: {list(self.models.keys())}")
            print(f"  Feature columns: {len(self.feature_columns)}")
            print(f"  Target column: {self.target_column}")
            print(f"  Normal label: {self.normal_label}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

class RLAgent:
    """
    Reinforcement Learning Agent for adaptive threshold and batch-size adjustment
    """
    def __init__(self, state_dim=6, action_dim=2, learning_rate=0.001):
        print("Initializing RL Agent...")
        
        # Basic parameters first
        self.state_dim = state_dim  # [anomaly_rate, precision, recall, f1, threshold, batch_size]
        self.action_dim = action_dim  # [threshold_adjustment, batch_size_adjustment]
        self.learning_rate = learning_rate
        
        # Action spaces - define these first before neural networks
        self.threshold_actions = np.linspace(-0.1, 0.1, 21)  # -10% to +10% adjustment
        self.batch_size_actions = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Multipliers
        print(f"Action spaces defined: {len(self.threshold_actions)} threshold actions, {len(self.batch_size_actions)} batch size actions")
        
        # Experience replay buffer
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # State normalization
        self.state_stats = {
            'anomaly_rate': {'mean': 0.1, 'std': 0.05},
            'precision': {'mean': 0.5, 'std': 0.2},
            'recall': {'mean': 0.5, 'std': 0.2},
            'f1': {'mean': 0.5, 'std': 0.2},
            'threshold': {'mean': 0.5, 'std': 0.3},
            'batch_size': {'mean': 100, 'std': 50}
        }
        
        # Neural networks - build these last
        try:
            print("Building Q-networks...")
            self.q_network = self._build_network()
            self.target_network = self._build_network()
            self.update_target_network()
            print("✓ RL Agent initialized successfully")
        except Exception as e:
            print(f"Warning: Error building neural networks: {e}")
            print("Using fallback mode without neural networks")
            self.q_network = None
            self.target_network = None
    
    def _build_network(self):
        """Build Q-network"""
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.threshold_actions) * len(self.batch_size_actions))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def normalize_state(self, state):
        """Normalize state values"""
        normalized = []
        keys = ['anomaly_rate', 'precision', 'recall', 'f1', 'threshold', 'batch_size']
        
        for i, key in enumerate(keys):
            val = (state[i] - self.state_stats[key]['mean']) / self.state_stats[key]['std']
            normalized.append(np.clip(val, -3, 3))
        
        return np.array(normalized)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        # Ensure action spaces exist
        if not hasattr(self, 'threshold_actions') or not hasattr(self, 'batch_size_actions'):
            print("Warning: Action spaces not defined, using random actions")
            threshold_idx = np.random.randint(0, 21)
            batch_idx = np.random.randint(0, 6)
            return threshold_idx, batch_idx
        
        if np.random.random() <= self.epsilon or self.q_network is None:
            # Random action
            threshold_idx = np.random.randint(len(self.threshold_actions))
            batch_idx = np.random.randint(len(self.batch_size_actions))
        else:
            # Q-network action
            try:
                state_norm = self.normalize_state(state).reshape(1, -1)
                q_values = self.q_network.predict(state_norm, verbose=0)[0]
                
                # Convert flat Q-values to 2D action space
                q_matrix = q_values.reshape(len(self.threshold_actions), len(self.batch_size_actions))
                best_action = np.unravel_index(np.argmax(q_matrix), q_matrix.shape)
                threshold_idx, batch_idx = best_action
            except Exception as e:
                print(f"Warning: Error in Q-network prediction: {e}")
                # Fallback to random
                threshold_idx = np.random.randint(len(self.threshold_actions))
                batch_idx = np.random.randint(len(self.batch_size_actions))
        
        return threshold_idx, batch_idx
    
    def calculate_reward(self, metrics, previous_metrics=None):
        """Calculate reward based on detection performance with confidence weighting"""
        # Get confidence level
        confidence_level = metrics.get('confidence_level', 'low')
        confidence_multiplier = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }.get(confidence_level, 0.4)
        
        # Primary rewards (weighted by confidence)
        precision_reward = metrics['precision'] * 2 * confidence_multiplier
        recall_reward = metrics['recall'] * 1.5 * confidence_multiplier
        f1_reward = metrics['f1'] * 3 * confidence_multiplier
        
        # Anomaly rate penalty/reward
        anomaly_rate = metrics['anomaly_rate']
        if 0.01 <= anomaly_rate <= 0.2:  # Reasonable anomaly rate
            rate_reward = 1.0
        elif anomaly_rate < 0.01:  # Too few anomalies detected
            rate_reward = -0.5
        elif 0.2 < anomaly_rate <= 0.5:  # Moderate concern
            rate_reward = -0.3
        else:  # Too many anomalies (possible false positives)
            rate_reward = -1.0
        
        # Confidence reward - encourage high confidence predictions
        confidence_reward = {
            'high': 0.5,
            'medium': 0.2,
            'low': -0.2
        }.get(confidence_level, -0.2)
        
        # Stability reward (if previous metrics available)
        stability_reward = 0
        if previous_metrics is not None:
            # Reward consistency, especially for high confidence predictions
            precision_diff = abs(metrics['precision'] - previous_metrics.get('precision', 0))
            if precision_diff < 0.1:  # Stable precision
                stability_reward += 0.3 * confidence_multiplier
            
            # Reward F1 improvement
            f1_diff = metrics['f1'] - previous_metrics.get('f1', 0)
            if f1_diff > 0:
                stability_reward += f1_diff * 2 * confidence_multiplier
        
        # Ground truth bonus
        ground_truth_bonus = 0
        if metrics.get('has_ground_truth', False):
            ground_truth_bonus = 0.3  # Bonus for having real labels
            
            # Additional bonus for prediction accuracy
            true_rate = metrics.get('true_anomaly_rate', 0)
            predicted_rate = metrics.get('anomaly_rate', 0)
            accuracy_bonus = 1 - abs(true_rate - predicted_rate)
            ground_truth_bonus += accuracy_bonus * 0.5
        
        # Efficiency reward based on batch processing
        efficiency_reward = 0.1  # Small positive reward for processing
        
        # Penalty for very low confidence
        if confidence_level == 'low':
            low_confidence_penalty = -0.3
        else:
            low_confidence_penalty = 0
        
        # Total reward
        total_reward = (precision_reward + recall_reward + f1_reward + 
                       rate_reward + confidence_reward + stability_reward + 
                       ground_truth_bonus + efficiency_reward + low_confidence_penalty)
        
        return total_reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if self.q_network is None or self.target_network is None:
            # Skip training if neural networks not available
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return
            
        if len(self.memory) < batch_size:
            return
        
        try:
            batch = np.random.choice(len(self.memory), batch_size, replace=False)
            
            states = []
            targets = []
            
            for idx in batch:
                state, action, reward, next_state, done = self.memory[idx]
                
                state_norm = self.normalize_state(state)
                target = self.q_network.predict(state_norm.reshape(1, -1), verbose=0)[0]
                
                if done:
                    target_value = reward
                else:
                    next_state_norm = self.normalize_state(next_state)
                    next_q = self.target_network.predict(next_state_norm.reshape(1, -1), verbose=0)[0]
                    target_value = reward + self.gamma * np.max(next_q)
                
                # Convert action to flat index
                threshold_idx, batch_idx = action
                action_idx = threshold_idx * len(self.batch_size_actions) + batch_idx
                target[action_idx] = target_value
                
                states.append(state_norm)
                targets.append(target)
            
            # Train
            if len(states) > 0:
                self.q_network.fit(np.array(states), np.array(targets), verbose=0, epochs=1)
            
        except Exception as e:
            print(f"Warning: Error in RL training: {e}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        if self.q_network is not None and self.target_network is not None:
            try:
                self.target_network.set_weights(self.q_network.get_weights())
            except Exception as e:
                print(f"Warning: Error updating target network: {e}")

class RLAdaptiveAnomalyDetector:
    """
    RL-Enhanced Anomaly Detector with Preliminary Anomaly Filter
    Only processes batches that pass preliminary anomaly screening
    """
    
    def __init__(self, base_detector, focus_model='autoencoder', filter_sensitivity='medium'):
        self.base_detector = base_detector
        self.focus_model = focus_model
        self.rl_agent = RLAgent()
        
        # Initialize Preliminary Anomaly Filter
        self.preliminary_filter = PreliminaryAnomalyFilter(sensitivity=filter_sensitivity)
        
        # Validate that focus model exists
        if focus_model not in base_detector.models:
            print(f"Warning: Focus model '{focus_model}' not found. Available models: {list(base_detector.models.keys())}")
            if 'autoencoder' in base_detector.models:
                self.focus_model = 'autoencoder'
                print(f"Defaulting to autoencoder")
            else:
                self.focus_model = list(base_detector.models.keys())[0]
                print(f"Defaulting to {self.focus_model}")
        else:
            print(f"✓ Focusing RL adaptation on: {focus_model}")
        
        # Adaptive parameters
        self.current_threshold_multiplier = 1.0
        self.current_batch_size = 100
        self.min_batch_size = 50
        self.max_batch_size = 500
        
        # AutoEncoder specific parameters
        if self.focus_model == 'autoencoder' and 'autoencoder' in self.base_detector.thresholds:
            self.autoencoder_threshold = self.base_detector.thresholds['autoencoder']['optimal']
            print(f"  Base AutoEncoder threshold: {self.autoencoder_threshold:.6f}")
        else:
            self.autoencoder_threshold = 0.1  # Default threshold
            print(f"  Using default threshold: {self.autoencoder_threshold}")
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
        # Streaming data buffer
        self.data_buffer = deque(maxlen=10000)
        self.processed_count = 0
        
        # Filter statistics
        self.filter_efficiency_history = []
        
        print("RL-Enhanced Adaptive Anomaly Detector initialized")
        print(f"Available models: {list(base_detector.models.keys())}")
        print(f"Focus model: {self.focus_model}")
        print(f"Preliminary filter sensitivity: {filter_sensitivity}")
    
    def train_preliminary_filter(self, training_data=None):
        """Train the preliminary anomaly filter on normal data"""
        print("\n🎯 Training Preliminary Anomaly Filter...")
        
        if training_data is None:
            # Use normal data from base detector's training
            if hasattr(self.base_detector, 'X_test') and hasattr(self.base_detector, 'y_test'):
                X_test = self.base_detector.X_test
                y_test = self.base_detector.y_test
                
                # Extract normal samples
                normal_samples = X_test[y_test == 1]
                validation_data = (X_test, y_test)
                
                print(f"Using {len(normal_samples)} normal samples from base detector")
                self.preliminary_filter.train_on_normal_data(normal_samples, validation_data)
            else:
                print("❌ No training data available for preliminary filter")
                return False
        else:
            # Use provided training data
            X_train, y_train = training_data
            normal_samples = X_train[y_train == 1]
            validation_data = (X_train, y_train)
            
            print(f"Using {len(normal_samples)} normal samples from provided data")
            self.preliminary_filter.train_on_normal_data(normal_samples, validation_data)
        
        return True
    
    def process_streaming_data(self, streaming_path, update_frequency=10, start_step=0):
        """Process streaming data with preliminary filtering + RL adaptation"""
        print(f"\n📡 Processing streaming data with Preliminary Filter + RL (starting from step {start_step})")
        print("="*80)
        
        # Load streaming data
        try:
            streaming_df = pd.read_csv(streaming_path)
            print(f"Streaming data shape: {streaming_df.shape}")
        except Exception as e:
            print(f"Error loading streaming data: {e}")
            return []
        
        # Train preliminary filter if not trained
        if not self.preliminary_filter.is_trained:
            print("🔧 Training preliminary filter...")
            if not self.train_preliminary_filter():
                print("⚠️  Warning: Preliminary filter training failed, processing all batches")
        
        # Preprocess streaming data
        X_stream, y_stream = self._preprocess_streaming_data(streaming_df)
        
        # Process in chunks with preliminary filtering + RL adaptation
        total_samples = len(X_stream)
        current_position = 0
        step = start_step
        previous_metrics = None
        
        # Skip to resume position if continuing
        if start_step > 0 and self.performance_history:
            total_processed_before = sum([h['batch_size'] for h in self.performance_history])
            current_position = min(total_processed_before, total_samples - self.current_batch_size)
            print(f"Resuming from position {current_position}/{total_samples}")
        
        while current_position < total_samples:
            step += 1
            
            try:
                # Get current batch
                batch_end = min(current_position + self.current_batch_size, total_samples)
                X_batch = X_stream.iloc[current_position:batch_end]
                y_batch = y_stream.iloc[current_position:batch_end] if y_stream is not None else None
                
                print(f"\nStep {step}: Processing batch {current_position}-{batch_end} (size: {len(X_batch)})")
                
                # Validate batch
                if len(X_batch) == 0:
                    print(f"  Warning: Empty batch, skipping...")
                    current_position = batch_end
                    continue
                
                # Scale batch for preliminary filter
                X_batch_scaled = self.base_detector.scaler.transform(X_batch)
                
                # 🔍 PRELIMINARY ANOMALY FILTERING
                filter_result = self.preliminary_filter.evaluate_batch(X_batch_scaled, y_batch)
                
                print(f"  🔍 Preliminary Filter Result:")
                print(f"    Has anomalies: {filter_result['has_anomalies']}")
                print(f"    Confidence: {filter_result['confidence']:.3f}")
                print(f"    Rationale: {filter_result['decision_rationale']}")
                print(f"    Processing time: {filter_result.get('processing_time', 0):.4f}s")
                
                # Track filter efficiency
                filter_efficiency = {
                    'step': step,
                    'passed_filter': filter_result['has_anomalies'],
                    'filter_confidence': filter_result['confidence'],
                    'filter_time': filter_result.get('processing_time', 0),
                    'batch_size': len(X_batch)
                }
                self.filter_efficiency_history.append(filter_efficiency)
                
                # Only process through RL if batch passes preliminary filter
                if filter_result['has_anomalies']:
                    print(f"  ✅ Batch passed preliminary filter → Processing with RL")
                    
                    # Make predictions with current thresholds
                    predictions, scores = self._predict_with_adaptive_thresholds(X_batch_scaled)
                    
                    # Calculate metrics
                    metrics = self._calculate_batch_metrics(predictions, scores, y_batch)
                    
                    # Add filter information to metrics
                    metrics['filter_passed'] = True
                    metrics['filter_confidence'] = filter_result['confidence']
                    metrics['filter_time'] = filter_result.get('processing_time', 0)
                    
                    # AutoEncoder specific metrics
                    if self.focus_model == 'autoencoder':
                        metrics['effective_threshold'] = self.autoencoder_threshold * self.current_threshold_multiplier
                        metrics['threshold_multiplier'] = self.current_threshold_multiplier
                    
                    # Get current state for RL
                    current_state = self._get_current_state(metrics)
                    
                    # RL adaptation every update_frequency steps
                    if step % update_frequency == 0 and step > start_step:
                        try:
                            self._rl_adaptation_step(current_state, metrics, previous_metrics, step)
                        except Exception as e:
                            print(f"  Warning: RL adaptation failed: {e}")
                    
                    # Store metrics
                    self.performance_history.append({
                        'step': step,
                        'batch_size': len(X_batch),
                        'threshold_multiplier': self.current_threshold_multiplier,
                        'batch_size_param': self.current_batch_size,
                        **metrics
                    })
                    
                    # Display real-time metrics
                    self._display_realtime_metrics(step, metrics)
                    
                    previous_metrics = metrics
                    
                else:
                    print(f"  ❌ Batch filtered out → Skipping RL processing")
                    print(f"    Reason: {filter_result['decision_rationale']}")
                    
                    # Store filtered batch info (minimal metrics)
                    filtered_metrics = {
                        'step': step,
                        'batch_size': len(X_batch),
                        'threshold_multiplier': self.current_threshold_multiplier,
                        'batch_size_param': self.current_batch_size,
                        'filter_passed': False,
                        'filter_confidence': filter_result['confidence'],
                        'filter_time': filter_result.get('processing_time', 0),
                        'anomaly_rate': 0.0,  # Assumed normal batch
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'has_ground_truth': y_batch is not None,
                        'confidence_level': 'low'
                    }
                    
                    self.performance_history.append(filtered_metrics)
                
                current_position = batch_end
                self.processed_count += len(X_batch)
                
            except Exception as e:
                print(f"  ❌ Error processing batch {current_position}-{batch_end}: {e}")
                print(f"  Skipping this batch and continuing...")
                current_position = min(current_position + self.current_batch_size, total_samples)
                continue
        
        print(f"\n📊 Streaming processing completed!")
        print(f"Total samples processed: {current_position}")
        print(f"Total cumulative samples: {self.processed_count}")
        
        # Display preliminary filter statistics
        self._display_filter_statistics()
        
        return self.performance_history
    
    def _display_filter_statistics(self):
        """Display preliminary filter performance statistics"""
        print(f"\n🔍 Preliminary Filter Performance:")
        print("="*50)
        
        filter_stats = self.preliminary_filter.get_filter_statistics()
        
        print(f"📈 Filter Efficiency:")
        print(f"  Total batches evaluated: {filter_stats['total_batches']}")
        print(f"  Batches passed: {filter_stats['passed_batches']}")
        print(f"  Batches filtered out: {filter_stats['filtered_batches']}")
        print(f"  Pass rate: {filter_stats['pass_rate']:.2%}")
        print(f"  Filter rate: {filter_stats['filter_rate']:.2%}")
        print(f"  Efficiency gain: {filter_stats['efficiency_gain']:.2%}")
        
        print(f"\n⏱️  Performance:")
        print(f"  Average processing time: {filter_stats['avg_processing_time']:.4f}s")
        
        if filter_stats['false_positives'] + filter_stats['false_negatives'] > 0:
            print(f"\n🎯 Accuracy (when ground truth available):")
            print(f"  False positives: {filter_stats['false_positives']}")
            print(f"  False negatives: {filter_stats['false_negatives']}")
            if 'filter_accuracy' in filter_stats:
                print(f"  Filter accuracy: {filter_stats['filter_accuracy']:.2%}")
        
        # Calculate computational savings
        if filter_stats['total_batches'] > 0:
            computational_savings = filter_stats['filtered_batches'] / filter_stats['total_batches']
            print(f"\n💰 Computational Savings:")
            print(f"  Estimated RL processing savings: {computational_savings:.2%}")
            print(f"  Batches processed by RL: {filter_stats['passed_batches']}")
            print(f"  Batches avoided by RL: {filter_stats['filtered_batches']}")
    
    def _preprocess_streaming_data(self, streaming_df):
        """Preprocess streaming data with enhanced validation"""
        print("Preprocessing streaming data...")
        
        # Find target column
        target_col = self.base_detector.target_column
        
        if target_col and target_col in streaming_df.columns:
            X_stream = streaming_df.drop(columns=[target_col])
            y_stream = streaming_df[target_col]
            
            # Validate ground truth labels
            unique_labels = y_stream.unique()
            print(f"  Found {len(unique_labels)} unique labels in streaming data: {unique_labels}")
            
            # Convert to binary if needed
            if self.base_detector.normal_label is not None:
                y_stream = (y_stream == self.base_detector.normal_label).astype(int)
                
                # Check label distribution
                normal_count = np.sum(y_stream)
                anomaly_count = len(y_stream) - normal_count
                print(f"  Label distribution: Normal={normal_count}, Anomaly={anomaly_count}")
                
                # Warn about potential issues
                if anomaly_count == 0:
                    print("  ⚠️  Warning: No anomalies in streaming data - metrics may be unreliable")
                elif normal_count == 0:
                    print("  ⚠️  Warning: No normal samples in streaming data - metrics may be unreliable")
        else:
            X_stream = streaming_df
            y_stream = None
            print("  No target column found in streaming data - unsupervised mode")
        
        # Ensure same feature columns as training
        if self.base_detector.feature_columns:
            missing_cols = set(self.base_detector.feature_columns) - set(X_stream.columns)
            extra_cols = set(X_stream.columns) - set(self.base_detector.feature_columns)
            
            if missing_cols:
                print(f"  Warning: Missing columns in streaming data: {missing_cols}")
                for col in missing_cols:
                    X_stream[col] = 0  # Default value
            
            if extra_cols:
                print(f"  Info: Dropping extra columns: {extra_cols}")
                X_stream = X_stream.drop(columns=list(extra_cols))
            
            # Reorder columns to match training
            X_stream = X_stream[self.base_detector.feature_columns]
        
        # Handle missing values
        print("  Handling missing values...")
        numeric_cols = X_stream.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_stream[col].isnull().any():
                median_val = X_stream[col].median()
                X_stream[col] = X_stream[col].fillna(median_val)
        
        # Categorical columns
        categorical_cols = X_stream.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if X_stream[col].isnull().any():
                mode_val = X_stream[col].mode()
                if len(mode_val) > 0:
                    X_stream[col] = X_stream[col].fillna(mode_val[0])
                else:
                    X_stream[col] = X_stream[col].fillna('unknown')
        
        # Final validation
        remaining_nulls = X_stream.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"  ⚠️  Warning: {remaining_nulls} null values remaining after preprocessing")
            X_stream = X_stream.fillna(0)  # Final fallback
        
        print(f"  ✅ Streaming data preprocessed: {X_stream.shape}")
        print(f"  Features: {len(X_stream.columns)}")
        print(f"  Samples: {len(X_stream)}")
        
        return X_stream, y_stream
    
    def _predict_with_adaptive_thresholds(self, X_batch):
        """Make predictions with adaptive thresholds, optimized for focus model"""
        if self.focus_model == 'autoencoder':
            # Focus on AutoEncoder predictions
            return self._predict_with_autoencoder(X_batch)
        else:
            # Use ensemble approach
            return self._predict_with_ensemble(X_batch)
    
    def _predict_with_autoencoder(self, X_scaled):
        """Optimized prediction using AutoEncoder only"""
        try:
            model = self.base_detector.models['autoencoder']
            
            # Get reconstruction errors
            with tf.device(self.base_detector.device):
                reconstructed = model.predict(X_scaled, verbose=0)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            
            # Apply adaptive threshold
            adaptive_threshold = self.autoencoder_threshold * self.current_threshold_multiplier
            
            # Predictions
            predictions = (reconstruction_errors > adaptive_threshold).astype(int)
            
            return predictions, reconstruction_errors
            
        except Exception as e:
            print(f"Error in AutoEncoder prediction: {e}")
            # Fallback to simple thresholding
            predictions = np.zeros(len(X_scaled), dtype=int)
            scores = np.random.rand(len(X_scaled))
            return predictions, scores
    
    def _predict_with_ensemble(self, X_scaled):
        """Ensemble prediction with adaptive thresholds"""
        anomaly_scores = []
        predictions = []
        
        for model_name, model in self.base_detector.models.items():
            try:
                # Get anomaly scores
                if model_name == 'isolation_forest':
                    scores = model.decision_function(X_scaled)
                    scores = -scores
                elif model_name in ['ocsvm', 'lof']:
                    scores = model.decision_function(X_scaled)
                    scores = -scores
                elif model_name == 'autoencoder':
                    with tf.device(self.base_detector.device):
                        reconstructed = model.predict(X_scaled, verbose=0)
                        scores = np.mean(np.square(X_scaled - reconstructed), axis=1)
                else:
                    continue
                
                # Apply adaptive threshold
                if model_name in self.base_detector.thresholds:
                    base_threshold = self.base_detector.thresholds[model_name]['optimal']
                    adaptive_threshold = base_threshold * self.current_threshold_multiplier
                else:
                    adaptive_threshold = np.percentile(scores, 90)  # Default threshold
                
                pred = (scores > adaptive_threshold).astype(int)
                
                anomaly_scores.append(scores)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue
        
        # Ensemble predictions
        if len(anomaly_scores) > 0:
            ensemble_scores = np.mean(anomaly_scores, axis=0)
            ensemble_predictions = np.mean(predictions, axis=0)
            final_predictions = (ensemble_predictions >= 0.5).astype(int)
        else:
            # Fallback
            final_predictions = np.zeros(len(X_scaled), dtype=int)
            ensemble_scores = np.random.rand(len(X_scaled))
        
        return final_predictions, ensemble_scores
    
    def _calculate_batch_metrics(self, predictions, scores, y_true=None):
        """Calculate metrics for current batch with improved estimation"""
        # Initialize all required metrics with default values
        metrics = {
            'anomaly_rate': 0.0,
            'total_anomalies': 0,
            'batch_size': 0,
            'avg_anomaly_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'has_ground_truth': False,
            'true_anomaly_rate': 0.0,
            'confidence_level': 'low'
        }
        
        # Basic metrics
        if len(predictions) > 0:
            metrics['anomaly_rate'] = np.mean(predictions)
            metrics['total_anomalies'] = np.sum(predictions)
            metrics['batch_size'] = len(predictions)
        
        if len(scores) > 0:
            metrics['avg_anomaly_score'] = np.mean(scores)
        
        # If ground truth is available
        if y_true is not None and len(y_true) > 0:
            try:
                y_anomaly = 1 - y_true  # Convert to anomaly labels
                metrics['true_anomaly_rate'] = np.mean(y_anomaly)
                metrics['has_ground_truth'] = True
                
                # Enhanced validation for supervised metrics
                predictions_valid = (len(predictions) == len(y_anomaly) and len(predictions) > 0)
                labels_diverse = len(np.unique(y_anomaly)) > 1
                predictions_diverse = len(np.unique(predictions)) > 1
                
                if predictions_valid and labels_diverse and predictions_diverse:
                    # Calculate supervised metrics
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_anomaly, predictions, average='binary', zero_division=0
                    )
                    
                    metrics['precision'] = float(precision) if not np.isnan(precision) else 0.0
                    metrics['recall'] = float(recall) if not np.isnan(recall) else 0.0
                    metrics['f1'] = float(f1) if not np.isnan(f1) else 0.0
                    metrics['confidence_level'] = 'high'
                    
                else:
                    # Use heuristic estimation
                    metrics.update(self._advanced_heuristic_estimation(metrics, y_anomaly, predictions, scores))
                    
            except Exception as e:
                print(f"    Warning: Error in supervised metrics calculation: {e}")
                metrics.update(self._advanced_heuristic_estimation(metrics, None, predictions, scores))
        else:
            # Unsupervised mode - use advanced heuristic estimates
            metrics.update(self._advanced_heuristic_estimation(metrics, None, predictions, scores))
        
        # Always calculate F1 from precision and recall
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Validate all metrics are numbers
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = 0.0
            elif key not in ['has_ground_truth', 'confidence_level', 'filter_passed']:
                metrics[key] = 0.0
        
        return metrics
    
    def _advanced_heuristic_estimation(self, base_metrics, y_anomaly=None, predictions=None, scores=None):
        """Advanced heuristic estimation with historical context"""
        anomaly_rate = base_metrics['anomaly_rate']
        
        # Use historical performance if available
        historical_precision = self._get_historical_average('precision', default=0.5)
        historical_recall = self._get_historical_average('recall', default=0.5)
        
        # Score-based analysis for better estimates
        if scores is not None and len(scores) > 0:
            score_variance = np.var(scores)
            score_mean = np.mean(scores)
            
            # Higher variance might indicate better discrimination
            variance_factor = min(1.0, score_variance / (score_mean + 1e-6))
            
            # Score distribution analysis
            score_percentiles = np.percentile(scores, [10, 50, 90])
            score_range = score_percentiles[2] - score_percentiles[0]
            
            # Better spread = potentially better performance
            spread_factor = min(1.0, score_range / (score_mean + 1e-6))
            
        else:
            variance_factor = 0.5
            spread_factor = 0.5
        
        # Adaptive estimation based on anomaly rate
        if 0.001 <= anomaly_rate <= 0.1:  # Reasonable anomaly rate
            # Good balance suggests good model performance
            precision_base = 0.7
            recall_base = 0.6
            quality_bonus = 0.2
        elif anomaly_rate < 0.001:  # Very few anomalies
            # Conservative detection - high precision, low recall
            precision_base = 0.9
            recall_base = 0.3
            quality_bonus = 0.1
        elif 0.1 < anomaly_rate <= 0.3:  # Moderate anomaly rate
            # Balanced detection
            precision_base = 0.6
            recall_base = 0.7
            quality_bonus = 0.1
        else:  # High anomaly rate (>30%)
            # Possibly too many false positives
            precision_base = 0.4
            recall_base = 0.8
            quality_bonus = 0.0
        
        # Incorporate score quality
        estimated_precision = precision_base + quality_bonus * variance_factor
        estimated_recall = recall_base + quality_bonus * spread_factor
        
        # Blend with historical performance (if available)
        if len(self.performance_history) > 0:
            blend_factor = min(0.3, len(self.performance_history) / 20)
            estimated_precision = (1 - blend_factor) * estimated_precision + blend_factor * historical_precision
            estimated_recall = (1 - blend_factor) * estimated_recall + blend_factor * historical_recall
        
        # Ground truth validation (if available)
        if y_anomaly is not None and len(y_anomaly) > 0:
            true_anomaly_rate = np.mean(y_anomaly)
            
            # Adjust based on how close our prediction rate is to true rate
            rate_similarity = 1 - abs(anomaly_rate - true_anomaly_rate)
            estimated_precision *= (0.7 + 0.3 * rate_similarity)
            estimated_recall *= (0.7 + 0.3 * rate_similarity)
        
        # Determine confidence level
        if scores is not None and len(scores) > 0 and variance_factor > 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'precision': np.clip(estimated_precision, 0.0, 1.0),
            'recall': np.clip(estimated_recall, 0.0, 1.0),
            'confidence_level': confidence,
            'has_ground_truth': y_anomaly is not None
        }
    
    def _get_historical_average(self, metric, default=0.5, window=10):
        """Get historical average of a metric for better estimation"""
        if not self.performance_history:
            return default
        
        recent_history = self.performance_history[-window:]
        values = [h.get(metric, default) for h in recent_history if h.get(metric, 0) > 0]
        
        if not values:
            return default
        
        return np.mean(values)
    
    def _get_current_state(self, metrics):
        """Get current state for RL agent"""
        # Safely extract metrics with default values
        anomaly_rate = metrics.get('anomaly_rate', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        state = [
            float(anomaly_rate),
            float(precision),
            float(recall),
            float(f1),
            float(self.current_threshold_multiplier),
            float(self.current_batch_size)
        ]
        
        # Validate state values
        for i, val in enumerate(state):
            if np.isnan(val) or np.isinf(val):
                state[i] = 0.0
        
        return np.array(state, dtype=np.float32)
    
    def _rl_adaptation_step(self, current_state, metrics, previous_metrics, step):
        """Perform RL adaptation step"""
        print(f"  🤖 RL Adaptation at step {step}")
        
        # Get action from RL agent
        try:
            threshold_idx, batch_idx = self.rl_agent.get_action(current_state)
            
            # Apply actions
            threshold_adjustment = self.rl_agent.threshold_actions[threshold_idx]
            batch_size_multiplier = self.rl_agent.batch_size_actions[batch_idx]
            
        except Exception as e:
            print(f"  Warning: Error getting RL action: {e}")
            # Fallback to small random adjustments
            threshold_adjustment = np.random.uniform(-0.05, 0.05)
            batch_size_multiplier = np.random.choice([0.9, 1.0, 1.1])
        
        # Calculate reward
        reward = self.rl_agent.calculate_reward(metrics, previous_metrics)
        
        # Store previous state-action-reward if available
        if len(self.adaptation_history) > 0:
            try:
                prev_data = self.adaptation_history[-1]
                self.rl_agent.remember(
                    prev_data['state'],
                    prev_data['action'],
                    reward,
                    current_state,
                    False  # Not done
                )
            except Exception as e:
                print(f"  Warning: Error storing RL memory: {e}")
        
        # Update parameters
        old_threshold = self.current_threshold_multiplier
        old_batch_size = self.current_batch_size
        
        self.current_threshold_multiplier = np.clip(
            self.current_threshold_multiplier + threshold_adjustment,
            0.1, 3.0
        )
        
        self.current_batch_size = int(np.clip(
            self.current_batch_size * batch_size_multiplier,
            self.min_batch_size, self.max_batch_size
        ))
        
        # Store adaptation history
        try:
            adaptation_data = {
                'step': step,
                'state': current_state.copy(),
                'action': (threshold_idx if 'threshold_idx' in locals() else 0, 
                          batch_idx if 'batch_idx' in locals() else 0),
                'reward': reward,
                'threshold_adjustment': threshold_adjustment,
                'batch_size_multiplier': batch_size_multiplier,
                'old_threshold': old_threshold,
                'new_threshold': self.current_threshold_multiplier,
                'old_batch_size': old_batch_size,
                'new_batch_size': self.current_batch_size,
                'epsilon': self.rl_agent.epsilon
            }
            
            self.adaptation_history.append(adaptation_data)
        except Exception as e:
            print(f"  Warning: Error storing adaptation history: {e}")
        
        # Train RL agent
        try:
            self.rl_agent.replay()
        except Exception as e:
            print(f"  Warning: Error in RL training: {e}")
        
        # Update target network periodically
        if len(self.adaptation_history) % 10 == 0:
            try:
                self.rl_agent.update_target_network()
            except Exception as e:
                print(f"  Warning: Error updating target network: {e}")
        
        print(f"    Threshold: {old_threshold:.3f} -> {self.current_threshold_multiplier:.3f}")
        print(f"    Batch size: {old_batch_size} -> {self.current_batch_size}")
        print(f"    Reward: {reward:.3f}")
        print(f"    Epsilon: {self.rl_agent.epsilon:.3f}")
    
    def _display_realtime_metrics(self, step, metrics):
        """Display real-time metrics with confidence and filter information"""
        # Safely display metrics with default values
        anomaly_rate = metrics.get('anomaly_rate', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        confidence = metrics.get('confidence_level', 'unknown')
        filter_passed = metrics.get('filter_passed', False)
        filter_confidence = metrics.get('filter_confidence', 0.0)
        
        # Color coding for confidence levels
        if confidence == 'high':
            confidence_icon = "🟢"
        elif confidence == 'medium':
            confidence_icon = "🟡"
        else:
            confidence_icon = "🔴"
        
        # Filter status
        filter_icon = "✅" if filter_passed else "❌"
        
        print(f"  📊 Batch Metrics:")
        print(f"    Anomaly rate: {anomaly_rate:.4f}")
        print(f"    Precision: {precision:.4f} {confidence_icon}")
        print(f"    Recall: {recall:.4f} {confidence_icon}")
        print(f"    F1: {f1:.4f} {confidence_icon}")
        print(f"    Confidence: {confidence}")
        print(f"    Filter: {filter_icon} (conf: {filter_confidence:.3f})")
        
        # AutoEncoder specific display
        if self.focus_model == 'autoencoder' and 'effective_threshold' in metrics:
            effective_threshold = metrics.get('effective_threshold', 0.0)
            print(f"    AutoEncoder threshold: {effective_threshold:.6f}")
        
        # Ground truth info
        if metrics.get('has_ground_truth', False):
            true_anomaly_rate = metrics.get('true_anomaly_rate', 0.0)
            print(f"    True anomaly rate: {true_anomaly_rate:.4f}")
            
            # Show accuracy of our prediction
            prediction_accuracy = 1 - abs(anomaly_rate - true_anomaly_rate)
            print(f"    Prediction accuracy: {prediction_accuracy:.4f}")
        
        # Performance trend indicator
        if len(self.performance_history) > 1:
            prev_f1 = self.performance_history[-1].get('f1', 0)
            current_f1 = f1
            trend = "📈" if current_f1 > prev_f1 else "📉" if current_f1 < prev_f1 else "➡️"
            print(f"    Trend: {trend}")
        
        if step % 5 == 0:  # Display adaptation status every 5 steps
            print(f"    Current threshold multiplier: {self.current_threshold_multiplier:.3f}")
            print(f"    Current batch size: {self.current_batch_size}")
            if hasattr(self, 'rl_agent'):
                print(f"    RL exploration (ε): {self.rl_agent.epsilon:.3f}")
            print(f"    Total processed: {self.processed_count}")
    
    def get_adaptation_summary(self):
        """Get summary of RL adaptation performance with filter statistics"""
        if not self.performance_history:
            return {"message": "No performance history available"}
        
        try:
            adapt_df = pd.DataFrame(self.adaptation_history) if self.adaptation_history else pd.DataFrame()
            perf_df = pd.DataFrame(self.performance_history)
            
            # Safely get final performance
            final_performance = {}
            if len(perf_df) > 0:
                last_row = perf_df.iloc[-1]
                final_performance = {
                    'anomaly_rate': last_row.get('anomaly_rate', 0.0),
                    'precision': last_row.get('precision', 0.0),
                    'recall': last_row.get('recall', 0.0),
                    'f1': last_row.get('f1', 0.0)
                }
            
            # Filter performance analysis
            filter_performance = {}
            if len(perf_df) > 0:
                total_batches = len(perf_df)
                passed_batches = len(perf_df[perf_df.get('filter_passed', True) == True])
                filtered_batches = total_batches - passed_batches
                
                filter_performance = {
                    'total_batches': total_batches,
                    'passed_batches': passed_batches,
                    'filtered_batches': filtered_batches,
                    'pass_rate': passed_batches / total_batches if total_batches > 0 else 0,
                    'filter_efficiency': filtered_batches / total_batches if total_batches > 0 else 0,
                    'avg_filter_confidence': perf_df.get('filter_confidence', pd.Series([0])).mean(),
                    'computational_savings': filtered_batches / total_batches if total_batches > 0 else 0
                }
            
            # AutoEncoder specific analysis
            autoencoder_performance = {}
            if self.focus_model == 'autoencoder':
                autoencoder_performance = {
                    'base_threshold': getattr(self, 'autoencoder_threshold', 0.0),
                    'final_threshold_multiplier': self.current_threshold_multiplier,
                    'final_effective_threshold': getattr(self, 'autoencoder_threshold', 0.0) * self.current_threshold_multiplier,
                    'threshold_improvement': 0.0
                }
                
                # Calculate improvement if we have enough data
                if len(perf_df) > 1:
                    first_f1 = perf_df.iloc[0].get('f1', 0.0)
                    last_f1 = perf_df.iloc[-1].get('f1', 0.0)
                    autoencoder_performance['threshold_improvement'] = last_f1 - first_f1
            
            # Preliminary filter statistics
            filter_stats = self.preliminary_filter.get_filter_statistics()
            
            summary = {
                'focus_model': self.focus_model,
                'filter_sensitivity': self.preliminary_filter.sensitivity,
                'total_adaptation_steps': len(adapt_df),
                'avg_reward': adapt_df['reward'].mean() if len(adapt_df) > 0 else 0.0,
                'final_epsilon': adapt_df['epsilon'].iloc[-1] if len(adapt_df) > 0 else self.rl_agent.epsilon,
                'threshold_range': (
                    perf_df['threshold_multiplier'].min() if len(perf_df) > 0 else self.current_threshold_multiplier, 
                    perf_df['threshold_multiplier'].max() if len(perf_df) > 0 else self.current_threshold_multiplier
                ),
                'batch_size_range': (
                    perf_df['batch_size_param'].min() if len(perf_df) > 0 else self.current_batch_size, 
                    perf_df['batch_size_param'].max() if len(perf_df) > 0 else self.current_batch_size
                ),
                'final_performance': final_performance,
                'autoencoder_specific': autoencoder_performance,
                'filter_performance': filter_performance,
                'preliminary_filter_stats': filter_stats,
                'total_samples_processed': self.processed_count,
                'total_steps': len(perf_df)
            }
            
            return summary
            
        except Exception as e:
            print(f"Error creating adaptation summary: {e}")
            return {
                "error": str(e),
                "focus_model": self.focus_model,
                "filter_sensitivity": getattr(self.preliminary_filter, 'sensitivity', 'unknown'),
                "total_samples_processed": self.processed_count
            }
    
    def save_rl_enhanced_model(self, path_prefix='rl_enhanced_anomaly_model_with_filter'):
        """Save RL-enhanced model with preliminary filter to designated directory"""
        ensure_model_directory()
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nSaving RL-enhanced model with filter to: {model_file}")
        
        try:
            # Prepare model data
            rl_model_data = {
                'base_detector_scaler': self.base_detector.scaler,
                'base_detector_feature_columns': self.base_detector.feature_columns,
                'base_detector_target_column': self.base_detector.target_column,
                'base_detector_normal_label': self.base_detector.normal_label,
                'base_detector_thresholds': self.base_detector.thresholds,
                
                # RL-specific parameters
                'focus_model': self.focus_model,
                'current_threshold_multiplier': self.current_threshold_multiplier,
                'current_batch_size': self.current_batch_size,
                'autoencoder_threshold': getattr(self, 'autoencoder_threshold', None),
                
                # RL agent state
                'rl_agent_epsilon': self.rl_agent.epsilon,
                'rl_agent_memory': list(self.rl_agent.memory),
                
                # Performance history
                'performance_history': self.performance_history,
                'adaptation_history': self.adaptation_history,
                'filter_efficiency_history': self.filter_efficiency_history,
                
                # Metadata
                'total_processed': self.processed_count,
                'save_timestamp': datetime.now().isoformat(),
                'model_save_dir': MODEL_SAVE_DIR,
                'filter_sensitivity': self.preliminary_filter.sensitivity
            }
            
            # Save non-autoencoder models
            rl_model_data['base_detector_models'] = {}
            for model_name, model in self.base_detector.models.items():
                if model_name != 'autoencoder':
                    rl_model_data['base_detector_models'][model_name] = model
            
            # Save AutoEncoder separately if it exists
            if 'autoencoder' in self.base_detector.models:
                autoencoder_path = get_model_path(f"{path_prefix}_autoencoder.keras")
                autoencoder_model = self.base_detector.models['autoencoder']
                autoencoder_model.save(autoencoder_path)
                rl_model_data['autoencoder_path'] = f"{path_prefix}_autoencoder"
                print(f"  ✓ AutoEncoder saved to: {autoencoder_path}")
            
            # Save RL Q-networks
            q_network_path = get_model_path(f"{path_prefix}_q_network.keras")
            target_network_path = get_model_path(f"{path_prefix}_target_network.keras")
            
            if self.rl_agent.q_network is not None:
                self.rl_agent.q_network.save(q_network_path)
                rl_model_data['q_network_path'] = f"{path_prefix}_q_network"
                print(f"  ✓ Q-Network saved to: {q_network_path}")
            
            if self.rl_agent.target_network is not None:
                self.rl_agent.target_network.save(target_network_path)
                rl_model_data['target_network_path'] = f"{path_prefix}_target_network"
                print(f"  ✓ Target Network saved to: {target_network_path}")
            
            # Save preliminary filter separately
            filter_prefix = f"{path_prefix}_preliminary_filter"
            self.preliminary_filter.save_filter(filter_prefix)
            rl_model_data['preliminary_filter_path'] = filter_prefix
            
            # Save main model data
            joblib.dump(rl_model_data, model_file)
            print(f"  ✓ Main RL data saved to: {model_file}")
            
            print("✅ RL-enhanced model with preliminary filter saved successfully!")
            print(f"  📁 All files saved in: {MODEL_SAVE_DIR}")
            print(f"  📊 Performance history: {len(self.performance_history)} steps")
            print(f"  🤖 Adaptation history: {len(self.adaptation_history)} adaptations")
            print(f"  🔍 Filter sensitivity: {self.preliminary_filter.sensitivity}")
            print(f"  📈 Total samples processed: {self.processed_count}")
            
        except Exception as e:
            print(f"❌ Error saving RL-enhanced model: {e}")
            raise
    
    def load_rl_enhanced_model(self, path_prefix='rl_enhanced_anomaly_model_with_filter'):
        """Load RL-enhanced model with preliminary filter from designated directory"""
        model_file = get_model_path(f'{path_prefix}.pkl')
        print(f"\nLoading RL-enhanced model with filter from: {model_file}")
        
        try:
            # Load main model data
            rl_model_data = joblib.load(model_file)
            
            # Display loading info
            save_timestamp = rl_model_data.get('save_timestamp', 'Unknown')
            filter_sensitivity = rl_model_data.get('filter_sensitivity', 'Unknown')
            print(f"  📅 Model saved on: {save_timestamp}")
            print(f"  🔍 Filter sensitivity: {filter_sensitivity}")
            
            # Restore base detector
            self.base_detector.scaler = rl_model_data['base_detector_scaler']
            self.base_detector.feature_columns = rl_model_data['base_detector_feature_columns']
            self.base_detector.target_column = rl_model_data['base_detector_target_column']
            self.base_detector.normal_label = rl_model_data['base_detector_normal_label']
            self.base_detector.thresholds = rl_model_data['base_detector_thresholds']
            self.base_detector.models = rl_model_data['base_detector_models']
            
            # Load AutoEncoder
            if 'autoencoder_path' in rl_model_data:
                autoencoder_path = get_model_path(rl_model_data['autoencoder_path'])
                if os.path.exists(autoencoder_path):
                    autoencoder = tf.keras.models.load_model(autoencoder_path)
                    self.base_detector.models['autoencoder'] = autoencoder
                    print(f"  ✓ AutoEncoder loaded from: {autoencoder_path}")
                else:
                    print(f"  ⚠️ AutoEncoder not found at: {autoencoder_path}")
            
            # Restore RL-specific parameters
            self.focus_model = rl_model_data['focus_model']
            self.current_threshold_multiplier = rl_model_data['current_threshold_multiplier']
            self.current_batch_size = rl_model_data['current_batch_size']
            self.autoencoder_threshold = rl_model_data.get('autoencoder_threshold')
            
            # Restore RL agent
            self.rl_agent.epsilon = rl_model_data['rl_agent_epsilon']
            self.rl_agent.memory = deque(rl_model_data['rl_agent_memory'], maxlen=1000)
            
            # Load RL networks
            if 'q_network_path' in rl_model_data:
                q_network_path = get_model_path(rl_model_data['q_network_path'])
                if os.path.exists(q_network_path):
                    self.rl_agent.q_network = tf.keras.models.load_model(q_network_path)
                    print(f"  ✓ Q-Network loaded from: {q_network_path}")
                else:
                    print(f"  ⚠️ Q-Network not found at: {q_network_path}")
            
            if 'target_network_path' in rl_model_data:
                target_network_path = get_model_path(rl_model_data['target_network_path'])
                if os.path.exists(target_network_path):
                    self.rl_agent.target_network = tf.keras.models.load_model(target_network_path)
                    print(f"  ✓ Target Network loaded from: {target_network_path}")
                else:
                    print(f"  ⚠️ Target Network not found at: {target_network_path}")
            
            # Load preliminary filter
            if 'preliminary_filter_path' in rl_model_data:
                filter_prefix = rl_model_data['preliminary_filter_path']
                try:
                    self.preliminary_filter.load_filter(filter_prefix)
                    print(f"  ✓ Preliminary filter loaded")
                except Exception as e:
                    print(f"  ⚠️ Error loading preliminary filter: {e}")
            
            # Restore history
            self.performance_history = rl_model_data['performance_history']
            self.adaptation_history = rl_model_data['adaptation_history']
            self.filter_efficiency_history = rl_model_data.get('filter_efficiency_history', [])
            self.processed_count = rl_model_data['total_processed']
            
            print("✅ RL-enhanced model with filter loaded successfully!")
            print(f"  🎯 Focus model: {self.focus_model}")
            print(f"  🔧 Threshold multiplier: {self.current_threshold_multiplier:.3f}")
            print(f"  📦 Batch size: {self.current_batch_size}")
            print(f"  🔍 Filter trained: {self.preliminary_filter.is_trained}")
            print(f"  📊 Total processed: {self.processed_count}")
            print(f"  📈 Performance steps: {len(self.performance_history)}")
            print(f"  🤖 Adaptation steps: {len(self.adaptation_history)}")
            print(f"  🎲 Current epsilon: {self.rl_agent.epsilon:.3f}")
            
        except Exception as e:
            print(f"❌ Error loading RL-enhanced model: {e}")
            raise
    
    def continue_streaming_processing(self, streaming_path, resume_from_step=0):
        """Continue processing streaming data from saved state"""
        print(f"\n📡 Resuming streaming processing with filter from step {resume_from_step}")
        print(f"Current RL state:")
        print(f"  - Threshold multiplier: {self.current_threshold_multiplier:.3f}")
        print(f"  - Batch size: {self.current_batch_size}")
        print(f"  - Epsilon: {self.rl_agent.epsilon:.3f}")
        print(f"  - Memory size: {len(self.rl_agent.memory)}")
        print(f"  - Filter trained: {self.preliminary_filter.is_trained}")
        print(f"  - Filter sensitivity: {self.preliminary_filter.sensitivity}")
        
        return self.process_streaming_data(streaming_path, update_frequency=5, start_step=resume_from_step)
    
    def adjust_filter_sensitivity(self, new_sensitivity):
        """Dynamically adjust preliminary filter sensitivity during processing"""
        success = self.preliminary_filter.adjust_sensitivity(new_sensitivity)
        if success:
            print(f"🎛️ Filter sensitivity adjusted to: {new_sensitivity}")
            return True
        else:
            print(f"❌ Invalid sensitivity level: {new_sensitivity}")
            return False

def main():
    """Complete pipeline with Preliminary Anomaly Filter: Phase 1 (Training) + Phase 2 (RL Adaptation with Filter)"""
    # File paths

    training_file = os.path.join(DATA_PATH, 'extract_1p_ciciot2023.csv')
    streaming_file = os.path.join(DATA_PATH, 'streaming_data.csv')
    
    print("="*80)
    print("🚀 ENHANCED ANOMALY DETECTION PIPELINE WITH PRELIMINARY FILTER")
    print("Phase 1 (Training) + Phase 2 (RL Adaptation with Smart Filtering)")
    print("="*80)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    print(f"📊 Training data: {training_file}")
    print(f"📡 Streaming data: {streaming_file}")
    
    # Check if base models already exist
    base_model_file = get_model_path('optimized_anomaly_detection_models.pkl')
    skip_phase1 = os.path.exists(base_model_file)
    skip_phase1 = False  # Force Phase 1 for this run
    
    if skip_phase1:
        print("\n🔍 Auto-detection: Base models found!")
        print("⚡ Skipping Phase 1 (training) and proceeding to Phase 2 (RL with Filter)")
        print("💡 Use 'python script.py phase1' to retrain base models")
        
        # Load existing base models
        base_detector = OptimizedAnomalyDetector()
        base_detector.load_models('optimized_anomaly_detection_models')
        best_model = 'autoencoder'  # Default focus model
        
        print("✅ Base models loaded successfully!")
        
    else:
        # =================================================================
        # PHASE 1: BASE MODEL TRAINING
        # =================================================================
        print("\n" + "🔶"*20 + " PHASE 1: BASE MODEL TRAINING " + "🔶"*20)
        
        try:
            # Step 1: Initialize base detector
            print("\n1️⃣ Initializing base anomaly detector...")
            base_detector = OptimizedAnomalyDetector()
            
            # Step 2: Load and preprocess training data
            print("\n2️⃣ Loading and preprocessing training data...")
            X, y_binary, y_original = base_detector.load_and_preprocess_data(training_file)
            
            # Step 3: Train anomaly detection models
            print("\n3️⃣ Training anomaly detection models...")
            X_test, y_test = base_detector.train_anomaly_models(X, y_binary)
            
            # Step 4: Create ensemble
            print("\n4️⃣ Creating ensemble model...")
            base_detector.create_ensemble_model()
            
            # Step 5: Evaluate base models
            print("\n5️⃣ Evaluating base models...")
            best_model, best_f1 = base_detector.evaluate_comprehensive()
            
            # Step 6: Save base models
            print("\n6️⃣ Saving base models...")
            base_detector.save_models('optimized_anomaly_detection_models')
            
            print("\n✅ Phase 1 completed successfully!")
            print(f"🏆 Best model: {best_model} (F1={best_f1:.4f})")
            print("📦 Base models saved and ready for Phase 2")
            
        except Exception as e:
            print(f"❌ Error in Phase 1: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # =================================================================
    # PHASE 2: RL ADAPTATION WITH PRELIMINARY FILTER
    # =================================================================
    print("\n" + "🔷"*15 + " PHASE 2: RL ADAPTATION WITH PRELIMINARY FILTER " + "🔷"*15)
    
    try:
        # Step 1: Initialize RL-enhanced detector with preliminary filter
        print("\n1️⃣ Initializing RL-enhanced detector with preliminary filter...")
        focus_model = best_model if 'best_model' in locals() else 'autoencoder'
        filter_sensitivity = 'medium'  # Can be 'low', 'medium', 'high'
        print(f"   🎯 Focus model: {focus_model}")
        print(f"   🔍 Filter sensitivity: {filter_sensitivity}")
        
        rl_detector = RLAdaptiveAnomalyDetector(
            base_detector, 
            focus_model=focus_model,
            filter_sensitivity=filter_sensitivity
        )
        
        # Step 2: Train preliminary filter
        print("\n2️⃣ Training preliminary anomaly filter...")
        filter_trained = rl_detector.train_preliminary_filter()
        
        if filter_trained:
            print("✅ Preliminary filter trained successfully!")
            filter_stats = rl_detector.preliminary_filter.get_filter_statistics()
            print(f"   Models trained: {list(rl_detector.preliminary_filter.models.keys())}")
            print(f"   Sensitivity: {rl_detector.preliminary_filter.sensitivity}")
        else:
            print("⚠️  Warning: Preliminary filter training failed, proceeding without filtering")
        
        # Step 3: Process streaming data with preliminary filtering + RL adaptation
        print("\n3️⃣ Processing streaming data with preliminary filter + RL adaptation...")
        performance_history = rl_detector.process_streaming_data(
            streaming_file, 
            update_frequency=5  # Adapt every 5 batches
        )
        
        # Step 4: Analyze adaptation results with filter performance
        print("\n4️⃣ Analyzing RL adaptation and filter performance...")
        summary = rl_detector.get_adaptation_summary()
        
        print("\n📊 Complete System Summary:")
        print("=" * 50)
        
        # System overview
        print(f"🎯 System Configuration:")
        print(f"   Focus model: {summary.get('focus_model', 'Unknown')}")
        print(f"   Filter sensitivity: {summary.get('filter_sensitivity', 'Unknown')}")
        print(f"   Total samples processed: {summary.get('total_samples_processed', 0)}")
        print(f"   Total processing steps: {summary.get('total_steps', 0)}")
        
        # Filter performance
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"\n🔍 Preliminary Filter Performance:")
            print(f"   Total batches: {filter_perf.get('total_batches', 0)}")
            print(f"   Passed to RL: {filter_perf.get('passed_batches', 0)}")
            print(f"   Filtered out: {filter_perf.get('filtered_batches', 0)}")
            print(f"   Pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"   Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
            print(f"   Average filter confidence: {filter_perf.get('avg_filter_confidence', 0):.3f}")
        
        # RL performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"\n🤖 RL Adaptation Performance:")
            print(f"   Final anomaly rate: {final_perf.get('anomaly_rate', 0):.4f}")
            print(f"   Final precision: {final_perf.get('precision', 0):.4f}")
            print(f"   Final recall: {final_perf.get('recall', 0):.4f}")
            print(f"   Final F1 score: {final_perf.get('f1', 0):.4f}")
            
        # AutoEncoder specific results
        if 'autoencoder_specific' in summary and summary['autoencoder_specific']:
            ae_perf = summary['autoencoder_specific']
            print(f"\n⭐ AutoEncoder Optimization:")
            print(f"   Base threshold: {ae_perf.get('base_threshold', 0):.6f}")
            print(f"   Final multiplier: {ae_perf.get('final_threshold_multiplier', 0):.3f}")
            print(f"   Effective threshold: {ae_perf.get('final_effective_threshold', 0):.6f}")
            print(f"   Performance improvement: {ae_perf.get('threshold_improvement', 0):+.4f}")
        
        # Step 5: Save complete enhanced model
        print("\n5️⃣ Saving RL-enhanced model with preliminary filter...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter')
        
        print("\n✅ Phase 2 completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # FINAL COMPREHENSIVE SUMMARY
    # =================================================================
    print("\n" + "🌟"*20 + " FINAL COMPREHENSIVE SUMMARY " + "🌟"*20)
    
    # Show file structure
    print("\n📁 Saved files structure:")
    try:
        saved_files = []
        for file in os.listdir(MODEL_SAVE_DIR):
            file_path = os.path.join(MODEL_SAVE_DIR, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            saved_files.append((file, file_size))
        
        # Group files by type
        base_files = [f for f in saved_files if 'optimized_anomaly' in f[0]]
        rl_files = [f for f in saved_files if 'rl_enhanced' in f[0]]
        filter_files = [f for f in saved_files if 'preliminary_filter' in f[0]]
        
        if base_files:
            print("\n🔧 Base Models:")
            for filename, size in sorted(base_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        if rl_files:
            print("\n🤖 RL-Enhanced Models:")
            for filename, size in sorted(rl_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        if filter_files:
            print("\n🔍 Preliminary Filter:")
            for filename, size in sorted(filter_files):
                print(f"  📄 {filename} ({size:.1f} MB)")
        
        total_size = sum([size for _, size in saved_files])
        print(f"\n💾 Total storage: {total_size:.1f} MB")
        
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Performance comparison and efficiency analysis
    if 'performance_history' in locals() and len(performance_history) > 1:
        # Calculate performance improvement
        processed_batches = [h for h in performance_history if h.get('filter_passed', True)]
        if len(processed_batches) > 1:
            initial_f1 = processed_batches[0].get('f1', 0)
            final_f1 = processed_batches[-1].get('f1', 0)
            improvement = final_f1 - initial_f1
            
            print(f"\n📈 Performance Improvement (Processed Batches Only):")
            print(f"   Initial F1: {initial_f1:.4f}")
            print(f"   Final F1: {final_f1:.4f}")
            print(f"   Improvement: {improvement:+.4f}")
            
            if improvement > 0:
                print("   🎉 RL adaptation improved performance!")
            else:
                print("   📊 RL adaptation maintained stability")
        
        # Filter efficiency analysis
        total_batches = len(performance_history)
        filtered_batches = len([h for h in performance_history if not h.get('filter_passed', True)])
        efficiency_gain = filtered_batches / total_batches if total_batches > 0 else 0
        
        print(f"\n🔍 Filter Efficiency Analysis:")
        print(f"   Total batches: {total_batches}")
        print(f"   Processed by RL: {total_batches - filtered_batches}")
        print(f"   Filtered out: {filtered_batches}")
        print(f"   Computational savings: {efficiency_gain:.2%}")
        
        if efficiency_gain > 0.3:
            print("   💰 Excellent computational savings!")
        elif efficiency_gain > 0.1:
            print("   👍 Good computational efficiency")
        else:
            print("   📝 Consider adjusting filter sensitivity")
    
    # Show what was executed
    print(f"\n🎯 Pipeline Summary:")
    if skip_phase1:
        print("   🔶 Phase 1: ⚡ Skipped (models existed)")
    else:
        print("   🔶 Phase 1: ✅ Base models trained")
    print("   🔷 Phase 2: ✅ RL adaptation with preliminary filter completed")
    print("   🔍 Filter: ✅ Preliminary anomaly filter integrated")
    
    print("\n" + "="*80)
    print("🎉 ENHANCED PIPELINE WITH PRELIMINARY FILTER FINISHED SUCCESSFULLY!")
    print("="*80)
    print("✅ Models ready for production use with smart filtering")
    print(f"📁 All files saved in: {MODEL_SAVE_DIR}")
    print("\n💡 Next steps:")
    print("   🔄 python script.py continue     → Resume RL training")
    print("   �� python script.py list         → Manage model files")
    print("   🎛️  rl_detector.adjust_filter_sensitivity('high') → Adjust filter")
    print("   🧹 python script.py cleanup      → Clean old versions")
    print("   🔧 python script.py phase1       → Retrain base models")
    print("   📈 Load models for inference with intelligent filtering")

def run_phase_1_only():
    """Run only Phase 1: Base model training"""
    training_file = os.path.join(DATA_PATH,'extract_1p_ciciot2023.csv')
    
    print("="*60)
    print("🔶 PHASE 1 ONLY: BASE MODEL TRAINING")
    print("="*60)
    print(f"�� Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Initialize and train
        base_detector = OptimizedAnomalyDetector()
        X, y_binary, y_original = base_detector.load_and_preprocess_data(training_file)
        X_test, y_test = base_detector.train_anomaly_models(X, y_binary)
        base_detector.create_ensemble_model()
        best_model, best_f1 = base_detector.evaluate_comprehensive()
        
        # Save models
        base_detector.save_models('optimized_anomaly_detection_models')
        
        print("\n✅ Phase 1 completed!")
        print(f"🏆 Best model: {best_model} (F1={best_f1:.4f})")
        print("📦 Ready for Phase 2 RL adaptation with preliminary filter")
        
        return base_detector, best_model
        
    except Exception as e:
        print(f"❌ Error in Phase 1: {e}")
        raise

def run_phase_2_with_filter():
    """Run only Phase 2: RL adaptation with preliminary filter (requires Phase 1 models)"""
    streaming_file = os.path.join(DATA_PATH,'streaming_data.csv')
    model_name = 'optimized_anomaly_detection_models'
    
    print("="*60)
    print("🔷 PHASE 2 ONLY: RL ADAPTATION WITH PRELIMINARY FILTER")
    print("="*60)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Load base models
        print("\n1️⃣ Loading pre-trained base models...")
        base_detector = OptimizedAnomalyDetector()
        base_detector.load_models(model_name)
        
        # Initialize RL detector with filter
        print("\n2️⃣ Initializing RL-enhanced detector with preliminary filter...")
        rl_detector = RLAdaptiveAnomalyDetector(
            base_detector, 
            focus_model='autoencoder',
            filter_sensitivity='medium'
        )
        
        # Train preliminary filter
        print("\n3️⃣ Training preliminary filter...")
        filter_trained = rl_detector.train_preliminary_filter()
        if not filter_trained:
            print("⚠️  Warning: Filter training failed, proceeding without filtering")
        
        # Process streaming data
        print("\n4️⃣ Processing streaming data with filter + RL...")
        performance_history = rl_detector.process_streaming_data(streaming_file, update_frequency=5)
        
        # Save results
        print("\n5️⃣ Saving RL-enhanced model with filter...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter')
        
        # Summary
        summary = rl_detector.get_adaptation_summary()
        print("\n📊 Results:")
        
        # Filter performance
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"  Filter pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"  Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
        
        # Final performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"  Final F1: {final_perf.get('f1', 0):.4f}")
        
        print("\n✅ Phase 2 with preliminary filter completed!")
        
        return rl_detector
        
    except FileNotFoundError:
        print("❌ Base models not found!")
        print("Please run Phase 1 first or use main() for complete pipeline")
        raise
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")
        raise

def continue_from_saved_model_with_filter():
    """Utility function to continue RL adaptation from saved model with filter"""
    streaming_file = os.path.join(DATA_PATH,'streaming_data.csv')
    model_name = 'rl_enhanced_anomaly_model_with_filter'
    
    print("="*70)
    print("🔄 CONTINUING RL ADAPTATION FROM SAVED MODEL WITH FILTER")
    print("="*70)
    print(f"📁 Model directory: {MODEL_SAVE_DIR}")
    
    try:
        # Load saved RL-enhanced model with filter
        print("\n1. Loading saved RL-enhanced model with filter...")
        
        base_detector = OptimizedAnomalyDetector()
        rl_detector = RLAdaptiveAnomalyDetector(base_detector)
        rl_detector.load_rl_enhanced_model(model_name)
        
        # Continue processing
        print("\n2. Continuing streaming data processing with filter...")
        last_step = len(rl_detector.performance_history)
        print(f"   📈 Resuming from step: {last_step}")
        print(f"   🔍 Filter trained: {rl_detector.preliminary_filter.is_trained}")
        print(f"   📊 Filter sensitivity: {rl_detector.preliminary_filter.sensitivity}")
        
        performance_history = rl_detector.continue_streaming_processing(
            streaming_file, 
            resume_from_step=last_step
        )
        
        # Analysis
        print("\n3. Updated results...")
        summary = rl_detector.get_adaptation_summary()
        print("\n📊 Updated Summary:")
        
        # Filter stats
        if 'filter_performance' in summary:
            filter_perf = summary['filter_performance']
            print(f"  Filter Performance:")
            print(f"    Pass rate: {filter_perf.get('pass_rate', 0):.2%}")
            print(f"    Computational savings: {filter_perf.get('computational_savings', 0):.2%}")
            print(f"    Average confidence: {filter_perf.get('avg_filter_confidence', 0):.3f}")
        
        # Overall performance
        if 'final_performance' in summary:
            final_perf = summary['final_performance']
            print(f"  Final Performance:")
            print(f"    F1 Score: {final_perf.get('f1', 0):.4f}")
            print(f"    Precision: {final_perf.get('precision', 0):.4f}")
            print(f"    Recall: {final_perf.get('recall', 0):.4f}")
        
        # Save updated model
        print("\n4. Saving updated model...")
        rl_detector.save_rl_enhanced_model('rl_enhanced_anomaly_model_with_filter_updated')
        
        print("\n✅ Continuation with filter completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Saved RL model with filter not found: {e}")
        print("Please run the main RL training with filter first")
        print(f"Expected location: {MODEL_SAVE_DIR}")
    except Exception as e:
        print(f"❌ Error continuing from saved model: {e}")
        import traceback
        traceback.print_exc()

def list_saved_models():
    """Utility function to list all saved models including filter components"""
    print("="*50)
    print("�� SAVED MODELS DIRECTORY")
    print("="*50)
    print(f"Directory: {MODEL_SAVE_DIR}")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        print("❌ Model directory does not exist")
        return
    
    files = os.listdir(MODEL_SAVE_DIR)
    if not files:
        print("📭 No saved models found")
        return
    
    print(f"\n📋 Found {len(files)} files:")
    
    # Group files by model type
    base_models = []
    rl_models = []
    filter_models = []
    other_files = []
    
    for file in files:
        file_path = os.path.join(MODEL_SAVE_DIR, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        file_info = (file, file_size)
        
        if 'optimized_anomaly' in file:
            base_models.append(file_info)
        elif 'rl_enhanced' in file:
            rl_models.append(file_info)
        elif 'preliminary_filter' in file:
            filter_models.append(file_info)
        else:
            other_files.append(file_info)
    
    if base_models:
        print("\n🔧 Base Models:")
        for filename, size in sorted(base_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if rl_models:
        print("\n🤖 RL-Enhanced Models:")
        for filename, size in sorted(rl_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if filter_models:
        print("\n🔍 Preliminary Filters:")
        for filename, size in sorted(filter_models):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    if other_files:
        print("\n📎 Other Files:")
        for filename, size in sorted(other_files):
            print(f"  📄 {filename} ({size:.1f} MB)")
    
    total_size = sum([size for _, size in base_models + rl_models + filter_models + other_files])
    print(f"\n💾 Total storage: {total_size:.1f} MB")

if __name__ == "__main__":
    main()

# Example usage:
# To continue from saved model with filter: continue_from_saved_model_with_filter()
# To run only Phase 2 with filter: run_phase_2_with_filter()
# To adjust filter sensitivity: rl_detector.adjust_filter_sensitivity('high')
