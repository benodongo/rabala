import requests
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from collections import defaultdict

from .fuzzy_logic import determine_lake_position

def get_scientific_weather(lat, lon):
    """Fetch real-time weather data from Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m"
    try:
        response = requests.get(url)
        data = response.json()['current']
        return {
            'temp': data['temperature_2m'],
            'humidity': data['relative_humidity_2m'],
            'precipitation': data['precipitation'],
            'cloud_cover': data['cloud_cover'],
            'wind_speed': data['wind_speed_10m']
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None

class ScientificModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.classes_ = None

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes_
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls()
        model.model = data['model']
        model.scaler = data['scaler']
        model.classes_ = data['classes']
        return model

def integrate_predictions(indigenous_pred, scientific_data):
    """Combine both prediction systems using adaptive weighting"""
    risk_mapping = {'Good': 0, 'Normal': 0.4, 'Risky': 0.7, 'Bad': 1.0}
    indigenous_risk = risk_mapping.get(indigenous_pred[0], 0.5)

    if scientific_data:
        sci_risk = (scientific_data['precipitation']/50 * 0.6 +
                   scientific_data['wind_speed']/15 * 0.4)
    else:
        sci_risk = 0.5  # Fallback value

    combined_risk = (indigenous_risk * 0.6 + sci_risk * 0.4)

    if combined_risk < 0.3:
        return "Excellent fishing conditions 🌞", combined_risk
    elif combined_risk < 0.6:
        return "Normal fishing conditions ⛅", combined_risk
    else:
        return "Dangerous fishing conditions ⚠️", combined_risk


class EnsembleModel:
    def __init__(self):
        self.ik_encoder = LabelEncoder()
        self.sci_model = ScientificModel()  # Initialize the scientific model
        self.ensemble = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.calibrated_ensemble = None
        self.classes_ = None
        self.class_weights_ = None
        self.feature_names_ = None

    def train(self, X_ik, X_sci, y, ik_preds=None, class_weights=None):
        """
        Train the ensemble model with improved handling of class imbalance
        
        Args:
            X_ik: List of IK feature sets
            X_sci: Array of scientific features
            y: Target labels
            ik_preds: Optional pre-computed IK predictions
            class_weights: Dictionary of class weights
        """
        # Initialize and encode target classes
        self.classes_ = np.unique(y)
        self.ik_encoder.fit(self.classes_)
        self.class_weights_ = class_weights
        
        # 1. Process IK predictions
        if ik_preds is None:
            ik_preds = [determine_lake_position(*params)[0] for params in X_ik]
        
        # Ensure all predictions are valid classes
        valid_classes = set(self.ik_encoder.classes_)
        ik_preds = [p if p in valid_classes else self.ik_encoder.classes_[0] for p in ik_preds]
        
        # 2. Train scientific model (without sample weights if not supported)
        try:
            if class_weights:
                sample_weights = compute_sample_weight(class_weights, y)
                self.sci_model.train(X_sci, y, sample_weight=sample_weights)
            else:
                self.sci_model.train(X_sci, y)
        except TypeError:
            # Fallback if ScientificModel doesn't support sample_weight
            self.sci_model.train(X_sci, y)
        
        # 3. Prepare ensemble features
        sci_probs = self.sci_model.predict_proba(X_sci)
        ik_encoded = self.ik_encoder.transform(ik_preds)
        ik_onehot = np.eye(len(self.ik_encoder.classes_))[ik_encoded]
        
        ensemble_X = np.hstack((sci_probs, ik_onehot))
        self.feature_names_ = (
            [f"sci_prob_{cls}" for cls in self.ik_encoder.classes_] +
            [f"ik_{cls}" for cls in self.ik_encoder.classes_]
        )
        
        # 4. Train ensemble classifier with calibration
        y_encoded = self.ik_encoder.transform(y)
        
        # Apply class weights to ensemble training
        if class_weights:
            sample_weights = compute_sample_weight(class_weights, y)
            self.ensemble.fit(ensemble_X, y_encoded, sample_weight=sample_weights)
        else:
            self.ensemble.fit(ensemble_X, y_encoded)
        
        # Calibrate for better probability estimates
        self.calibrated_ensemble = CalibratedClassifierCV(
            self.ensemble, method='sigmoid', cv='prefit'
        )
        self.calibrated_ensemble.fit(ensemble_X, y_encoded)

    def predict(self, ik_inputs, sci_features, return_proba=False, adjust_thresholds=False):
        """
        Make predictions with optional probability output and threshold adjustment
        
        Args:
            ik_inputs: IK feature values
            sci_features: Scientific feature values
            return_proba: Whether to return probabilities
            adjust_thresholds: Whether to use class-specific thresholds
            
        Returns:
            Predicted class or (class, probabilities) if return_proba=True
        """
        # 1. Get indigenous prediction
        lake_pos, _, _, _ = determine_lake_position(*ik_inputs)
        if lake_pos not in self.ik_encoder.classes_:
            lake_pos = self.ik_encoder.classes_[0]  # Default to first class
        
        ik_encoded = self.ik_encoder.transform([lake_pos])[0]
        ik_onehot = np.eye(len(self.ik_encoder.classes_))[ik_encoded]
        
        # 2. Get scientific probabilities
        sci_probs = self.sci_model.predict_proba([sci_features])[0]
        
        # 3. Combine features
        ensemble_input = np.hstack((sci_probs, ik_onehot))
        
        # 4. Make prediction
        if adjust_thresholds:
            # Get calibrated probabilities
            proba = self.calibrated_ensemble.predict_proba([ensemble_input])[0]
            proba_dict = dict(zip(self.ik_encoder.classes_, proba))
            
            # Apply class-specific thresholds (default 0.5 for all)
            thresholds = getattr(self, 'class_thresholds_', 
                               {cls: 0.5 for cls in self.ik_encoder.classes_})
            
            # Find all classes that meet their threshold
            predicted = [cls for cls in self.ik_encoder.classes_ 
                        if proba_dict[cls] >= thresholds[cls]]
            
            if not predicted:
                # Default to class with highest probability if none meet threshold
                final_pred = self.ik_encoder.classes_[np.argmax(proba)]
            else:
                # Choose the class with highest probability among those meeting thresholds
                final_pred = max(predicted, key=lambda x: proba_dict[x])
            
            if return_proba:
                return final_pred, proba_dict
            return final_pred
        
        else:
            # Standard prediction
            if return_proba:
                proba = self.calibrated_ensemble.predict_proba([ensemble_input])[0]
                proba_dict = dict(zip(self.ik_encoder.classes_, proba))
                pred = self.calibrated_ensemble.predict([ensemble_input])[0]
                return self.ik_encoder.inverse_transform([pred])[0], proba_dict
            else:
                pred = self.calibrated_ensemble.predict([ensemble_input])[0]
                return self.ik_encoder.inverse_transform([pred])[0]

    def predict_proba(self, ik_inputs, sci_features):
        """Get calibrated class probabilities"""
        _, proba = self.predict(ik_inputs, sci_features, return_proba=True)
        return proba

    def set_class_thresholds(self, thresholds):
        """
        Set custom prediction thresholds for each class
        
        Args:
            thresholds: Dictionary of {class_name: threshold}
        """
        self.class_thresholds_ = thresholds

    def save(self, path):
        """Save the complete model state"""
        joblib.dump({
            'ik_encoder': self.ik_encoder,
            'sci_model': self.sci_model,
            'ensemble': self.ensemble,
            'calibrated_ensemble': self.calibrated_ensemble,
            'classes': self.classes_,
            'class_weights': self.class_weights_,
            'feature_names': self.feature_names_
        }, path)

    @classmethod
    def load(cls, path):
        """Load a saved model"""
        data = joblib.load(path)
        model = cls()
        model.ik_encoder = data['ik_encoder']
        model.sci_model = data['sci_model']
        model.ensemble = data['ensemble']
        model.calibrated_ensemble = data['calibrated_ensemble']
        model.classes_ = data['classes']
        model.class_weights_ = data.get('class_weights')
        model.feature_names_ = data.get('feature_names')
        return model

    def get_feature_importance(self):
        """Get feature importance from the ensemble model"""
        if hasattr(self.ensemble, 'coef_'):
            return dict(zip(self.feature_names_, self.ensemble.coef_[0]))
        return None