import logging
from datetime import datetime, timezone

import requests
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

from .fuzzy_logic import determine_lake_position, fuzzy_posterior, LAKE_POS_CLASSES

logger = logging.getLogger(__name__)

MODEL_SCHEMA_VERSION = 3

# Shared risk taxonomy — single source of truth for both rule-based fusion
# and the ensemble view. Keys must match the labels produced by the fuzzy
# rule base and the trained ensemble classifier.
RISK_MAPPING = {
    'Good':   (0.2, "Excellent fishing conditions"),
    'Normal': (0.4, "Normal fishing conditions"),
    'Risky':  (0.7, "Caution advised — risky conditions"),
    'Bad':    (1.0, "Dangerous fishing conditions"),
}


def get_scientific_weather(lat, lon, timeout=None, use_cache=True):
    """Fetch current + daily forecast from Open-Meteo.

    Results are memoised per-(lat,lon) in the Django cache for
    settings.WEATHER_CACHE_SECONDS to limit upstream load. Returns a dict
    with the four features the ensemble was trained on plus current
    readings, or None on failure.
    """
    cache_key = None
    if use_cache:
        try:
            from django.core.cache import cache
            from django.conf import settings as dj_settings
            cache_key = f"weather:{round(float(lat), 3)}:{round(float(lon), 3)}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            if timeout is None:
                timeout = getattr(dj_settings, 'WEATHER_API_TIMEOUT', 5)
        except Exception:
            # Django not configured (e.g. unit test calling directly) — skip cache.
            cache_key = None

    if timeout is None:
        timeout = 5

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,"
        "cloud_cover,wind_speed_10m"
        "&daily=temperature_2m_min,temperature_2m_max,precipitation_sum,"
        "wind_speed_10m_max"
        "&timezone=auto&forecast_days=1"
    )
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        current = payload['current']
        daily = payload['daily']
        result = {
            'temp':          current['temperature_2m'],
            'humidity':      current['relative_humidity_2m'],
            'precipitation': current['precipitation'],
            'cloud_cover':   current['cloud_cover'],
            'wind_speed':    current['wind_speed_10m'],
            'temp_min':      daily['temperature_2m_min'][0],
            'temp_max':      daily['temperature_2m_max'][0],
            'precip_sum':    daily['precipitation_sum'][0],
            'wind_max':      daily['wind_speed_10m_max'][0],
            'fetched_at':    datetime.now(timezone.utc).isoformat(),
        }
        if cache_key:
            try:
                from django.core.cache import cache
                from django.conf import settings as dj_settings
                cache.set(
                    cache_key,
                    result,
                    getattr(dj_settings, 'WEATHER_CACHE_SECONDS', 600),
                )
            except Exception:
                pass
        return result
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("Open-Meteo fetch failed for (%s, %s): %s", lat, lon, exc)
        return None


class ScientificModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.classes_ = None

    def train(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
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
    """Combine fuzzy and scientific signals.

    Sci component uses tighter denominators so realistic Lake Victoria
    precipitation (~15 mm/h is already heavy) and wind (~10 m/s drives
    dangerous waves) can saturate to 1.0. Fusion is a weighted blend with
    a max-override so a strongly dangerous sci reading isn't diluted by a
    calm indigenous verdict.
    """
    indigenous_risk = RISK_MAPPING.get(indigenous_pred[0], (0.5, ""))[0]

    if scientific_data:
        precip_risk = min(1.0, scientific_data['precipitation'] / 15.0)
        wind_risk   = min(1.0, scientific_data['wind_speed']   / 10.0)
        sci_risk = max(0.0, min(1.0, 0.55 * precip_risk + 0.45 * wind_risk))
        weighted = 0.5 * indigenous_risk + 0.5 * sci_risk
        combined_risk = max(weighted, 0.85 * max(indigenous_risk, sci_risk))
    else:
        combined_risk = indigenous_risk

    if combined_risk < 0.3:
        return "Excellent fishing conditions", combined_risk
    if combined_risk < 0.55:
        return "Normal fishing conditions", combined_risk
    if combined_risk < 0.75:
        return "Caution advised — risky conditions", combined_risk
    return "Dangerous fishing conditions", combined_risk


class EnsembleModel:
    def __init__(self):
        self.ik_encoder = LabelEncoder()
        self.sci_model = ScientificModel()
        self.ensemble = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.calibrated_ensemble = None
        self.classes_ = None
        self.class_weights_ = None
        self.feature_names_ = None
        self.class_thresholds_ = None
        # When True, the IK side of the input is a normalised fuzzy
        # posterior vector instead of a one-hot encoding of the
        # argmax — gives the linear ensemble graded evidence rather
        # than a single high-signal discrete feature.
        self.uses_posterior_ = True
        self.metadata_ = {
            'schema_version': MODEL_SCHEMA_VERSION,
            'trained_at': None,
            'n_training_samples': None,
            'class_distribution': None,
        }

    def _ik_vector(self, ik_inputs):
        """Per-class IK feature vector aligned to self.ik_encoder.classes_."""
        classes = list(self.ik_encoder.classes_)
        if self.uses_posterior_:
            post = fuzzy_posterior(*ik_inputs, classes=classes)
            if sum(post) <= 0.0:
                # No rule fired — fall back to argmax one-hot on the default.
                lake_pos, *_ = determine_lake_position(*ik_inputs)
                if lake_pos not in classes:
                    lake_pos = classes[0]
                vec = [0.0] * len(classes)
                vec[classes.index(lake_pos)] = 1.0
                return vec
            return post
        lake_pos, *_ = determine_lake_position(*ik_inputs)
        if lake_pos not in classes:
            lake_pos = classes[0]
        vec = [0.0] * len(classes)
        vec[classes.index(lake_pos)] = 1.0
        return vec

    def train(self, X_ik, X_sci, y, ik_preds=None, class_weights=None,
              ik_posteriors=None):
        """Train the ensemble with proper (non-leaky) calibration.

        ik_posteriors : optional pre-computed list of fuzzy posterior
            vectors (one per row of X_ik). When omitted and
            ``self.uses_posterior_`` is True they are computed from
            X_ik on the fly. Order must match ``self.ik_encoder.classes_``
            after fitting — pass classes through ``LAKE_POS_CLASSES``.
        """
        self.classes_ = np.unique(y)
        self.ik_encoder.fit(self.classes_)
        self.class_weights_ = class_weights
        classes = list(self.ik_encoder.classes_)

        if self.uses_posterior_:
            if ik_posteriors is None:
                ik_posteriors = [
                    fuzzy_posterior(*params, classes=classes) for params in X_ik
                ]
            ik_feat = np.asarray(ik_posteriors, dtype=float)
            # Rows with no rule-firings get an argmax one-hot fallback so
            # the model still sees a signal.
            zero_rows = np.where(ik_feat.sum(axis=1) <= 0.0)[0]
            if zero_rows.size:
                eye = np.eye(len(classes))
                fallback_preds = [determine_lake_position(*X_ik[i])[0] for i in zero_rows]
                fallback_preds = [
                    p if p in classes else classes[0] for p in fallback_preds
                ]
                for row, pred in zip(zero_rows, fallback_preds):
                    ik_feat[row] = eye[classes.index(pred)]
            feature_prefix = "ik_post_"
        else:
            if ik_preds is None:
                ik_preds = [determine_lake_position(*params)[0] for params in X_ik]
            valid_classes = set(classes)
            ik_preds = [p if p in valid_classes else classes[0] for p in ik_preds]
            ik_encoded = self.ik_encoder.transform(ik_preds)
            ik_feat = np.eye(len(classes))[ik_encoded]
            feature_prefix = "ik_"

        sample_weights = (
            compute_sample_weight(class_weights, y) if class_weights else None
        )

        self.sci_model.train(X_sci, y, sample_weight=sample_weights)
        sci_probs = self.sci_model.predict_proba(X_sci)

        ensemble_X = np.hstack((sci_probs, ik_feat))
        self.feature_names_ = (
            [f"sci_prob_{cls}" for cls in classes]
            + [f"{feature_prefix}{cls}" for cls in classes]
        )

        y_encoded = self.ik_encoder.transform(y)

        if sample_weights is not None:
            self.ensemble.fit(ensemble_X, y_encoded, sample_weight=sample_weights)
        else:
            self.ensemble.fit(ensemble_X, y_encoded)

        min_class_count = int(np.bincount(y_encoded).min())
        cv_folds = max(2, min(3, min_class_count))
        try:
            base = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.calibrated_ensemble = CalibratedClassifierCV(
                base, method='sigmoid', cv=cv_folds
            )
            if sample_weights is not None:
                self.calibrated_ensemble.fit(
                    ensemble_X, y_encoded, sample_weight=sample_weights
                )
            else:
                self.calibrated_ensemble.fit(ensemble_X, y_encoded)
        except ValueError as exc:
            logger.warning("Cross-fit calibration failed (%s); using prefit.", exc)
            self.calibrated_ensemble = CalibratedClassifierCV(
                self.ensemble, method='sigmoid', cv='prefit'
            )
            self.calibrated_ensemble.fit(ensemble_X, y_encoded)

        unique, counts = np.unique(y, return_counts=True)
        self.metadata_ = {
            'schema_version': MODEL_SCHEMA_VERSION,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'n_training_samples': int(len(y)),
            'class_distribution': {str(k): int(v) for k, v in zip(unique, counts)},
            'uses_posterior': bool(self.uses_posterior_),
        }

    def predict(self, ik_inputs, sci_features, return_proba=False,
                adjust_thresholds=False, thresholds=None):
        """Predict ensemble label. Always returns a string (or (str, dict))."""
        ik_feat = np.asarray(self._ik_vector(ik_inputs), dtype=float)

        sci_probs = self.sci_model.predict_proba([sci_features])[0]
        ensemble_input = np.hstack((sci_probs, ik_feat))

        proba = self.calibrated_ensemble.predict_proba([ensemble_input])[0]
        proba_dict = dict(zip(self.ik_encoder.classes_, proba))

        if adjust_thresholds:
            active_thresholds = (
                thresholds
                or self.class_thresholds_
                or {cls: 0.5 for cls in self.ik_encoder.classes_}
            )
            qualifying = [
                cls for cls in self.ik_encoder.classes_
                if proba_dict[cls] >= active_thresholds.get(cls, 0.5)
            ]
            if qualifying:
                final_pred = max(qualifying, key=lambda c: proba_dict[c])
            else:
                final_pred = self.ik_encoder.classes_[int(np.argmax(proba))]
        else:
            pred_idx = int(self.calibrated_ensemble.predict([ensemble_input])[0])
            final_pred = self.ik_encoder.inverse_transform([pred_idx])[0]

        final_pred = str(final_pred)
        if return_proba:
            return final_pred, proba_dict
        return final_pred

    def predict_proba(self, ik_inputs, sci_features):
        """Get calibrated class probabilities."""
        _, proba = self.predict(ik_inputs, sci_features, return_proba=True)
        return proba

    def set_class_thresholds(self, thresholds):
        """Persist custom per-class decision thresholds."""
        self.class_thresholds_ = thresholds

    def save(self, path):
        joblib.dump({
            'ik_encoder': self.ik_encoder,
            'sci_model': self.sci_model,
            'ensemble': self.ensemble,
            'calibrated_ensemble': self.calibrated_ensemble,
            'classes': self.classes_,
            'class_weights': self.class_weights_,
            'feature_names': self.feature_names_,
            'class_thresholds': self.class_thresholds_,
            'uses_posterior': self.uses_posterior_,
            'metadata': self.metadata_,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls()
        model.ik_encoder = data['ik_encoder']
        model.sci_model = data['sci_model']
        model.ensemble = data['ensemble']
        model.calibrated_ensemble = data['calibrated_ensemble']
        model.classes_ = data['classes']
        model.class_weights_ = data.get('class_weights')
        model.feature_names_ = data.get('feature_names')
        model.class_thresholds_ = data.get('class_thresholds')
        # Older v2 models used one-hot IK encoding — fall back automatically.
        loaded_meta = data.get('metadata') or {}
        if 'uses_posterior' in data:
            model.uses_posterior_ = bool(data['uses_posterior'])
        else:
            model.uses_posterior_ = bool(loaded_meta.get('uses_posterior', False))
        model.metadata_.update(loaded_meta)
        return model

    def get_feature_importance(self):
        """Coefficients from the underlying (uncalibrated) logistic model."""
        if hasattr(self.ensemble, 'coef_'):
            return dict(zip(self.feature_names_, self.ensemble.coef_[0]))
        return None
