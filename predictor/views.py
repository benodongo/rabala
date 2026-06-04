import logging
import os

import numpy as np
from django.conf import settings
from django.shortcuts import render

from .forms import IndigenousForm, IntegratedForm
from .fuzzy_logic import determine_lake_position
from .models import Prediction
from .scientific import (
    EnsembleModel,
    RISK_MAPPING,
    get_scientific_weather,
    integrate_predictions,
)

logger = logging.getLogger(__name__)

# Per-class decision thresholds used by the ensemble's threshold-adjusted
# mode. Tuned to favour recall on minority risk classes.
DEFAULT_CLASS_THRESHOLDS = {'Normal': 0.6, 'Good': 0.3, 'Risky': 0.3, 'Bad': 0.3}

MOON_MAP = {
    "New": 0, "Ascending": 5, "Midway": 14, "Descending": 22, "Low": 28,
    "Any Position": 14,
}

CLOUD_MAP = {"Clear": 5, "Light": 20, "Cloudy": 50, "Heavy": 85}

TEMP_MAP = {
    "Cold": 34.5, "Cool": 36.0, "Low": 35.5, "Moderate": 37.0,
    "Warm": 38.0, "High": 39.0, "Hot": 39.5,
}

# ---------------------------------------------------------------------------
# Pre-load ensemble model
# ---------------------------------------------------------------------------

ENSEMBLE_MODEL = None
ENSEMBLE_LOAD_ERROR = None
try:
    _model_path = os.path.join(
        settings.BASE_DIR, 'predictor', 'models', 'ensemble_model.joblib'
    )
    if os.path.exists(_model_path):
        ENSEMBLE_MODEL = EnsembleModel.load(_model_path)
        ENSEMBLE_MODEL.set_class_thresholds(DEFAULT_CLASS_THRESHOLDS)
        logger.info("Ensemble model loaded: %s", ENSEMBLE_MODEL.metadata_)
    else:
        ENSEMBLE_LOAD_ERROR = f"Model file not found at {_model_path}"
except Exception as exc:
    ENSEMBLE_LOAD_ERROR = str(exc)
    logger.exception("Failed to load ensemble model: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_version():
    if ENSEMBLE_MODEL and ENSEMBLE_MODEL.metadata_.get('trained_at'):
        return (
            f"v{ENSEMBLE_MODEL.metadata_.get('schema_version', '?')}"
            f"@{ENSEMBLE_MODEL.metadata_['trained_at']}"
        )
    return ''


def _log_prediction(*, mode, data, ik_label, final_label, sci_data=None,
                    risk_score=None, probabilities=None, version=''):
    """Persist a prediction record. Failures are logged but never raised."""
    try:
        sci_data = sci_data or {}
        Prediction.objects.create(
            mode=mode,
            wind_type=data.get('wind_type', ''),
            moon_phase=data.get('moon_phase', ''),
            cloud_condition=data.get('cloud_condition', ''),
            body_temperature=data.get('body_temperature', ''),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            sci_temp=sci_data.get('temp'),
            sci_precipitation=sci_data.get('precipitation'),
            sci_wind_speed=sci_data.get('wind_speed'),
            sci_humidity=sci_data.get('humidity'),
            ik_label=ik_label or '',
            final_label=final_label,
            risk_score=risk_score,
            probabilities=probabilities,
            model_version=version,
        )
    except Exception as exc:
        logger.warning("Failed to persist prediction: %s", exc)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def indigenous_prediction(request):
    if request.method == 'POST':
        form = IndigenousForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            result = determine_lake_position(
                data['wind_type'],
                MOON_MAP[data['moon_phase']],
                CLOUD_MAP[data['cloud_condition']],
                TEMP_MAP[data['body_temperature']],
            )
            _log_prediction(
                mode='indigenous',
                data=data,
                ik_label=result[0],
                final_label=result[0],
                risk_score=RISK_MAPPING.get(result[0], (None,))[0],
            )
            return render(request, 'indigenous.html', {'form': form, 'result': result})
    else:
        form = IndigenousForm()
    return render(request, 'indigenous.html', {'form': form})


def integrated_prediction(request):
    if request.method == 'POST':
        form = IntegratedForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            ik_result = determine_lake_position(
                data['wind_type'],
                MOON_MAP[data['moon_phase']],
                CLOUD_MAP[data['cloud_condition']],
                TEMP_MAP[data['body_temperature']],
            )
            sci_data = get_scientific_weather(data['latitude'], data['longitude'])
            prediction, risk = integrate_predictions(ik_result, sci_data)

            # Map the prose verdict back to a class label so we can store it.
            label_from_risk = (
                'Good' if risk < 0.3
                else 'Normal' if risk < 0.6
                else 'Bad'
            )
            _log_prediction(
                mode='integrated',
                data=data,
                ik_label=ik_result[0],
                final_label=label_from_risk,
                sci_data=sci_data,
                risk_score=risk,
            )

            return render(request, 'integrated.html', {
                'form':            form,
                'ik_result':       ik_result,
                'sci_data':        sci_data,
                'prediction':      prediction,
                'risk':            risk,
                'risk_percentage': risk * 100,
            })
    else:
        form = IntegratedForm()
    return render(request, 'integrated.html', {'form': form})


def ensemble_prediction(request):
    if not ENSEMBLE_MODEL:
        return render(request, 'error.html', {
            'message': f'Ensemble model not loaded: {ENSEMBLE_LOAD_ERROR or "unknown error"}',
        })

    form = IntegratedForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data

        ik_inputs = (
            str(data['wind_type']),
            int(MOON_MAP.get(data['moon_phase'], 14)),
            int(CLOUD_MAP.get(data['cloud_condition'], 50)),
            int(TEMP_MAP.get(data['body_temperature'], 37)),
        )

        sci_data = get_scientific_weather(data['latitude'], data['longitude']) or {}
        current_temp = float(sci_data.get('temp', 27.5))
        sci_features = np.array([
            float(sci_data.get('temp_min', current_temp - 2.5)),
            float(sci_data.get('temp_max', current_temp + 2.5)),
            float(sci_data.get('precip_sum', sci_data.get('precipitation', 0.0))),
            float(sci_data.get('wind_max', sci_data.get('wind_speed', 5.0))),
        ], dtype=np.float32)

        ik_result = determine_lake_position(*ik_inputs)

        try:
            ensemble_pred, proba_dict = ENSEMBLE_MODEL.predict(
                ik_inputs, sci_features, adjust_thresholds=True, return_proba=True
            )
        except Exception as exc:
            logger.exception("Ensemble prediction failed: %s", exc)
            return render(request, 'error.html', {
                'message': 'Prediction failed. Please try again later.',
            })

        risk_value, risk_str = RISK_MAPPING.get(
            ensemble_pred, (0.5, "Uncertain conditions")
        )

        # Sort probabilities high → low for display.
        proba_sorted = sorted(
            ((cls, float(p)) for cls, p in proba_dict.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        top_confidence = proba_sorted[0][1] if proba_sorted else None

        _log_prediction(
            mode='ensemble',
            data=data,
            ik_label=ik_result[0],
            final_label=ensemble_pred,
            sci_data=sci_data,
            risk_score=risk_value,
            probabilities={k: float(v) for k, v in proba_dict.items()},
            version=_model_version(),
        )

        return render(request, 'ensemble.html', {
            'form': form,
            'ensemble_pred': ensemble_pred,
            'ensemble_str': risk_str,
            'ensemble_risk': risk_value,
            'ik_result': ik_result,
            'sci_data': sci_data,
            'probabilities': proba_sorted,
            'top_confidence': top_confidence,
            'model_version': _model_version(),
        })

    return render(request, 'ensemble.html', {'form': form})


def landing(request):
    return render(request, 'landing.html')


def about(request):
    return render(request, 'about.html')


def how_it_works(request):
    return render(request, 'how_it_works.html')
