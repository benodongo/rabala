"""JSON endpoints for programmatic access (mobile/USSD gateways, scripts)."""
import json
import logging

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .fuzzy_logic import determine_lake_position
from .models import Prediction
from .scientific import RISK_MAPPING, get_scientific_weather
from .views import (
    CLOUD_MAP,
    ENSEMBLE_LOAD_ERROR,
    ENSEMBLE_MODEL,
    MOON_MAP,
    TEMP_MAP,
    _log_prediction,
    _model_version,
)

logger = logging.getLogger(__name__)


def _bad_request(message, **extra):
    return JsonResponse({'error': message, **extra}, status=400)


def _parse_json_body(request):
    if request.content_type and 'application/json' in request.content_type:
        return json.loads(request.body.decode('utf-8') or '{}')
    return request.POST.dict()


@csrf_exempt
@require_http_methods(['POST'])
def predict_api(request):
    """Run the full ensemble pipeline from a JSON body.

    Required: wind_type, moon_phase, cloud_condition, body_temperature.
    Optional: latitude, longitude (defaults to Kabuto).
    """
    if not ENSEMBLE_MODEL:
        return JsonResponse(
            {'error': 'model_unavailable', 'detail': ENSEMBLE_LOAD_ERROR},
            status=503,
        )

    try:
        payload = _parse_json_body(request)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _bad_request(f'invalid_json: {exc}')

    required = ['wind_type', 'moon_phase', 'cloud_condition', 'body_temperature']
    missing = [k for k in required if not payload.get(k)]
    if missing:
        return _bad_request('missing_fields', fields=missing)

    try:
        wind = str(payload['wind_type'])
        moon = MOON_MAP[payload['moon_phase']]
        cloud = CLOUD_MAP[payload['cloud_condition']]
        body_temp = TEMP_MAP[payload['body_temperature']]
    except KeyError as exc:
        return _bad_request(f'unknown_category: {exc}')

    try:
        lat = float(payload.get('latitude', -0.419))
        lon = float(payload.get('longitude', 31.893))
    except (TypeError, ValueError):
        return _bad_request('latitude/longitude must be numeric')

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return _bad_request('latitude/longitude out of range')

    ik_inputs = (wind, int(moon), int(cloud), int(body_temp))
    sci_data = get_scientific_weather(lat, lon) or {}
    current_temp = float(sci_data.get('temp', 27.5))
    sci_features = np.array([
        float(sci_data.get('temp_min', current_temp - 2.5)),
        float(sci_data.get('temp_max', current_temp + 2.5)),
        float(sci_data.get('precip_sum', sci_data.get('precipitation', 0.0))),
        float(sci_data.get('wind_max', sci_data.get('wind_speed', 5.0))),
    ], dtype=np.float32)

    ik_result = determine_lake_position(*ik_inputs)

    try:
        label, proba = ENSEMBLE_MODEL.predict(
            ik_inputs, sci_features, adjust_thresholds=True, return_proba=True,
        )
    except Exception as exc:
        logger.exception("API prediction failed: %s", exc)
        return JsonResponse({'error': 'prediction_failed'}, status=500)

    risk_value, risk_str = RISK_MAPPING.get(label, (0.5, "Uncertain conditions"))
    proba_clean = {k: float(v) for k, v in proba.items()}

    _log_prediction(
        mode='api',
        data={
            'wind_type': wind,
            'moon_phase': payload['moon_phase'],
            'cloud_condition': payload['cloud_condition'],
            'body_temperature': payload['body_temperature'],
            'latitude': lat, 'longitude': lon,
        },
        ik_label=ik_result[0],
        final_label=label,
        sci_data=sci_data,
        risk_score=risk_value,
        probabilities=proba_clean,
        version=_model_version(),
    )

    return JsonResponse({
        'label': label,
        'recommendation': risk_str,
        'risk_score': risk_value,
        'probabilities': proba_clean,
        'indigenous_label': ik_result[0],
        'scientific_data': sci_data or None,
        'model_version': _model_version(),
    })


@require_http_methods(['POST'])
@csrf_exempt
def feedback_api(request, prediction_id):
    """Attach an observed outcome to a previously-logged prediction."""
    try:
        payload = _parse_json_body(request)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _bad_request(f'invalid_json: {exc}')

    outcome = payload.get('actual_outcome', '').strip().title()
    notes = payload.get('notes', '')

    valid = {c for c, _ in Prediction.LABEL_CHOICES}
    if outcome and outcome not in valid:
        return _bad_request('invalid_outcome', allowed=sorted(valid))

    try:
        pred = Prediction.objects.get(pk=prediction_id)
    except Prediction.DoesNotExist:
        return JsonResponse({'error': 'not_found'}, status=404)

    pred.actual_outcome = outcome
    if notes:
        pred.notes = notes
    pred.save(update_fields=['actual_outcome', 'notes'])
    return JsonResponse({'ok': True, 'id': pred.id})


@require_http_methods(['GET'])
def healthz(request):
    """Lightweight liveness + model status check for monitoring."""
    model_loaded = ENSEMBLE_MODEL is not None
    metadata = ENSEMBLE_MODEL.metadata_ if model_loaded else None

    try:
        total_predictions = Prediction.objects.count()
        db_ok = True
    except Exception as exc:
        logger.warning("Healthz DB check failed: %s", exc)
        total_predictions = None
        db_ok = False

    status = 'ok' if model_loaded and db_ok else 'degraded'
    return JsonResponse({
        'status': status,
        'model_loaded': model_loaded,
        'model_metadata': metadata,
        'model_load_error': ENSEMBLE_LOAD_ERROR,
        'database_ok': db_ok,
        'total_predictions': total_predictions,
    }, status=200 if status == 'ok' else 503)
