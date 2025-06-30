from django.shortcuts import render
from .forms import IndigenousForm, IntegratedForm
from .fuzzy_logic import determine_lake_position
from .scientific import get_scientific_weather, integrate_predictions, EnsembleModel
import os
from django.conf import settings
import numpy as np
import pandas as pd

MOON_MAP = {"New": 0, "Low": 28, "Midway": 14, "Ascending": 5, "Descending": 22}
CLOUD_MAP = {"Clear": 0, "Light": 15, "Cloudy": 50, "Heavy": 100}
TEMP_MAP = {"Cold": 35, "Moderate": 36.5, "High": 38, "Warm": 37}
# Pre-load ensemble model (put this at top of views.py)
ENSEMBLE_MODEL = None
try:
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'ensemble_model.joblib')
    if os.path.exists(model_path):
        ENSEMBLE_MODEL = EnsembleModel.load(model_path)
except Exception as e:
    print(f"Error loading ensemble model: {e}")

def indigenous_prediction(request):
    if request.method == 'POST':
        form = IndigenousForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            result = determine_lake_position(
                data['wind_type'],
                MOON_MAP[data['moon_phase']],
                CLOUD_MAP[data['cloud_condition']],
                TEMP_MAP[data['body_temperature']]
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
            # Indigenous prediction
            ik_result = determine_lake_position(
                data['wind_type'],
                MOON_MAP[data['moon_phase']],
                CLOUD_MAP[data['cloud_condition']],
                TEMP_MAP[data['body_temperature']]
            )
            # Scientific data
            sci_data = get_scientific_weather(data['latitude'], data['longitude'])
            # Integrated prediction
            prediction, risk = integrate_predictions(ik_result, sci_data)
            risk_percentage = risk * 100
            return render(request, 'integrated.html', {
                'form': form,
                'ik_result': ik_result,
                'sci_data': sci_data,
                'prediction': prediction,
                'risk': risk,
                'risk_percentage': risk_percentage,
            })
    else:
        form = IntegratedForm()
    return render(request, 'integrated.html', {'form': form})


def ensemble_prediction(request):
    if not ENSEMBLE_MODEL:
        return render(request, 'error.html', {'message': 'Ensemble model not loaded'})

    form = IntegratedForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data

        # 1. Prepare indigenous knowledge inputs
        ik_inputs = (
            str(data['wind_type']),
            int(MOON_MAP.get(data['moon_phase'], 14)),
            int(CLOUD_MAP.get(data['cloud_condition'], 50)),
            int(TEMP_MAP.get(data['body_temperature'], 37))
        )

        # 2. Prepare scientific features
        sci_data = get_scientific_weather(data['latitude'], data['longitude']) or {}
        temp_value = float(sci_data.get('temp', 27.5))
        
        sci_features = np.array([
            [
                float(sci_data.get('temp_min', temp_value - 2.5)),
                float(sci_data.get('temp_max', temp_value + 2.5)),
                float(sci_data.get('precipitation', 0.0)),
                float(sci_data.get('wind_speed', 5.0))
            ]
        ], dtype=np.float32)

        # 3. Make prediction
        #ensemble_pred = ENSEMBLE_MODEL.predict(ik_inputs, sci_features[0])
        model = ENSEMBLE_MODEL
        model.set_class_thresholds({'Normal': 0.6, 'Good': 0.3, 'Risky': 0.3, 'Bad': 0.3})
        ensemble_pred = model.predict(ik_inputs, sci_features[0], adjust_thresholds=True)

        # 4. Format prediction output
        ensemble_pred_label = str(ensemble_pred[0] if isinstance(ensemble_pred, (np.ndarray, list)) else ensemble_pred)

        # 5. Map prediction to risk information
        risk_mapping = {
            'Good': (0.2, "Excellent fishing conditions 🌞"),
            'Normal': (0.4, "Normal fishing conditions ⛅"),
            'Risky': (0.7, "Caution advised fishing conditions ⚠️"),
            'Bad': (1.0, "Dangerous fishing conditions 🛑")
        }
        risk_value, risk_str = risk_mapping.get(ensemble_pred_label, (0.5, "Uncertain conditions"))

        return render(request, 'ensemble.html', {
            'form': form,
            'ensemble_pred': ensemble_pred_label,
            'ensemble_str': risk_str,
            'ensemble_risk': risk_value
        })

    return render(request, 'ensemble.html', {'form': form})

def landing(request):
      
    return render(request, 'landing.html')

def about(request):
      
    return render(request, 'about.html')

def how_it_works(request):
      
    return render(request, 'how_it_works.html')