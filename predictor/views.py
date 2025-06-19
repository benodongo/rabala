from django.shortcuts import render
from .forms import IndigenousForm, IntegratedForm
from .fuzzy_logic import determine_lake_position
from .scientific import get_scientific_weather, integrate_predictions

MOON_MAP = {"New": 0, "Low": 28, "Midway": 14, "Ascending": 5, "Descending": 22}
CLOUD_MAP = {"Clear": 0, "Light": 15, "Cloudy": 50, "Heavy": 100}
TEMP_MAP = {"Cold": 35, "Moderate": 36.5, "High": 38, "Warm": 37}

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

def landing(request):
      
    return render(request, 'landing.html')

def about(request):
      
    return render(request, 'about.html')

def how_it_works(request):
      
    return render(request, 'how_it_works.html')