from django import forms

WIND_TYPES = [("Genya", "Genya"), ("Kus", "Kus"), ("Nyabukoba", "Nyabukoba"),
              ("Nyakoi", "Nyakoi"), ("Tarai", "Tarai"), ("Nyagire", "Nyagire"),
              ("Nyadhiwa", "Nyadhiwa"), ("Marimbe", "Marimbe")]

MOON_PHASES = [("New", "New"), ("Low", "Low"), ("Midway", "Midway"),
               ("Ascending", "Ascending"), ("Descending", "Descending"),
               ("Any Position", "Any Position")]

CLOUD_CONDITIONS = [("Clear", "Clear"), ("Light", "Light"),
                    ("Cloudy", "Cloudy"), ("Heavy", "Heavy")]

BODY_TEMPS = [("Cold", "Cold"), ("Moderate", "Moderate"),
              ("High", "High"), ("Warm", "Warm")]


class IndigenousForm(forms.Form):
    wind_type = forms.ChoiceField(choices=WIND_TYPES)
    moon_phase = forms.ChoiceField(choices=MOON_PHASES)
    cloud_condition = forms.ChoiceField(choices=CLOUD_CONDITIONS)
    body_temperature = forms.ChoiceField(choices=BODY_TEMPS)

class IntegratedForm(IndigenousForm):
    latitude = forms.FloatField(
        label="Latitude (Kabuto)",
        initial=-0.419,
        widget=forms.NumberInput(attrs={'readonly': 'readonly'})
    )
    longitude = forms.FloatField(
        label="Longitude (Kabuto)",
        initial=31.893,
        widget=forms.NumberInput(attrs={'readonly': 'readonly'})
    )
