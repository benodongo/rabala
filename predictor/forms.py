from django import forms

# "Genga" exists in the fuzzy rule base but is intentionally omitted from the
# user-facing list — it wasn't part of the original expert elicitation. Add it
# back here if you want to expose it.
WIND_TYPES = [
    ("Genya",     "Genya"),
    ("Kus",       "Kus"),
    ("Nyabukoba", "Nyabukoba"),
    ("Nyakoi",    "Nyakoi"),
    ("Tarai",     "Tarai"),
    ("Nyagire",   "Nyagire"),
    ("Nyadhiwa",  "Nyadhiwa"),
    ("Marimbe",   "Marimbe"),
]

MOON_PHASES = [
    ("New",        "New"),
    ("Ascending",  "Ascending"),
    ("Midway",     "Midway"),
    ("Descending", "Descending"),
    ("Low",        "Low"),
]

CLOUD_CONDITIONS = [
    ("Clear",  "Clear"),
    ("Light",  "Light"),
    ("Cloudy", "Cloudy"),
    ("Heavy",  "Heavy"),
]

BODY_TEMPS = [
    ("Cold",     "Cold"),
    ("Cool",     "Cool"),
    ("Low",      "Low"),
    ("Moderate", "Moderate"),
    ("Warm",     "Warm"),
    ("High",     "High"),
    ("Hot",      "Hot"),
]

# Pre-set fishing communities around Lake Victoria. Users can pick one or
# enter custom coordinates.
NAMED_LOCATIONS = [
    ("kabuto",  "Kabuto (-0.419, 31.893)"),
    ("kisumu",  "Kisumu (-0.0917, 34.7680)"),
    ("mwanza",  "Mwanza (-2.5164, 32.9175)"),
    ("entebbe", "Entebbe (0.0500, 32.4600)"),
    ("custom",  "Custom coordinates"),
]

LOCATION_PRESETS = {
    "kabuto":  (-0.419, 31.893),
    "kisumu":  (-0.0917, 34.7680),
    "mwanza":  (-2.5164, 32.9175),
    "entebbe": (0.0500, 32.4600),
}


_INPUT_CLS = "form-control field-input"
_SELECT_CLS = "form-select field-select"


class IndigenousForm(forms.Form):
    wind_type        = forms.ChoiceField(choices=WIND_TYPES,       label="Wind Type",
                                         widget=forms.Select(attrs={'class': _SELECT_CLS}))
    moon_phase       = forms.ChoiceField(choices=MOON_PHASES,      label="Moon Phase",
                                         widget=forms.Select(attrs={'class': _SELECT_CLS}))
    cloud_condition  = forms.ChoiceField(choices=CLOUD_CONDITIONS, label="Cloud Condition",
                                         widget=forms.Select(attrs={'class': _SELECT_CLS}))
    body_temperature = forms.ChoiceField(choices=BODY_TEMPS,       label="Body Temperature",
                                         widget=forms.Select(attrs={'class': _SELECT_CLS}))


class IntegratedForm(IndigenousForm):
    location = forms.ChoiceField(
        choices=NAMED_LOCATIONS,
        label="Location",
        initial="kabuto",
        required=True,
        widget=forms.Select(attrs={'class': _SELECT_CLS}),
    )
    latitude = forms.FloatField(
        label="Latitude",
        initial=-0.419,
        min_value=-90.0,
        max_value=90.0,
        required=True,
        widget=forms.NumberInput(attrs={'class': _INPUT_CLS, 'step': '0.0001'}),
    )
    longitude = forms.FloatField(
        label="Longitude",
        initial=31.893,
        min_value=-180.0,
        max_value=180.0,
        required=True,
        widget=forms.NumberInput(attrs={'class': _INPUT_CLS, 'step': '0.0001'}),
    )

    def clean(self):
        cleaned = super().clean()
        loc = cleaned.get('location')
        if loc and loc != 'custom' and loc in LOCATION_PRESETS:
            lat, lon = LOCATION_PRESETS[loc]
            cleaned['latitude'] = lat
            cleaned['longitude'] = lon
        return cleaned
