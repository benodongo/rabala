from django import forms
 
# NOTE: "Genga" exists in the fuzzy rule base but is omitted here intentionally
# — it was not part of the original expert elicitation for user-facing input.
# If Genga is to be exposed, add ("Genga", "Genga") to this list.
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
    # FIX: "Any Position" removed from user-facing choices.
    # It is a fuzzy rule wildcard (internal), not a selectable moon phase.
    # A user cannot observe "Any Position" — they observe a specific phase.
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
    # FIX: "Hot" added to match v2 fuzzy rule base.
    ("Hot",      "Hot"),
]
 
 
class IndigenousForm(forms.Form):
    wind_type         = forms.ChoiceField(choices=WIND_TYPES,       label="Wind Type")
    moon_phase        = forms.ChoiceField(choices=MOON_PHASES,      label="Moon Phase")
    cloud_condition   = forms.ChoiceField(choices=CLOUD_CONDITIONS, label="Cloud Condition")
    body_temperature  = forms.ChoiceField(choices=BODY_TEMPS,       label="Body Temperature")

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
