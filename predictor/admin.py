from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        'created_at', 'mode', 'final_label', 'ik_label',
        'risk_score', 'latitude', 'longitude', 'actual_outcome',
    )
    list_filter = ('mode', 'final_label', 'actual_outcome')
    search_fields = ('wind_type', 'moon_phase', 'notes')
    readonly_fields = (
        'created_at', 'mode', 'wind_type', 'moon_phase', 'cloud_condition',
        'body_temperature', 'latitude', 'longitude', 'sci_temp',
        'sci_precipitation', 'sci_wind_speed', 'sci_humidity', 'ik_label',
        'final_label', 'risk_score', 'probabilities', 'model_version',
    )
    fields = readonly_fields + ('actual_outcome', 'notes')
