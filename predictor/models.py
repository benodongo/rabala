from django.db import models


class Prediction(models.Model):
    MODE_CHOICES = [
        ('indigenous', 'Indigenous'),
        ('integrated', 'Integrated'),
        ('ensemble',   'Ensemble'),
        ('api',        'API'),
    ]

    LABEL_CHOICES = [
        ('Good',   'Good'),
        ('Normal', 'Normal'),
        ('Risky',  'Risky'),
        ('Bad',    'Bad'),
    ]

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    mode = models.CharField(max_length=16, choices=MODE_CHOICES, db_index=True)

    wind_type        = models.CharField(max_length=32)
    moon_phase       = models.CharField(max_length=32)
    cloud_condition  = models.CharField(max_length=32)
    body_temperature = models.CharField(max_length=32)

    latitude  = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    sci_temp          = models.FloatField(null=True, blank=True)
    sci_precipitation = models.FloatField(null=True, blank=True)
    sci_wind_speed    = models.FloatField(null=True, blank=True)
    sci_humidity      = models.FloatField(null=True, blank=True)

    ik_label       = models.CharField(max_length=16, blank=True, default='')
    final_label    = models.CharField(max_length=16, choices=LABEL_CHOICES, db_index=True)
    risk_score     = models.FloatField(null=True, blank=True)
    probabilities  = models.JSONField(null=True, blank=True)
    model_version  = models.CharField(max_length=64, blank=True, default='')

    actual_outcome = models.CharField(
        max_length=16, choices=LABEL_CHOICES, blank=True, default='',
        help_text="Reported outcome from the fisherman (optional feedback).",
    )
    notes = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['mode', 'created_at']),
        ]

    def __str__(self):
        return f"{self.mode}:{self.final_label} @ {self.created_at:%Y-%m-%d %H:%M}"
