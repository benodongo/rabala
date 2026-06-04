from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('mode', models.CharField(choices=[('indigenous', 'Indigenous'), ('integrated', 'Integrated'), ('ensemble', 'Ensemble'), ('api', 'API')], db_index=True, max_length=16)),
                ('wind_type', models.CharField(max_length=32)),
                ('moon_phase', models.CharField(max_length=32)),
                ('cloud_condition', models.CharField(max_length=32)),
                ('body_temperature', models.CharField(max_length=32)),
                ('latitude', models.FloatField(blank=True, null=True)),
                ('longitude', models.FloatField(blank=True, null=True)),
                ('sci_temp', models.FloatField(blank=True, null=True)),
                ('sci_precipitation', models.FloatField(blank=True, null=True)),
                ('sci_wind_speed', models.FloatField(blank=True, null=True)),
                ('sci_humidity', models.FloatField(blank=True, null=True)),
                ('ik_label', models.CharField(blank=True, default='', max_length=16)),
                ('final_label', models.CharField(choices=[('Good', 'Good'), ('Normal', 'Normal'), ('Risky', 'Risky'), ('Bad', 'Bad')], db_index=True, max_length=16)),
                ('risk_score', models.FloatField(blank=True, null=True)),
                ('probabilities', models.JSONField(blank=True, null=True)),
                ('model_version', models.CharField(blank=True, default='', max_length=64)),
                ('actual_outcome', models.CharField(blank=True, choices=[('Good', 'Good'), ('Normal', 'Normal'), ('Risky', 'Risky'), ('Bad', 'Bad')], default='', max_length=16)),
                ('notes', models.TextField(blank=True, default='')),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['mode', 'created_at'], name='predictor_p_mode_created_idx')],
            },
        ),
    ]
