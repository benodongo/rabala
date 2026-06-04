"""Sweep per-class decision thresholds to maximise macro-F1 on a validation split.

Usage:
    python manage.py tune_thresholds \\
        --data-path predictor/data/fishing_data.csv \\
        --model-path predictor/models/ensemble_model.joblib
"""
import itertools
import os

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.metrics import f1_score

from predictor.scientific import EnsembleModel


class Command(BaseCommand):
    help = 'Grid-search per-class thresholds that maximise validation macro-F1.'

    def add_arguments(self, parser):
        parser.add_argument('--data-path', type=str,
                            default='predictor/data/fishing_data.csv')
        parser.add_argument('--model-path', type=str,
                            default='predictor/models/ensemble_model.joblib')
        parser.add_argument('--grid', type=float, nargs='+',
                            default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            help='Threshold candidates to sweep per class.')
        parser.add_argument('--save', action='store_true',
                            help='Persist best thresholds back to the model file.')

    def handle(self, *args, **options):
        model_path = options['model_path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(settings.BASE_DIR, model_path)
        data_path = options['data_path']
        if not os.path.isabs(data_path):
            data_path = os.path.join(settings.BASE_DIR, data_path)

        self.stdout.write(self.style.HTTP_INFO(f"Loading model from {model_path}"))
        model = EnsembleModel.load(model_path)

        self.stdout.write(self.style.HTTP_INFO(f"Loading validation data from {data_path}"))
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data = data.sort_values('Date')

        moon_map = {'New moon': 0, 'Low Moon': 28, 'Midway': 14,
                    'Ascending': 5, 'Descending': 22}
        cloud_map = {'Clear': 0, 'Clear clouds': 15, 'Light': 15,
                     'Light clouds': 15, 'Cloudy': 50,
                     'Heavy Cloud': 100, 'Heavy clouds': 100}
        temp_map = {'Cold': 35, 'Cool': 36, 'Warm': 37, 'Hot': 38}

        X_ik, X_sci, y = [], [], []
        for _, row in data.iterrows():
            X_ik.append([
                row['Wind Circulation'],
                moon_map.get(row['Moon Phases'], 14),
                cloud_map.get(row['Nimbus Clouds'], 50),
                temp_map.get(row['Body feels at Night'], 37),
            ])
            X_sci.append([row['Temp (Min)'], row['Temp (max)'],
                          row['Rainfall (mm)'], row['Winds(m/s)']])
            y.append(str(row['Fishing Activity']).strip().title())

        # Use the last 20% as the tuning set — same convention as train_model.
        split_idx = int(len(data) * 0.8)
        X_ik_val = X_ik[split_idx:]
        X_sci_val = np.array(X_sci[split_idx:])
        y_val = np.array(y[split_idx:])

        self.stdout.write(f"Tuning on {len(y_val)} validation samples")

        # Cache calibrated probabilities once.
        proba_cache = []
        for ik, sci in zip(X_ik_val, X_sci_val):
            _, p = model.predict(ik, sci, return_proba=True)
            proba_cache.append(p)

        classes = list(model.ik_encoder.classes_)
        grid = options['grid']
        best = {'score': -1.0, 'thresholds': None}

        for combo in itertools.product(grid, repeat=len(classes)):
            thresholds = dict(zip(classes, combo))
            preds = []
            for p in proba_cache:
                qualifying = [c for c in classes if p.get(c, 0) >= thresholds[c]]
                if qualifying:
                    preds.append(max(qualifying, key=lambda c: p[c]))
                else:
                    preds.append(max(classes, key=lambda c: p.get(c, 0)))
            score = f1_score(y_val, preds, average='macro', zero_division=0)
            if score > best['score']:
                best = {'score': score, 'thresholds': thresholds}

        self.stdout.write(self.style.SUCCESS(
            f"\nBest macro-F1: {best['score']:.4f}"
        ))
        self.stdout.write(f"Best thresholds: {best['thresholds']}")

        if options['save'] and best['thresholds']:
            model.set_class_thresholds(best['thresholds'])
            model.save(model_path)
            self.stdout.write(self.style.SUCCESS(
                f"Saved tuned thresholds back to {model_path}"
            ))
