"""Smoke tests for fuzzy logic, the scientific weather client, and the views.

Run with: python manage.py test predictor
"""
import json
from unittest.mock import patch

import numpy as np
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from predictor.fuzzy_logic import determine_lake_position
from predictor.models import Prediction
from predictor.scientific import (
    RISK_MAPPING,
    ScientificModel,
    get_scientific_weather,
    integrate_predictions,
)


class FuzzyLogicTests(TestCase):
    """Lock in the rule-base behaviour against known-good cases."""

    def test_returns_four_string_outputs(self):
        result = determine_lake_position("Genya", 22, 85, 37.0)
        self.assertEqual(len(result), 4)
        for item in result:
            self.assertIsInstance(item, str)
            self.assertNotEqual(item, '')

    def test_label_in_known_classes(self):
        label, *_ = determine_lake_position("Kus", 2, 10, 37.0)
        self.assertIn(label, {'Good', 'Normal', 'Risky', 'Bad'})

    def test_extreme_hot_heavy_cloud_is_classified(self):
        label, *_ = determine_lake_position("Nyagire", 5, 90, 39.5)
        self.assertIn(label, {'Good', 'Normal', 'Risky', 'Bad'})


class ScientificModelTests(TestCase):
    """Tiny synthetic dataset round-trip — proves train/predict shape."""

    def setUp(self):
        rng = np.random.default_rng(seed=0)
        self.X = rng.normal(size=(40, 4))
        self.y = np.array(['Good'] * 10 + ['Normal'] * 10
                          + ['Risky'] * 10 + ['Bad'] * 10)

    def test_train_accepts_sample_weight(self):
        model = ScientificModel()
        weights = np.ones(len(self.y))
        model.train(self.X, self.y, sample_weight=weights)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_predict_proba_sums_to_one(self):
        model = ScientificModel()
        model.train(self.X, self.y)
        probs = model.predict_proba(self.X[:3])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class IntegratePredictionsTests(TestCase):
    def test_no_sci_data_uses_fallback(self):
        verdict, risk = integrate_predictions(('Good',), None)
        self.assertIsInstance(verdict, str)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)

    def test_calm_conditions_score_low(self):
        sci = {'precipitation': 0.0, 'wind_speed': 0.0}
        _, risk_calm = integrate_predictions(('Good',), sci)
        sci_storm = {'precipitation': 50.0, 'wind_speed': 15.0}
        _, risk_storm = integrate_predictions(('Bad',), sci_storm)
        self.assertLess(risk_calm, risk_storm)


class WeatherFetcherTests(TestCase):
    @patch('predictor.scientific.requests.get')
    def test_returns_none_on_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException('boom')
        self.assertIsNone(
            get_scientific_weather(-0.4, 31.9, use_cache=False)
        )

    @patch('predictor.scientific.requests.get')
    def test_parses_payload(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            'current': {
                'temperature_2m': 27.0, 'relative_humidity_2m': 70,
                'precipitation': 0.1, 'cloud_cover': 40,
                'wind_speed_10m': 4.0,
            },
            'daily': {
                'temperature_2m_min': [22.0], 'temperature_2m_max': [29.5],
                'precipitation_sum': [0.3], 'wind_speed_10m_max': [6.5],
            },
        }
        result = get_scientific_weather(-0.4, 31.9, use_cache=False)
        self.assertEqual(result['temp_min'], 22.0)
        self.assertEqual(result['wind_max'], 6.5)


@override_settings(
    CACHES={'default': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'}}
)
class ViewSmokeTests(TestCase):
    """Make sure the form-handling paths render and persist predictions."""

    def setUp(self):
        self.client = Client()

    def test_landing_renders(self):
        self.assertEqual(self.client.get(reverse('landing')).status_code, 200)

    def test_indigenous_post_logs_prediction(self):
        before = Prediction.objects.count()
        resp = self.client.post(reverse('indigenous'), data={
            'wind_type': 'Kus',
            'moon_phase': 'New',
            'cloud_condition': 'Clear',
            'body_temperature': 'Moderate',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Prediction.objects.count(), before + 1)
        pred = Prediction.objects.latest('created_at')
        self.assertEqual(pred.mode, 'indigenous')
        self.assertIn(pred.final_label, {'Good', 'Normal', 'Risky', 'Bad'})

    @patch('predictor.views.get_scientific_weather')
    def test_integrated_post_handles_missing_weather(self, mock_weather):
        mock_weather.return_value = None
        resp = self.client.post(reverse('integrated'), data={
            'wind_type': 'Genya', 'moon_phase': 'Midway',
            'cloud_condition': 'Cloudy', 'body_temperature': 'Warm',
            'location': 'kabuto',
            'latitude': '-0.419', 'longitude': '31.893',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Prediction.objects.filter(mode='integrated').count(), 1)

    def test_healthz_returns_json(self):
        resp = self.client.get(reverse('healthz'))
        self.assertIn(resp.status_code, (200, 503))
        body = json.loads(resp.content)
        self.assertIn('model_loaded', body)
        self.assertIn('database_ok', body)


class RiskMappingTests(TestCase):
    def test_all_labels_have_entries(self):
        for label in ('Good', 'Normal', 'Risky', 'Bad'):
            self.assertIn(label, RISK_MAPPING)
            risk, msg = RISK_MAPPING[label]
            self.assertIsInstance(risk, float)
            self.assertIsInstance(msg, str)
