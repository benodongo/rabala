import os
import pandas as pd
import numpy as np
from django.conf import settings
import joblib
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from predictor.scientific import EnsembleModel
from predictor.fuzzy_logic import determine_lake_position

class Command(BaseCommand):
    help = 'Train and save the ensemble model using the specific data source format'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-path',
            type=str,
            default='predictor/data/fishing_data.csv',
            help='Path to the fishing data CSV file'
        )
        parser.add_argument(
            '--output-path',
            type=str,
            default='predictor/models/ensemble_model.joblib',
            help='Path to save the trained model'
        )
        parser.add_argument(
            '--resampling',
            type=str,
            choices=['none', 'oversample', 'undersample'],
            default='none',
            help='Resampling strategy to handle class imbalance'
        )

    def handle(self, *args, **options):
        # 1. Load and prepare data
        data_path = os.path.join(settings.BASE_DIR, options['data_path'])
        self.stdout.write(self.style.HTTP_INFO("Loading and preprocessing data..."))
        
        try:
            data = pd.read_csv(data_path)
            
            # Convert date to datetime and sort
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
            data = data.sort_values('Date')
            
            # Map IK indicators to model inputs
            moon_phase_map = {
                'New moon': 0,
                'Low Moon': 28,
                'Midway': 14,
                'Ascending': 5,
                'Descending': 22
            }
            
            cloud_map = {
                'Clear': 0,
                'Clear clouds': 15,
                'Light': 15,
                'Light clouds': 15,
                'Cloudy': 50,
                'Heavy Cloud': 100,
                'Heavy clouds': 100
            }
            
            temp_map = {
                'Cold': 35,
                'Cool': 36,
                'Warm': 37,
                'Hot': 38
            }
            
            # Prepare feature arrays
            X_ik = []
            for _, row in data.iterrows():
                X_ik.append([
                    row['Wind Circulation'],
                    moon_phase_map.get(row['Moon Phases'], 14),  # Default to Midway
                    cloud_map.get(row['Nimbus Clouds'], 50),      # Default to Cloudy
                    temp_map.get(row['Body feels at Night'], 37)  # Default to Warm
                ])
            
            # Prepare scientific features
            X_sci = data[[
                'Temp (Min)',
                'Temp (max)',
                'Rainfall (mm)',
                'Winds(m/s)'
            ]].values
            
            # Prepare targets (convert to consistent casing)
            y = data['Fishing Activity'].str.strip().str.title().values
            
            self.stdout.write(self.style.SUCCESS(f"Loaded {len(data)} records"))
            
            # Display class distribution
            class_dist = pd.Series(y).value_counts(normalize=True)
            self.stdout.write(f"\nClass Distribution:\n{class_dist}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading data: {e}"))
            return

        # 2. Train-test split (temporal split - last 20% for test)
        split_idx = int(len(data) * 0.8)
        X_ik_train, X_ik_test = X_ik[:split_idx], X_ik[split_idx:]
        X_sci_train, X_sci_test = X_sci[:split_idx], X_sci[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 3. Handle class imbalance
        self.stdout.write(self.style.HTTP_INFO("\nHandling class imbalance..."))
        
        # Calculate class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        self.stdout.write(f"Class weights: {class_weights}")
        
        # Apply resampling if specified
        if options['resampling'] == 'oversample':
            self.stdout.write("Applying SMOTE oversampling...")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_sci_train, y_train = smote.fit_resample(X_sci_train, y_train)
            
            # Align IK features by resampling accordingly
            new_indices = []
            for cls in np.unique(y_train):
                cls_count = sum(y_train == cls)
                original_indices = np.where(y[:split_idx] == cls)[0]
                new_indices.extend(np.random.choice(original_indices, size=cls_count, replace=True))
            
            X_ik_train = [X_ik_train[i] for i in new_indices]
            
        elif options['resampling'] == 'undersample':
            self.stdout.write("Applying random undersampling...")
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_sci_train, y_train = rus.fit_resample(X_sci_train, y_train)
            
            # Get the selected indices from the undersampler
            _, selected_indices = rus.fit_resample(np.arange(len(y_train)).reshape(-1, 1), y_train)
            X_ik_train = [X_ik_train[i] for i in selected_indices]

        # 4. Train Ensemble Model
        self.stdout.write(self.style.HTTP_INFO("\nTraining Ensemble Model..."))
        ensemble_model = EnsembleModel()
        
        # Get IK predictions
        ik_preds_train = [determine_lake_position(*params)[0] for params in X_ik_train]
        ik_preds_train = [p if p in ["Normal", "Risky", "Bad", "Good"] else "Normal" for p in ik_preds_train]
        
        # Train with class weights
        ensemble_model.train(X_ik_train, X_sci_train, y_train, ik_preds_train, class_weights=class_weights)
        
        # 5. Evaluate
        self.stdout.write(self.style.HTTP_INFO("Evaluating model..."))
        
        # Predict on test set
        y_pred = []
        y_proba = []  # For threshold adjustment
        
        for ik, sci in zip(X_ik_test, X_sci_test):
            try:
                pred, proba = ensemble_model.predict(ik, sci, return_proba=True)
                y_pred.append(pred)
                y_proba.append(proba)
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Prediction error: {e}"))
                y_pred.append("Normal")  # Default to majority class
                y_proba.append({"Normal": 1.0})  # Default probability

        # Standard evaluation
        self.stdout.write("\nStandard Evaluation:")
        self.stdout.write(classification_report(y_test, y_pred, zero_division=0))
        self.stdout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}")
        self.stdout.write(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
        
        # Threshold-adjusted evaluation (optional)
        self.stdout.write("\nThreshold-Adjusted Evaluation:")
        thresholds = {'Normal': 0.6, 'Good': 0.25, 'Risky': 0.25, 'Bad': 0.25}  # Customizable
        y_pred_adj = []
        for proba in y_proba:
            predicted = [cls for cls, thresh in thresholds.items() if proba.get(cls, 0) >= thresh]
            y_pred_adj.append(predicted[0] if predicted else "Normal")
        
        self.stdout.write(classification_report(y_test, y_pred_adj, zero_division=0))
        self.stdout.write(f"Adjusted Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_adj):.2f}")
        self.stdout.write(f"Adjusted Macro F1 Score: {f1_score(y_test, y_pred_adj, average='macro'):.2f}")

        # 6. Save model
        model_path = options['output_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            ensemble_model.save(model_path)
            self.stdout.write(self.style.SUCCESS(f"\nSuccessfully saved model to {model_path}"))
            
            # Print sample predictions
            sample_ik = X_ik_test[0]
            sample_sci = X_sci_test[0]
            sample_pred, sample_proba = ensemble_model.predict(sample_ik, sample_sci, return_proba=True)
            self.stdout.write(f"\nSample prediction: {sample_pred} (True: {y_test[0]})")
            self.stdout.write(f"Prediction probabilities: {sample_proba}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error saving model: {e}"))

        self.stdout.write(self.style.SUCCESS("\nTraining completed!"))