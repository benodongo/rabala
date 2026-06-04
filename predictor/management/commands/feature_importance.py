import os
from django.conf import settings
from django.core.management.base import BaseCommand

from predictor.scientific import EnsembleModel


class Command(BaseCommand):
    help = "Inspect ensemble logistic-regression coefficients per feature."

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-path',
            type=str,
            default='predictor/models/ensemble_model.joblib',
            help='Path to the trained ensemble model',
        )

    def handle(self, *args, **options):
        path = os.path.join(settings.BASE_DIR, options['model_path'])
        if not os.path.exists(path):
            self.stdout.write(self.style.ERROR(f"Model not found at {path}"))
            return

        model = EnsembleModel.load(path)
        importance = model.get_feature_importance()

        if not importance:
            self.stdout.write(self.style.WARNING(
                "No coefficients available — model has no fitted ensemble."
            ))
            return

        meta = model.metadata_ or {}
        self.stdout.write(self.style.HTTP_INFO("Model metadata"))
        self.stdout.write(f"  schema_version : {meta.get('schema_version')}")
        self.stdout.write(f"  trained_at     : {meta.get('trained_at')}")
        self.stdout.write(f"  samples        : {meta.get('n_training_samples')}")
        self.stdout.write(f"  uses_posterior : {model.uses_posterior_}")
        self.stdout.write(f"  classes        : {list(model.ik_encoder.classes_)}")
        self.stdout.write("")

        sorted_items = sorted(
            importance.items(), key=lambda kv: abs(kv[1]), reverse=True
        )

        ik_total = sum(abs(v) for k, v in importance.items() if k.startswith('ik_'))
        sci_total = sum(abs(v) for k, v in importance.items() if k.startswith('sci_'))
        grand = ik_total + sci_total or 1.0

        self.stdout.write(self.style.HTTP_INFO("Aggregate |coefficient| share"))
        self.stdout.write(f"  IK side  : {ik_total / grand:.1%}")
        self.stdout.write(f"  Sci side : {sci_total / grand:.1%}")
        self.stdout.write("")

        self.stdout.write(self.style.HTTP_INFO("Per-feature coefficients (sorted by |coef|)"))
        width = max(len(k) for k in importance.keys())
        for name, coef in sorted_items:
            self.stdout.write(f"  {name.ljust(width)}  {coef:+.4f}")
