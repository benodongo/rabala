"""URL configuration for Rabala project."""
from django.contrib import admin
from django.urls import path

from predictor import api, views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.landing, name='landing'),
    path('indigenous/', views.indigenous_prediction, name='indigenous'),
    path('integrated/', views.integrated_prediction, name='integrated'),
    path('ensemble/', views.ensemble_prediction, name='ensemble'),
    path('about/', views.about, name='about'),
    path('how-it-works/', views.how_it_works, name='how_it_works'),

    # JSON / monitoring endpoints.
    path('api/predict/', api.predict_api, name='api_predict'),
    path('api/feedback/<int:prediction_id>/', api.feedback_api, name='api_feedback'),
    path('healthz/', api.healthz, name='healthz'),
]
