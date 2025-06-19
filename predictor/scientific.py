import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_scientific_weather(lat, lon):
    """Fetch real-time weather data from Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m"
    try:
        response = requests.get(url)
        data = response.json()['current']
        return {
            'temp': data['temperature_2m'],
            'humidity': data['relative_humidity_2m'],
            'precipitation': data['precipitation'],
            'cloud_cover': data['cloud_cover'],
            'wind_speed': data['wind_speed_10m']
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None
    
class ScientificModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
    
def integrate_predictions(indigenous_pred, scientific_data):
    """Combine both prediction systems using adaptive weighting"""
    risk_mapping = {'Good': 0, 'Normal': 0.4, 'Risky': 0.7, 'Bad': 1.0}
    indigenous_risk = risk_mapping.get(indigenous_pred[0], 0.5)

    if scientific_data:
        sci_risk = (scientific_data['precipitation']/50 * 0.6 +
                   scientific_data['wind_speed']/15 * 0.4)
    else:
        sci_risk = 0.5  # Fallback value

    combined_risk = (indigenous_risk * 0.6 + sci_risk * 0.4)

    if combined_risk < 0.3:
        return "Excellent fishing conditions 🌞", combined_risk
    elif combined_risk < 0.6:
        return "Normal fishing conditions ⛅", combined_risk
    else:
        return "Dangerous fishing conditions ⚠️", combined_risk