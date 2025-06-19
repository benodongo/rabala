from collections import defaultdict
def triangular(x, a, b, c):
    if a == b == c:
        return 1.0 if x == a else 0.0
    if b == a:
        return max(0.0, min(1.0, (c - x)/(c - b)))
    if b == c:
        return max(0.0, min(1.0, (x - a)/(b - a)))
    return max(0.0, min((x - a)/(b - a), (c - x)/(c - b)))

def moon_membership(moon_day):
    return {
        'New': triangular(moon_day, 0, 0, 2),
        'Ascending': triangular(moon_day, 1, 5, 9),
        'Midway': triangular(moon_day, 8, 14, 20),
        'Descending': triangular(moon_day, 18, 22, 26),
        'Low': triangular(moon_day, 24, 28, 29)
    }

def cloud_membership(cloud_percent):
    return {
        'Light': triangular(cloud_percent, 0, 0, 30),
        'Cloudy': triangular(cloud_percent, 20, 50, 80),
        'Heavy': triangular(cloud_percent, 70, 100, 100)
    }

def body_temp_membership(temp):
    return {
        'Low': triangular(temp, 35, 35, 36),
        'Moderate': triangular(temp, 35.5, 36.5, 37.5),
        'High': triangular(temp, 37, 38, 40)
    }

def determine_lake_position(wind_type, moon_day, cloud_percent, body_temp_value):
    # Fuzzify numerical inputs
    moon_mem = moon_membership(moon_day)
    cloud_mem = cloud_membership(cloud_percent)
    body_mem = body_temp_membership(body_temp_value)

    # Initialize output accumulators
    lake_pos_counts = defaultdict(float)
    wind_out_counts = defaultdict(float)
    temp_out_counts = defaultdict(float)
    rainfall_out_counts = defaultdict(float)

    rules = [
        # Risky Lake Position Rules
        ({"Genya"}, {"Descending"}, {"Heavy"}, {"Moderate"}, "Risky", "High", "Very Heavy", "Risky"),
        ({"Genya", "Kus", "Nyabukoba"}, {"Descending"}, {"Heavy"}, {"Moderate"}, "Risky", "High", "Very Heavy", "Risky"),
        ({"Genya", "Kus", "Nyabukoba"}, {"Low"}, {"Heavy"}, {"Low"}, "Risky", "Low", "Very Heavy", "Risky"),
        ({"Nyakoi"}, {"New", "Low"}, {"Heavy"}, {"Low"}, "Risky", "Low", "Very Heavy", "Risky"),
        ({"Genya"}, {"New", "Low"}, {"Heavy"}, {"Moderate"}, "Risky", "Moderate", "Heavy", "Risky"),
        ({"Nyakoi"}, {"New", "Low"}, {"Heavy"}, {"Moderate"}, "Risky", "Moderate", "Heavy", "Bad"),
        ({"Nyakoi"}, {"Any Position"}, {"Light"}, {"Any"}, "Risky", "Moderate", "Moderate", "Bad"),
        ({"Kus", "Genya", "Tarai"}, {"New", "Midway"}, {"Heavy"}, {"Moderate"}, "Risky", "Moderate", "Moderate", "Bad"),
        ({"Kus", "Genya", "Tarai", "Nyagire", "Nyabukoba"}, {"Any Position"}, {"Cloudy"}, {"Moderate"}, "Windy", "High", "Heavy", "Bad"),
        ({"Genya"}, {"Descending"}, {"Light"}, {"Moderate"}, "Risky", "Moderate", "Moderate", "Bad"),
        ({"Tarai"}, {"Descending"}, {"Light"}, {"High"}, "Risky", "High", "Low", "Bad"),
        ({"Kus", "Genya"}, {"Low"}, {"Cloudy"}, {"High"}, "Risky", "High", "Low", "Bad"),
        ({"Kus", "Genya"}, {"Ascending"}, {"Cloudy"}, {"Moderate"}, "Risky", "High", "Low", "Bad"),
        ({"Kus", "Nyagire"}, {"Midway"}, {"Cloudy"}, {"Moderate"}, "Risky", "Moderate", "Heavy", "Bad"),
        ({"Kus"}, {"Low"}, {"Heavy"}, {"Moderate"}, "Windy", "Moderate", "Moderate", "Bad"),
        ({"Nyagire"}, {"New", "Ascending"}, {"Cloudy"}, {"Moderate"}, "Risky", "Moderate", "Moderate", "Bad"),
        ({"Nyadhiwa"}, {"Ascending", "Descending"}, {"Cloudy"}, {"High"}, "Windy", "High", "Moderate", "Bad"),

        # Good Lake Position Rules
        ({"Kus"}, {"Ascending"}, {"Clear"}, {"Moderate"}, "Normal", "Moderate", "No", "Good"),
        ({"Kus"}, {"Descending", "Low"}, {"Cloudy"}, {"High"}, "Windy", "High", "Light", "Good"),
        ({"Kus"}, {"Descending", "Low"}, {"Clear"}, {"High"}, "Normal", "Moderate", "No", "Good"),
        ({"Kus"}, {"Midway", "Ascending"}, {"Cloudy"}, {"Moderate"}, "Windy", "Moderate", "Light", "Good"),
        ({"Genya"}, {"New"}, {"Heavy"}, {"Moderate"}, "Normal", "Moderate", "Moderate", "Good"),
        ({"Genya"}, {"Midway"}, {"Clear"}, {"Moderate"}, "Normal", "Moderate", "No", "Good"),
        ({"Genya"}, {"Descending"}, {"Clear"}, {"High"}, "Windy", "Moderate", "No", "Good"),

        # Normal Lake Position Rules
        ({"Nyadhiwa"}, {"Descending"}, {"Clear"}, {"Cold"}, "Normal", "Moderate", "No", "Normal"),
        ({"Nyadhiwa"}, {"Ascending"}, {"Light", "Cloudy"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Nyadhiwa"}, {"New"}, {"Light"}, {"High"}, "Windy", "High", "No", "Normal"),
        ({"Nyadhiwa"}, {"New"}, {"Cloudy"}, {"Moderate"}, "Windy", "Moderate", "Light", "Normal"),
        ({"Nyadhiwa"}, {"Midway"}, {"Clear"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Nyadhiwa"}, {"Midway"}, {"Light"}, {"Moderate"}, "Windy", "High", "Light", "Normal"),
        ({"Nyadhiwa"}, {"Low"}, {"Clear", "Light"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Nyadhiwa"}, {"Low"}, {"Cloudy"}, {"Moderate"}, "Windy", "Moderate", "Moderate", "Normal"),
        ({"Marimbe"}, {"Low"}, {"Clear"}, {"High"}, "Windy", "High", "No", "Normal"),
        ({"Marimbe"}, {"Low"}, {"Light"}, {"Moderate"}, "Windy", "Warm", "Light", "Normal"),
        ({"Marimbe"}, {"Low"}, {"Cloudy"}, {"Moderate"}, "Windy", "Warm", "Moderate", "Normal"),
        ({"Marimbe"}, {"Midway"}, {"Light"}, {"Moderate"}, "Windy", "High", "Light", "Normal"),
        ({"Marimbe"}, {"Descending"}, {"Light"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Marimbe"}, {"New", "Ascending"}, {"Cloudy"}, {"Moderate"}, "Normal", "High", "No", "Normal"),
        ({"Marimbe"}, {"Descending"}, {"Light"}, {"Moderate"}, "Windy", "Warm", "Light", "Normal"),
        ({"Genya"}, {"New", "Low"}, {"Light"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Genya"}, {"New", "Low"}, {"Cloudy"}, {"Moderate"}, "Windy", "Moderate", "Moderate", "Normal"),
        ({"Genya"}, {"Ascending", "Midway", "Descending"}, {"Clear"}, {"Moderate"}, "Windy", "Moderate", "No", "Normal"),
        ({"Genya"}, {"Ascending", "Midway", "Descending"}, {"Heavy"}, {"Moderate"}, "Windy", "Moderate", "Heavy", "Normal"),
        ({"Kus"}, {"New", "Low"}, {"Clear"}, {"Moderate"}, "Windy", "Moderate", "Light", "Normal"),
        ({"Kus"}, {"New", "Low"}, {"Heavy"}, {"Moderate"}, "Windy", "Moderate", "Heavy", "Normal"),
        ({"Kus"}, {"Ascending", "Midway", "Descending"}, {"Clear"}, {"Moderate"}, "Windy", "High", "No", "Normal"),
        ({"Kus"}, {"Ascending", "Midway", "Descending"}, {"Cloudy"}, {"High"}, "Windy", "High", "Moderate", "Normal"),
        ({"Tarai"}, {"New", "Low"}, {"Light"}, {"Moderate"}, "Normal", "High", "No", "Normal"),
        ({"Tarai"}, {"New", "Low"}, {"Cloudy"}, {"High"}, "Windy", "Moderate", "Moderate", "Normal"),
        ({"Tarai"}, {"Ascending", "Midway", "Descending"}, {"Clear"}, {"Moderate"}, "Windy", "Moderate", "No", "Normal"),
        ({"Tarai"}, {"Ascending", "Midway", "Descending"}, {"Cloudy"}, {"High"}, "Windy", "High", "Moderate", "Normal"),
    ]

    for rule in rules:
        wind_set, moon_set, cloud_set, body_temp_set, wind_out, temp_out, rainfall_out, lake_pos = rule

        # Check wind condition
        if 'Any' in wind_set:
            wind_ok = True
        else:
            wind_ok = wind_type in wind_set
        if not wind_ok:
            continue

        # Compute moon membership
        if 'Any Position' in moon_set:
            moon_str = 1.0
        else:
            moon_str = max([moon_mem.get(m, 0.0) for m in moon_set])

        # Compute cloud membership
        if 'Any' in cloud_set:
            cloud_str = 1.0
        else:
            cloud_str = max([cloud_mem.get(c, 0.0) for c in cloud_set])

        # Compute body_temp membership
        if 'Any' in body_temp_set:
            body_str = 1.0
        else:
            body_str = max([body_mem.get(bt, 0.0) for bt in body_temp_set])

        firing = min(moon_str, cloud_str, body_str)
        if firing <= 0:
            continue

        # Accumulate firing strengths
        lake_pos_counts[lake_pos] += firing
        wind_out_counts[wind_out] += firing
        temp_out_counts[temp_out] += firing
        rainfall_out_counts[rainfall_out] += firing

    def get_max_category(counts):
        if not counts:
            return "Unknown"
        return max(counts, key=lambda k: counts[k])

    lake_pos = get_max_category(lake_pos_counts)
    wind_out = get_max_category(wind_out_counts)
    temp_out = get_max_category(temp_out_counts)
    rainfall_out = get_max_category(rainfall_out_counts)

    return (lake_pos, wind_out, temp_out, rainfall_out)