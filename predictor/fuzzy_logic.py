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
        'New': triangular(moon_day, 0, 0, 3),
        'Ascending': triangular(moon_day, 1, 5, 10),
        'Midway': triangular(moon_day, 8, 14, 21),
        'Descending': triangular(moon_day, 18, 22, 27),
        'Low': triangular(moon_day, 24, 28, 29)
    }

def cloud_membership(cloud_percent):
    return {
        'Clear': triangular(cloud_percent, 0, 0, 20),
        'Light': triangular(cloud_percent, 10, 25, 40),
        'Cloudy': triangular(cloud_percent, 30, 50, 70),
        'Heavy': triangular(cloud_percent, 60, 80, 100)
    }

def body_temp_membership(temp):
    return {
        'Cold': triangular(temp, 34, 34.5, 35.5),
        'Cool': triangular(temp, 35, 36, 37),
        'Low': triangular(temp, 35, 36, 37),  # Same as Cool
        'Moderate': triangular(temp, 36.5, 37, 37.5),
        'Warm': triangular(temp, 37, 38, 39),
        'High': triangular(temp, 38, 39, 40)
    }

def parse_rule_items(item_str):
    """Parse semicolon-separated items into a set"""
    if not item_str or item_str.strip() == "":
        return set()
    items = [i.strip() for i in item_str.split(';') if i.strip()]
    return set(items)

def clean_rule_sets(rules):
    """Clean up rule sets with semicolons and inconsistencies"""
    cleaned_rules = []
    for rule in rules:
        wind_str, moon_str, cloud_str, body_temp_str, wind_out, temp_out, rainfall_out, lake_pos = rule
        
        # Parse wind types
        wind_set = parse_rule_items(wind_str)
        
        # Parse moon phases
        moon_set = parse_rule_items(moon_str)
        
        # Parse cloud conditions
        cloud_set = parse_rule_items(cloud_str)
        
        # Parse body temperatures
        body_temp_set = parse_rule_items(body_temp_str)
        
        cleaned_rules.append((wind_set, moon_set, cloud_set, body_temp_set, 
                              wind_out, temp_out, rainfall_out, lake_pos))
    
    return cleaned_rules

def determine_lake_position(wind_type, moon_day, cloud_percent, body_temp_value):
    # Fuzzify numerical inputs
    moon_mem = moon_membership(moon_day)
    cloud_mem = cloud_membership(cloud_percent)
    body_mem = body_temp_membership(body_temp_value)
    
    # Initialize output MAX trackers
    lake_pos_max = defaultdict(float)
    wind_out_max = defaultdict(float)
    temp_out_max = defaultdict(float)
    rainfall_out_max = defaultdict(float)
    
    # Define ALL rules from your list
    rules = [
        # Format: (wind_str, moon_str, cloud_str, body_temp_str, wind_out, temp_out, rainfall_out, lake_pos)
        
        # Row 1
        ("Genya", "Descending", "Heavy", "Moderate", "Risky", "High", "Very Heavy", "Risky"),
        ("Genya", "New", "Heavy", "Cool", "Windy", "Warm", "Heavy", "Risky"),
        ("Genya", "Ascending", "Heavy", "Cool", "Windy", "Warm", "Heavy", "Bad"),
        
        # Row 4
        ("Genya;Kus;Nyabukoba", "Descending", "Heavy", "Moderate", "Risky", "High", "Very Heavy", "Risky"),
        
        # Row 5
        ("Genya;Kus;Nyabukoba", "Low", "Heavy", "Low", "Risky", "Low", "Very Heavy", "Risky"),
        
        # Row 6
        ("Nyakoi", "New;Low", "Heavy", "Low", "Risky", "Low", "Very Heavy", "Risky"),
        
        # Row 7
        ("Genga", "Low", "Heavy", "Moderate", "Risky", "Moderate", "Heavy", "Risky"),
        
        # Row 8
        ("Genga", "New", "Heavy", "Moderate", "Risky", "Moderate", "Moderate", "Risky"),
        
        # Row 9
        ("Nyakoi", "New;Low", "Heavy", "Moderate", "Risky", "Moderate", "Heavy", "Bad"),
        
        # Row 10
        ("Nyakoi", "Any Position", "Light", "Any", "Risky", "Moderate", "Moderate", "Bad"),
        
        # Row 11
        ("Kus;Tarai", "New", "Heavy", "Moderate", "Risky", "Moderate", "Moderate", "Bad"),
        
        # Row 12
        ("Genya;Tarai", "Midway", "Heavy", "Moderate", "Risky", "Moderate", "Moderate", "Bad"),
        
        # Row 13
        ("Kus", "Midway", "Heavy", "Moderate", "Stormy", "Moderate", "Moderate", "Risky"),
        
        # Row 14
        ("Genya", "New", "Heavy", "Moderate", "Risky", "Moderate", "Moderate", "Risky"),
        
        # Row 15
        ("Kus;Genya;Tarai;Nyagire;Nyabukoba", "Any Position", "Cloudy", "Moderate", "Windy", "High", "Heavy", "Bad"),
        
        # Row 16
        ("Genya", "Descending", "Light", "Moderate", "Risky", "Moderate", "Moderate", "Bad"),
        
        # Row 17
        ("Tarai", "Descending", "Light", "High", "Risky", "High", "Low", "Bad"),
        
        # Row 18
        ("Kus;Genya", "Low", "Cloudy", "High", "Risky", "High", "Low", "Bad"),
        
        # Row 19
        ("Genya;Kus", "Ascending", "Cloudy", "Moderate", "Windy", "High", "Heavy", "Bad"),
        
        # Row 20
        ("Kus;Nyagire", "Midway", "Cloudy", "Moderate", "Windy", "High", "Heavy", "Bad"),
        
        # Row 21
        ("Kus", "Low", "Heavy", "Moderate", "Windy", "Moderate", "Moderate", "Bad"),
        
        # Row 22
        ("Nyagire", "New;Ascending", "Cloudy", "Moderate", "Windy", "High", "Heavy", "Bad"),
        
        # Row 23 - THIS IS THE RULE YOU ASKED ABOUT!
        ("Nyagire", "New", "Heavy", "Warm", "Risky", "Moderate", "Moderate", "Risky"),
        
        # Row 24
        ("Nyadhiwa", "Ascending;Descending", "Cloudy", "High", "Windy", "High", "Moderate", "Bad"),
        
        # Row 25
        ("Kus", "Ascending", "Clear", "Moderate", "Normal", "Moderate", "No", "Good"),
        
        # Row 26
        ("Kus", "Descending", "Cloudy", "High", "Windy", "High", "Light", "Good"),
        
        # Row 27
        ("Kus", "Low", "Cloudy", "High", "Risky", "High", "Low", "Bad"),
        
        # Row 28
        ("Kus", "Descending;Low", "Clear", "High", "Normal", "Moderate", "No", "Good"),
        
        # Row 29
        ("Kus", "New", "Heavy", "Warm", "Risky", "Moderate", "Moderate", "Bad"),
        
        # Row 30
        ("Kus", "Midway;Ascending", "Cloudy", "Moderate", "Windy", "Moderate", "Light", "Good"),
        
        # Row 31
        ("Genya", "New", "Heavy", "Moderate", "Normal", "Moderate", "Moderate", "Good"),
        
        # Row 32
        ("Genya", "Midway", "Clear", "Moderate", "Normal", "Moderate", "No", "Good"),
        
        # Row 33
        ("Genya", "Midway", "Cloudy", "Warm", "Bad", "Warm", "Moderate", "Risky"),
        
        # Row 34
        ("Genya", "Descending", "Clear", "High", "Windy", "Moderate", "No", "Good"),
        
        # Row 35
        ("Genya", "Descending", "Heavy", "High", "Bad", "High", "Moderate", "Risky"),
        
        # Row 36
        ("Nyadhiwa", "Descending", "Clear", "Cold", "Normal", "Moderate", "No", "Normal"),
        
        # Row 37
        ("Nyadhiwa", "Ascending", "Light;Cloudy", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 38
        ("Nyadhiwa", "New", "Light", "High", "Windy", "High", "No", "Normal"),
        
        # Row 39
        ("Nyadhiwa", "New", "Cloudy", "Moderate", "Windy", "Moderate", "Light", "Normal"),
        
        # Row 40
        ("Nyadhiwa", "Midway", "Clear", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 41
        ("Nyadhiwa", "Midway", "Light", "Moderate", "Windy", "High", "Light", "Normal"),
        
        # Row 42
        ("Nyadhiwa", "Low", "Clear;Light", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 43
        ("Nyadhiwa", "Low", "Cloudy", "Moderate", "Windy", "Moderate", "Moderate", "Normal"),
        
        # Row 44
        ("Marimbe", "Low", "Clear", "High", "Windy", "High", "No", "Normal"),
        
        # Row 45
        ("Marimbe", "Low", "Light", "Moderate", "Windy", "Warm", "Light", "Normal"),
        
        # Row 46
        ("Marimbe", "Low", "Cloudy", "Moderate", "Windy", "Warm", "Moderate", "Normal"),
        
        # Row 47
        ("Marimbe", "Midway", "Light", "Moderate", "Windy", "High", "Light", "Normal"),
        
        # Row 48
        ("Marimbe", "Descending", "Light", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 49
        ("Marimbe", "New;Ascending", "Cloudy", "Moderate", "Normal", "High", "No", "Normal"),
        
        # Row 50
        ("Marimbe", "Descending", "Light", "Moderate", "Windy", "Warm", "Light", "Normal"),
        
        # Row 51
        ("Genya", "New;Low", "Light", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 52
        ("Genya", "New;Low", "Cloudy", "Moderate", "Windy", "Moderate", "Moderate", "Bad"),
        
        # Row 53
        ("Genya", "New", "Cloudy", "Warm", "Windy", "High", "Moderate", "Bad"),
        
        # Row 54 - THE Genya, New, Heavy, Warm RULE!
        ("Genya", "New", "Heavy", "Warm", "Risky", "High", "Moderate", "Risky"),
        
        # Row 55
        ("Genya", "Midway", "Clear", "Moderate", "Windy", "Moderate", "No", "Normal"),
        
        # Row 56
        ("Genya", "Ascending;Midway;Descending", "Clear", "Moderate", "Windy", "Moderate", "No", "Normal"),
        
        # Row 57
        ("Genya", "Ascending;Midway;Descending", "Heavy", "Moderate", "Windy", "Moderate", "Heavy", "Normal"),
        
        # Row 58
        ("Kus", "New;Low", "Clear", "Moderate", "Windy", "Moderate", "Light", "Normal"),
        
        # Row 59
        ("Kus", "New;Low", "Heavy", "Moderate", "Stormy", "Moderate", "Heavy", "Bad"),
        
        # Row 60
        ("Kus", "Ascending;Midway;Descending", "Clear", "Moderate", "Windy", "High", "No", "Normal"),
        
        # Row 61
        ("Kus", "Ascending;Midway;Descending", "Cloudy", "High", "Windy", "High", "Moderate", "Normal"),
        
        # Row 62
        ("Tarai", "New;Low", "Light", "Moderate", "Normal", "High", "No", "Normal"),
        
        # Row 63
        ("Tarai", "New;Low", "Cloudy", "High", "Windy", "Moderate", "Moderate", "Normal"),
        
        # Row 64
        ("Tarai", "Ascending;Midway;Descending", "Clear", "Moderate", "Windy", "Moderate", "No", "Normal"),
        
        # Row 65
        ("Tarai", "Ascending;Midway;Descending", "Cloudy", "High", "Windy", "High", "Moderate", "Normal"),
        
        # Row 66
        ("Tarai", "Descending", "Cloudy", "Warm", "Windy", "Warm", "Moderate", "Risky"),
        
        # Row 67
        ("Kus;Genya;Nyabukoba", "Low", "Cloudy", "Moderate", "Windy", "Moderate", "Heavy", "Bad"),
        
        # Row 68
        ("Kus;Genya;Nyabukoba", "Low", "Heavy", "Cool", "Windy", "Warm", "Heavy", "Risky"),
        
        # Row 69
        ("Kus;Genya;Nyabukoba", "New", "Heavy", "Cool", "Bad", "Warm", "Heavy", "Risky"),
        
        # Row 70
        ("Kus;Genya;Nyabukoba", "Low", "Heavy", "High", "Stormy", "High", "Very Heavy", "Risky"),
        
        # Row 71
        ("Kus;Genya", "Low", "Light", "Warm", "Stormy", "High", "Moderate", "Bad"),
        
        # Row 72
        ("Kus;Genya", "Low", "Clear", "Cool", "Normal", "Moderate", "No Rain", "Normal"),
        
        # Row 73
        ("Genya", "Midway", "Light", "Cool", "Stormy", "High", "Drizzling", "Bad"),
        
        # Row 74
        ("Kus", "Midway", "Heavy", "Warm", "Stormy", "Moderate", "Moderate", "Bad"),
        
        # Row 75
        ("Kus;Genya", "New", "Heavy", "Cool", "Stormy", "Moderate", "Moderate", "Bad"),
        
        # Row 76
        ("Kus", "New", "Clear", "Moderate", "Windy", "Moderate", "No Rain", "Bad"),
        
        # Row 77
        ("Kus", "New", "Light", "Warm", "Windy", "High", "Drizzling", "Bad"),
    ]
    
    # Clean the rules
    rules = clean_rule_sets(rules)
    
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

        # Track MAX firing strength (not sum!)
        if firing > lake_pos_max.get(lake_pos, 0):
            lake_pos_max[lake_pos] = firing
        if firing > wind_out_max.get(wind_out, 0):
            wind_out_max[wind_out] = firing
        if firing > temp_out_max.get(temp_out, 0):
            temp_out_max[temp_out] = firing
        if firing > rainfall_out_max.get(rainfall_out, 0):
            rainfall_out_max[rainfall_out] = firing

    def get_max_category(max_dict):
        if not max_dict:
            return "Normal"  # Default fallback
        return max(max_dict.items(), key=lambda x: x[1])[0]

    lake_pos = get_max_category(lake_pos_max)
    wind_out = get_max_category(wind_out_max)
    temp_out = get_max_category(temp_out_max)
    rainfall_out = get_max_category(rainfall_out_max)

    return (lake_pos, wind_out, temp_out, rainfall_out)