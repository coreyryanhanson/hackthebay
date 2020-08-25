import re


def standardize_CMC_tidal_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*outgoing.*ebb", value, flags=re.IGNORECASE):
        return "Outgoing (Ebb)"
    elif re.match(".*incoming.*flood", value, flags=re.IGNORECASE):
        return "Incoming (Flood)"
    elif re.match(".?high", value, flags=re.IGNORECASE):
        return "High"
    elif re.match(".?low", value, flags=re.IGNORECASE):
        return "Low"
    else:
        return value

    
def standardize_CMC_water_surf_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*white.*caps", value, flags=re.IGNORECASE):
        return "White Caps"
    elif re.match(".?calm", value, flags=re.IGNORECASE):
        return "Calm"
    elif re.match(".?ripple", value, flags=re.IGNORECASE):
        return "Ripple"
    elif re.match(".?waves", value, flags=re.IGNORECASE):
        return "Waves"
    else:
        return value
    
    
def standardize_CMC_wind_strings(value):
    if type(value) is not str:
        return value
    elif re.match("\D?1\D*10\D*knots", value, flags=re.IGNORECASE):
        return "1 To 10 Knots"
    elif re.match("\D?10\D*20\D*knots", value, flags=re.IGNORECASE):
        return "Calm"
    elif re.match("\D?20\D*30\D*knots", value, flags=re.IGNORECASE):
        return "20 To 30 Knots"
    elif re.match("\D?40\D*knots", value, flags=re.IGNORECASE):
        return "Above 40 Knots"
    else:
        return value

    
def standardize_CMC_weather_strings(value):
    if type(value) is not str:
        return value
    elif re.match(".*partly.*cloudy", value, flags=re.IGNORECASE):
        return "Partly cloudy"
    elif re.match(".*intermittent.*rain", value, flags=re.IGNORECASE):
        return "Intermittent rain"
    elif re.match(".*fog.*haze", value, flags=re.IGNORECASE):
        return "fog/haze"
    elif re.match(".?sunny", value, flags=re.IGNORECASE):
        return "Sunny"
    elif re.match(".?overcast", value, flags=re.IGNORECASE):
        return "Overcast"
    elif re.match(".?rain", value, flags=re.IGNORECASE):
        return "Rain"
    elif re.match(".?drizzle", value, flags=re.IGNORECASE):
        return "Drizzle"
    elif re.match(".?snow", value, flags=re.IGNORECASE):
        return "Snow"
    else:
        return value