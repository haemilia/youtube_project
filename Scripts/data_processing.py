import re
import pandas as pd

def parse_duration(duration_str):
        if isinstance(duration_str, pd.Timedelta):
            return duration_str
        if not isinstance(duration_str, str):
            return pd.NaT
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = int(match.group(3)) if match.group(3) else 0
            return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
        else:
            return pd.NaT
        
def get_tag_len(taglist):
    if taglist:
        return len(taglist)
    else:
        return 0
    
def default_language(item):
    result = {
        "default_language-None":0,
        "default_language-ko": 0,
        "default_language-other": 0,
    }
    result = pd.Series(result)
    if pd.isna(item):
        result["default_language-None"] = 1
        return result
    elif item in ["ko", "KO", "Ko"]:
        result["default_language-ko"] = 1
        return result
    else:
        result["default_language-other"] = 1
        return result

def default_audio_language(item):
    result = {
        "default_audio_language-None":0,
        "default_audio_language-ko": 0,
        "default_audio_language-zxx":0,
        "default_audio_language-other": 0,
    }
    result = pd.Series(result)
    if pd.isna(item):
        result["default_audio_language-None"] = 1
        return result
    elif item in ["ko", "KO", "Ko"]:
        result["default_audio_language-ko"] = 1
        return result
    elif item == "zxx":
        result["default_audio_language-zxx"] = 1
        return result
    else:
        result["default_audio_language-other"] = 1
        return result
    
def convert_str_int(b):
    # For safe conversion: from str to bool
    if isinstance(b, bool):
        return int(b)
    elif b in ["true", "True"]:
        return 1
    elif b in ["false", "False"]:
        return 0
    else:
        return 0
    
def is_blocked(item):
    if isinstance(item, (list, tuple)):
        return 1    
    elif pd.isna(item) or item in ["none", "None"]:
        return 0
    else:
        return 1
    
def check_number_notna(item):
    if isinstance(item, (int, float)):
        return item
    else:
        raise ValueError("This column must have values of int or float")
    
def check_number_na(item, feature):
    result = pd.Series({f"{feature}": 0, f"private-{feature}":0})
    if pd.isna(item) or item in ["none", "None"]:
        result[f"private-{feature}"] = 1
        return result
    elif isinstance(item, (int, float)):
        result[f"{feature}"] = item
        return result
    else:
        raise ValueError("This column must have values of int or float")
    
cleansing = {
    "length_of_title":lambda x: len(x.strip()),
    "length_of_description": lambda x: len(x.strip()),
    "number_of_tags": get_tag_len,
    "default_language": default_language,
    "default_audio_language": default_audio_language,
    "duration":lambda x: parse_duration(x).total_seconds(),
    "has_captions": convert_str_int,
    "represents_licensed_content": convert_str_int,
    "is_blocked_somewhere": is_blocked,
    "views":lambda x: check_number_notna(x),
    "likes": lambda x: check_number_na(x, "likes"),
    "comments": lambda x: check_number_na(x, "comments"),
    "has_ppl": convert_str_int,
    "was_streamed": lambda x: 0 if pd.isna(x) else 1,}