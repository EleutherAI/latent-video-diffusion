import json

def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        raise Exception("Config file not found.")
    except json.JSONDecodeError:
        raise Exception("Error decoding JSON in the config file.")
