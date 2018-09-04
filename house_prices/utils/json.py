import json


def load_json(json_path):
    with open(json_path, 'rb') as json_file:
        return json.load(json_file)
