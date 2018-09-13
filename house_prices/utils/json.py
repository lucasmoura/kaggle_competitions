import json


def load_json(json_path):
    with open(json_path, 'rb') as json_file:
        return json.load(json_file)


def save_json(save_path, json_dict):
    with open(save_path, 'w') as save_file:
        json.dump(json_dict, save_file)
