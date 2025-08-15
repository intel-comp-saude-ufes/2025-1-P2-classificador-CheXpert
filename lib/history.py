from collections import defaultdict
import json

class History:
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def last(self, key):
        return self.history[key][-1] if key in self.history and self.history[key] else None

    def get(self, key):
        return self.history[key]

    def summary(self):
        return dict(self.history)

    def save_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def load_json(self, filepath):
        with open(filepath, "r") as f:
            self.history = defaultdict(list, json.load(f))

    def set_inner_dict(self, history_dict):
        self.history = history_dict
    
    def get_inner_dict(self):
        return self.history