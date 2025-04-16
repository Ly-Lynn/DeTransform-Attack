import pickle
import os
import json

class Logger:
    def __init__(self, root_dir, types = ["txt", "pkl", "json"]):
        self.root_dir = root_dir
        self.types = types
        self.create_dir()
    def create_dir(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
    def pickle(self, data, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'ab') as f:
            pickle.dump(data, f)
    def load_pickle(self, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'rb') as f:
            data = pickle.load(f)
        return data
    def txt(self, data, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'a') as f:
            f.write(data)
    def load_txt(self, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'r') as f:
            data = f.read()
        return data
    def json(self, data, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'a') as f:
            json.dump(data, f)
    def load_json(self, filename):
        dirr = os.path.join(self.root_dir, filename)
        with open(dirr, 'r') as f:
            data = json.load(f)
        return data
    def __call__(self, *args, **kwds):
        if self.types == "txt":
            self.txt(*args, **kwds)
        elif self.types == "pkl":
            self.pickle(*args, **kwds)
        elif self.types == "json":
            self.json(*args, **kwds)
        else:
            raise ValueError(f"Unknown file type: {self.types}. Supported types are: {self.types}")