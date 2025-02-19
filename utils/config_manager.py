import yaml
import os

class ConfigManager:
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.config = {}

    def load_config(self, filename):
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                self.config[filename] = config
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file {filepath}: {e}")

    def get_config(self, filename):
        if filename in self.config:
            return self.config[filename]
        else:
            return self.load_config(filename)

    def get_model_config(self):
        return self.get_config('model_config.yaml')

    def get_data_config(self):
        return self.get_config('data_config.yaml')

    def get_logging_config(self):
        return self.get_config('logging_config.yaml')
