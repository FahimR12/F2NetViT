# Description: A class to load configurations from multiple YAML files

import yaml
import os
from pathlib import Path

class MyConfig: # - nested to avoid global state 
    class Configs:
        def __init__(self, dataset_config, paths_config, training_config=None, model_config=None):
            self.train_root_dir = os.path.join(dataset_config['root_dir'], dataset_config['train_sub_dir'])
            self.validation_dir = os.path.join(dataset_config['root_dir'], dataset_config['validation_sub_dir'])
            self.json_file = dataset_config.get('json_file')
            self.dataset_xlsx = dataset_config.get('dataset_xlsx')
            self.a_test_patient = dataset_config.get('a_test_patient')
            self.root_dir = dataset_config['root_dir']
            
            # Full paths if provided in a separate YAML
            self.full_paths = {
                'dataset_file': paths_config.get('dataset_file'),
                'json_file': paths_config.get('json_file'),
                'test_patient': paths_config.get('test_patient'),
                'train_path': paths_config.get('train_path'),
                'validation_path': paths_config.get('validation_path')
            }

            # Include additional configurations if available
            self.training_config = training_config
            self.model_config = model_config

    def __init__(self, conf_base_path):
        self.conf_base_path = conf_base_path
        self.load_configs()

    def load_configs(self):
        # Load each configuration part from the respective YAML file
        with open(os.path.join(self.conf_base_path, 'dataset/default.yaml'), 'r') as f:
            dataset_config = yaml.safe_load(f)

        with open(os.path.join(self.conf_base_path, 'paths/default.yaml'), 'r') as f:
            paths_config = yaml.safe_load(f)
        
        # Load training and model configs if available
        training_config = {}
        model_config = {}

        training_path = os.path.join(self.conf_base_path, 'training/default.yaml')
        if os.path.exists(training_path):
            with open(training_path, 'r') as f:
                training_config = yaml.safe_load(f)

        model_path = os.path.join(self.conf_base_path, 'model/default.yaml')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model_config = yaml.safe_load(f)

        self.Configs = self.Configs(dataset_config, paths_config, training_config, model_config)

# Usage
conf_base_path = '/home/fahim/F2NetViT/conf'
configs = MyConfig(conf_base_path)

# test_config.py

# Instantiate the MyConfig class
conf_base_path = '/home/fahim/F2NetViT/conf'
configs = MyConfig(conf_base_path)

"""# Access and print configuration details to verify correctness
print("Train Root Directory:", configs.Configs.train_root_dir)
print("Validation Directory:", configs.Configs.validation_dir)
print("JSON File:", configs.Configs.json_file)
print("Dataset XLSX:", configs.Configs.dataset_xlsx)
print("A Test Patient:", configs.Configs.a_test_patient)
print("Full Paths:", configs.Configs.full_paths)
print("Training Config:", configs.Configs.training_config)
print("Model Config:", configs.Configs.model_config)
print("ROOT PATH:", configs.Configs.root_dir)"""