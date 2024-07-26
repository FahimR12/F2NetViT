import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configures.my_config import MyConfig
import yaml
conf_base_path = '/home/fahim/F2NetViT/conf'
configs = MyConfig(conf_base_path)
path = configs.Configs.full_paths['train_path']
## ummcoment for path test
# check the path exits or not, if not correct it.
if os.path.exists(path):
    print("yes: the path found")
else:
    print("No: the path doesnt exists")

with open('configs.yaml', "r") as f:
    configs = yaml.safe_load(f)

print(configs)