'''
Created on 20-Sep-2024

@author: Nomura
'''
import configparser
import os

# Create a ConfigParser object
config = configparser.ConfigParser()

def load_config():

    # Read the properties file
    config_file_path = os.getenv("CONFIG_PATH")
        
    print(f" Config fle path: {config_file_path}")
    
    # Read the properties file
    config.read(config_file_path)
        
def get_property(section, key):
    return config[section][key]