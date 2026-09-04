"""
Utilities functions.
"""
import importlib

def load_config(config_filename):
    '''
    Load the config file as a module. 
    @param config_filename: name of the config file to use
    @return module.Config(): config info as a .py module
    '''
    config_path = "configs.{}".format(config_filename.split('.')[0])
    module = importlib.import_module(config_path)
    return module.Config()

    
