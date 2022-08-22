import json
from multiprocessing import Pipe

from data_pipeline import Pipeline

def extract_config():
    """
    Helper function to extract and handle basic formatting of config file
    """
    # 
    with open("config.json", "r") as read_content:
        ret_dict = json.load(read_content)
    return ret_dict

if __name__ == "__main__":
    """
    Main function for article validity model program   
    """
    
    print("Fake News Predictor")
    config_data = extract_config()
    pipeline = Pipeline(config_data=config_data)
    pipeline.run()
