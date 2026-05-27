import yaml

class Utils:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    @staticmethod
    def _save_config(config, config_path):
        with open(config_path, 'w') as file:
            yaml.dump(config, file)

    @staticmethod
    def update_config(config_path, new_config):
        config = Utils.load_config(config_path)
        for key, value in new_config.items():
            if key in config:
                config[key] = value
        Utils._save_config(config, config_path)
