import yaml
import atexit
import warnings

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

    @staticmethod
    def configure_multiprocessing():
        """Configure multiprocessing to prevent semaphore leaks at shutdown.
        
        Sets torch multiprocessing sharing strategy to 'file_system' to avoid
        resource_tracker warnings about leaked semaphore objects.
        """
        # Suppress resource_tracker warnings at exit
        Utils.suppress_resource_tracker_warnings()
        
        try:
            import torch
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            # Older torch versions or unsupported platforms may raise; ignore
            pass

    @staticmethod
    def suppress_resource_tracker_warnings():
        """Suppress resource_tracker warnings about leaked semaphores at shutdown.
        
        This registers an atexit handler that filters out resource_tracker
        warnings which are harmless in production but noisy in tests.
        """
        def _exit_handler():
            try:
                # Suppress UserWarning from resource_tracker at shutdown
                warnings.filterwarnings("ignore", category=UserWarning, 
                                      message=".*resource_tracker:.*")
            except Exception:
                pass
        
        atexit.register(_exit_handler)
