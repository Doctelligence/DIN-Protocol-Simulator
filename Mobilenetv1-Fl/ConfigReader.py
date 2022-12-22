import os, configparser
import yaml

class ConfigReader():
    def __init__(self):
        self.raw_data_root_dir = ""
        self.data_root_dir = ""
        self.test_dir = ""
        self.val_dir = ""
        self.model_save_dir = ""
        self.experiment_id = None
        self.root_dir = ""
        self.batch_size = 0
        self.num_epochs = 0
        self.max_rounds = 0
        self.no_patients = 0

        self.configRead()

    def readConfigFile(self,config_file):
        with open(config_file, 'r') as stream:
            try:
                app_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return app_config

    def configRead(self):
        # get the mluc config file name from the environment variable
        if "FL_CONFIG_FILE" in os.environ.keys():
            configFile = os.environ["FL_CONFIG_FILE"]
        else:
            raise Exception("FL_CONFIG_FILE not specified")

        if not os.path.isfile(configFile):
            raise Exception("Config file not found", configFile)

        # read the config file
        config = self.readConfigFile(configFile)

        if "experiment_id" in config.keys():
            self.experiment_id = str(config["experiment_id"])
        else:
            print("experiment_id not defined")
            exit(3)

        if "raw_data_root_dir" in config.keys():
            self.raw_data_root_dir = str(config["raw_data_root_dir"])
            try:
                os.path.exists(self.raw_data_root_dir)
            except:
                print("raw data root directory path {} invalid".format(self.raw_data_root_dir))

        else:
            print("raw data root directory not defined")


        # process the data from config file
        if "directories" in config.keys():

            if "root_dir" in config["directories"].keys():
                self.root_dir = os.path.join(str(config["directories"]["root_dir"]),self.experiment_id)
                try:
                    os.path.exists(self.data_root_dir)
                except:
                    print("data root directory path {} invalid".format(self.root_dir))
                    exit(1)

            if "data_root_dir" in config["directories"].keys():
                self.data_root_dir = os.path.join(self.root_dir,str(config["directories"]["data_root_dir"]))
                try:
                    os.path.exists(self.data_root_dir)
                except:
                    print("data root directory path {} invalid".format(self.data_root_dir))
                    exit(1)
            else:
                print("data root directory not defined")
                exit(1)

            if "val_dir" in config["directories"].keys():
                self.val_dir = os.path.join(str(config["directories"]["root_dir"]),str(config["directories"]["val_dir"]))
                try:
                    os.path.exists(self.val_dir)
                except:
                    print("val directory path {} invalid".format(self.val_dir))
                    exit(1)
            else:
                print("val directory not defined")
                exit(1)

            if "test_dir" in config["directories"].keys():
                self.test_dir = os.path.join(str(config["directories"]["root_dir"]),str(config["directories"]["test_dir"]))
                try:
                    os.path.exists(self.test_dir)
                except:
                    print("test_dir directory path {} invalid".format(self.test_dir))
                    exit(1)
            else:
                print("test_dir directory not defined")
                exit(1)

            if "model_save_dir" in config["directories"].keys():
                self.model_save_dir = os.path.join(self.root_dir,str(config["directories"]["model_save_dir"]))
                try:
                    os.path.exists(self.model_save_dir)
                except:
                    print("model_save_dir directory path invalid".format(self.model_save_dir))
                    exit(1)
            else:
                print("model_save_dir directory not defined")
                exit(1)

        else:
            print("directories list not defined in config file")
            exit(1)

        if "parameters" in config.keys():
            if "batch_size" in config["parameters"].keys():
                self.batch_size = int(config["parameters"]["batch_size"])
            else:
                print("batch_size not defined in directory")
                exit(2)

            if "num_epochs" in config["parameters"].keys():
                self.num_epochs = int(config["parameters"]["num_epochs"])
            else:
                print("num_epochs not defined in directory")
                exit(2)

            if "max_rounds" in config["parameters"].keys():
                self.max_rounds = int(config["parameters"]["max_rounds"])
            else:
                print("max_rounds not defined in directory")
                exit(2)

            if "num_patients" in config["parameters"].keys():
                self.no_patients = int(config["parameters"]["num_patients"])
            else:
                print("num_patients not defined in directory")
                exit(2)

    def printConfig(self):
        print("root_dir: {}".format(self.root_dir))
        print("data_root_dir: {}".format(self.data_root_dir))
        print("test_dir: {}".format(self.test_dir))
        print("val_dir: {}".format(self.val_dir))
        print("model_save_dir: {}".format(self.model_save_dir))

        print("batch_size: {}".format(self.batch_size))
        print("num_epochs: {}".format(self.num_epochs))
        print("max_rounds: {}".format(self.max_rounds))
        print("num_patients: {}".format(self.no_patients))
