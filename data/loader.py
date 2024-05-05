import os

import yaml
from roboflow import Roboflow


def get_roboflow_dataset(self, roboflow_api_key, roboflow_workspace_name, roboflow_project_name,
                         roboflow_version_number, download_path):
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(roboflow_workspace_name).project(roboflow_project_name)
    version = project.version(roboflow_version_number)
    version.download(model_format="yolov8", location=download_path, overwrite=True)
    # Get dataset directory and update dataset attributes
    self.dataset_directory = os.path.join(download_path)
    self.dataset_config = self.update_dataset_yaml(download_path)
    self.dataset_yaml_config, self.dataset_yaml_path = self.update_dataset_yaml(self.dataset_directory)
    return self.dataset_directory


def update_dataset_yaml(self, dataset_directory):
    yaml_path = os.path.join(dataset_directory, 'data.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"The data.yaml file does not exist at {yaml_path}")

    # Read the current contents of the file
    with open(yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)

    data_config['train'] = os.path.join(dataset_directory, 'train', 'images')
    data_config['val'] = os.path.join(dataset_directory, 'val', 'images')
    data_config['test'] = os.path.join(dataset_directory, 'test', 'images')

    # Write the updated contents back to the file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data_config, file, default_flow_style=False)

    # Return the updated data_config and yaml_path
    return data_config, yaml_path
