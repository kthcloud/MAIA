import os
import shutil
from pathlib import Path
import zipfile
import subprocess


#model_uri = os.environ["MLFLOW_MODEL"]
#Path(dst_path).mkdir(parents=True,exist_ok=True)
#try:
#    mlflow.pytorch.load_model(f"models:/{model_uri}/latest", dst_path=dst_path)
#except:
#    ...
dst_path = os.environ["MLFLOW_MODEL_PATH"]
monai_bundle_path = os.environ["MONAI_BUNDLE_PATH"]

Path(monai_bundle_path).mkdir(parents=True,exist_ok=True)

for file in os.listdir(dst_path):
    if file.endswith(".zip"):
        shutil.copy(Path(dst_path).joinpath(file), monai_bundle_path)

        with zipfile.ZipFile(Path(monai_bundle_path).joinpath(file), 'r') as zip_ref:
            zip_ref.extractall(monai_bundle_path)

        try:
            subprocess.run(["pip", "install", "-r",str(Path(monai_bundle_path).joinpath(file[:-len(".zip")], "requirements.txt"))])
        except:
            ...
