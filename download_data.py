import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load the hidden .env file
load_dotenv()

# Securely fetch the API key
API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=API_KEY)

# Download the data
project = rf.workspace("a188370").project("infrastructure-defects-detection")
version = project.version(4)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")