import os
import shutil

# List of configuration files to delete
config_files = [
    "config.json",
    "config_blend.json",
    "config_synthesize.json",
    "config_blend.json",
    "config_feature_extraction.json", 
    "config_train.json"
]

# Iterate over the list and delete each file if it exists
for file in config_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")
    else:
        print(f"{file} does not exist")


# Copy config copy.json to config.json
shutil.copy("config copy.json", "config.json")
print("Copied copy.json to config.json")