import os
import gradio as gr
from blend import blend
from feature_extraction import extract_features
from synthesize import synthesize
from train import train
from helpers.config_manager import ConfigManager

def greet(config_directory):
    main(config_directory)
    return "Video processed", gr.Video("./video.mp4")

demo = gr.Interface(
    fn=greet,
    inputs=["text"], 
    outputs=["text", "video"],
)

def main(base_path):
    config_manager = ConfigManager(base_path)

    params_path = os.path.join(base_path, f"config_feature_extraction.json")
    extract_features(params_path, config_manager)
    params_path = os.path.join(base_path, f"config_train.json")
    train(params_path, config_manager)
    params_path = os.path.join(base_path, f"config_synthesize.json")
    synthesize(params_path, config_manager)
    params_path = os.path.join(base_path, f"config_blend.json")
    blend(params_path)

demo.launch()


if __name__ == "__main__":
    demo.launch()