import os
import gradio as gr
from blend import blend
from feature_extraction import extract_features
from synthesize import synthesize
from train import train
from helpers.config_manager import ConfigManager


def extract_features_fn(base_path, config_manager):
    params_path = os.path.join(base_path, f"config_feature_extraction.json")
    extract_features(params_path, config_manager)
    return "Features extracted. You can now start training."

def train_fn(base_path, config_manager):
    params_path = os.path.join(base_path, f"config_train.json")
    train(params_path, config_manager)
    return "Training completed."

def synthesize_fn(base_path, config_manager):
    params_path = os.path.join(base_path, f"config_synthesize.json")
    synthesize(params_path, config_manager)
    return "Synthesis completed."

def blend_fn(base_path):
    params_path = os.path.join(base_path, f"config_blend.json")
    blend(params_path)
    return "Blending completed."

with gr.Blocks() as demo:
    video_input = gr.Video(label="Upload Video")
    duration_input = gr.Textbox(label="Duration")
    intensity_input = gr.Textbox(label="Intensity")

    init_config_manager = gr.Button("Initialize config manager")
    extract_button = gr.Button("Extract Features")
    train_button = gr.Button("Start Training")
    synthesize_button = gr.Button("Synthesize")
    blend_button = gr.Button("Blend")

    extract_output = gr.Textbox(label="Feature Extraction Output")
    train_output = gr.Textbox(label="Training Output")
    synthesize_output = gr.Textbox(label="Synthesis Output")
    blend_output = gr.Textbox(label="Blending Output")

    base_path = '.'
    config_manager = ConfigManager(base_path, video_path, duration, intensity)
    
    extract_button.click(extract_features_fn, inputs=[base_path, config_manager], outputs=extract_output)
    train_button.click(train_fn, inputs=[base_path, config_manager], outputs=train_output)
    synthesize_button.click(synthesize_fn, inputs=[base_path, config_manager], outputs=synthesize_output)
    blend_button.click(blend_fn, inputs=[base_path, config_manager], outputs=blend_output)

demo.launch()
