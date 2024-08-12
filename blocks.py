import os
import gradio as gr
from blend import blend
from feature_extraction import extract_features
from synthesize import synthesize
from train import train
from helpers.config_manager import ConfigManager

# Global variable to store the ConfigManager instance
config_manager = None

def initialize_config_manager(video, duration, intensity):
    global config_manager
    base_path = '.'
    config_manager = ConfigManager(base_path, video, duration, intensity)
    return 

def extract_features_fn():
    global config_manager
    params_path = os.path.join('.', f"config_feature_extraction.json")
    extract_features(params_path, config_manager)
    return "Features extracted."

def train_fn():
    global config_manager
    params_path = os.path.join('.', f"config_train.json")
    train(params_path, config_manager)
    return "Training completed."

def synthesize_fn():
    global config_manager
    params_path = os.path.join('.', f"config_synthesize.json")
    synthesize(params_path, config_manager)
    return "Synthesis completed."

def blend_fn():
    global config_manager
    params_path = os.path.join('.', f"config_blend.json")
    blend(params_path)
    return "Blending completed."

with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        duration_input = gr.Textbox(label="Duration (in frames)")
        intensity_input = gr.Textbox(label="Intensity")

    with gr.Row():
        extract_button = gr.Button("Extract Features")
        extract_output = gr.Textbox(label="Feature Extraction Output")

    with gr.Row():
        train_button = gr.Button("Start Training")
        train_output = gr.Textbox(label="Training Output")

    with gr.Row():
        synthesize_button = gr.Button("Synthesize")
        synthesize_output = gr.Textbox(label="Synthesis Output")

    with gr.Row():
        blend_button = gr.Button("Blend")
        blend_output = gr.Textbox(label="Blending Output")
    
    video_output = gr.Video(label="Generated Video")


    def check_and_initialize(video, duration, intensity):
        if video and duration and intensity:
            initialize_config_manager(video, duration, intensity)

    video_input.change(check_and_initialize, inputs=[video_input, duration_input, intensity_input])
    duration_input.change(check_and_initialize, inputs=[video_input, duration_input, intensity_input])
    intensity_input.change(check_and_initialize, inputs=[video_input, duration_input, intensity_input])

    extract_button.click(extract_features_fn, outputs=extract_output)
    train_button.click(train_fn, outputs=train_output)
    synthesize_button.click(synthesize_fn, outputs=synthesize_output)
    blend_button.click(blend_fn, outputs=[blend_output, video_output])

demo.launch()

if __name__ == "__main__":
    demo.launch()
