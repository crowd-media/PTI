import os
import gradio as gr
from blend import blend
from feature_extraction import extract_features
from synthesize import synthesize
from train import train
from helpers.config_manager import ConfigManager

# global variable 
config_manager = None

def initialize_config_manager(video, duration: int, intensity:float):
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
    model_id = train(params_path, config_manager)
    return f"Training completed, model_id: {model_id}"

def synthesize_fn(synthesize_input):
    global config_manager
    params_path = os.path.join('.', f"config_synthesize.json")
    if synthesize_input:
        udpate_synth_config(synthesize_input)
    synthesize(params_path, config_manager)

    params_path = os.path.join('.', f"config_blend.json")
    blend(params_path)
    return "video.mp4"

def blend_fn():
    global config_manager
    params_path = os.path.join('.', f"config_blend.json")
    blend(params_path)
    return "Blending completed.", "video.mp4"

with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        with gr.Column():
            duration_input = gr.Textbox(label="Duration (in frames)")
            intensity_input = gr.Slider(label="Intensity", minimum=-5, maximum=5, step=0.1)
            initialize_button = gr.Button("1. Submit")

    with gr.Row():
        extract_button = gr.Button("2. Extract Features")
        extract_output = gr.Textbox(label="Feature Extraction Output")

    with gr.Row():
        train_button = gr.Button("3. Train")
        train_output = gr.Textbox(label="Training Output")

    with gr.Row():
        gen_video_button = gr.Button("4. Generate video")
        synthesize_input = gr.Textbox(label="model_id")        
        # gen_video_output = gr.Textbox(label="Generate video Output")
    
    video_output = gr.Video(label="Generated Video")

    def check_and_initialize(video, duration, intensity):
        if video and duration and intensity:
            initialize_config_manager(video, duration, intensity)
            return "1. Submit"
        return "Please fill all fields."

    def udpate_synth_config(model_id):
        global config_manager
        config_data = config_manager.open_config("synthesize")
        print(config_data)
        config_manager.update_config(config_data, "model_id", model_id)
        config_manager.save_config("synthesize", config_data)
        return
    
    inputs=[video_input, duration_input, intensity_input]
    initialize_button.click(check_and_initialize, inputs)

    extract_button.click(extract_features_fn, outputs=extract_output)
    train_button.click(train_fn, outputs=train_output)
    gen_video_button.click(synthesize_fn, synthesize_input, video_output)

demo.launch()

if __name__ == "__main__":
    demo.launch()
    
