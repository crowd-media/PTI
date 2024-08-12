import gradio as gr

from helpers.config_manager import ConfigManager

with gr.Blocks() as demo:
    video_input = gr.Video(label="Upload Video")
    duration_input = gr.Textbox(label="Duration")
    intensity_input = gr.Textbox(label="Intensity")
    init_config_manager = gr.Button("Initialize config manager")
    extract_button = gr.Button("Extract Features")
    train_button = gr.Button("Start Training")
    generate_video_button = gr.Button("Synthesize")
    extract_output = gr.Textbox(label="Feature Extraction Output")
    train_output = gr.Textbox(label="Training Output")
    synthesize_output = gr.Textbox(label="Synthesis Output")
    blend_output = gr.Textbox(label="Blending Output")
    base_path = '.'
    config_manager = ConfigManager(base_path, video_path, duration, intensity)
    extract_button.click(extract_features_fn, inputs=[base_path, config_manager], outputs=extract_output)
    train_button.click(train_fn, inputs=[base_path, config_manager], outputs=train_output)
    generate_video_button.click(synthesize_fn, inputs=[base_path, config_manager], outputs=synthesize_output)
    

demo.launch()