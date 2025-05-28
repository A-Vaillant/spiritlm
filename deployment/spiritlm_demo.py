#!/usr/bin/env python3
"""
Spirit LM Gradio Demo
Interactive interface for Spirit LM text-to-speech and speech-to-text generation
"""

import os
import sys
import gradio as gr
import torch
import torchaudio
import logging
import tempfile
import time
import traceback
from pathlib import Path

from observability import resource_monitor

try:
    # from spiritlm import Spiritlm
    from spiritlm.model.spiritlm_model import ContentType, GenerationConfig, GenerationInput, OutputModality, Spiritlm
    print("Spirit LM imports successful!")
except ImportError as e:
    print(f"Error importing Spirit LM: {e}")
    print("Please ensure Spirit LM is properly installed.")
    sys.exit(1)
    
logger = logging.getLogger(__name__)

# Global variables
spirit_lm = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_tokens(text, outputs):
    """Create a visualization of text and speech tokens"""
    if not outputs or len(outputs) == 0:
        return "No tokens to visualize."
    
    try:
        # Get the token sequence from the model output
        # This assumes the output has token information - adjust based on actual Spirit LM API
        token_info = []
        
        # Add input text tokens
        token_info.append(f"📝 **Input Text**: {text}")
        token_info.append("")
        
        # Add generated tokens visualization
        output = outputs[0]
        if hasattr(output, 'tokens') or hasattr(output, 'token_sequence'):
            tokens = getattr(output, 'tokens', getattr(output, 'token_sequence', []))
            
            token_display = []
            for i, token in enumerate(tokens):
                if hasattr(token, 'type') or hasattr(token, 'modality'):
                    token_type = getattr(token, 'type', getattr(token, 'modality', 'unknown'))
                    if token_type == 'speech' or token_type == ContentType.SPEECH:
                        token_display.append(f"🔊 [SPEECH_{i}]")
                    elif token_type == 'text' or token_type == ContentType.TEXT:
                        content = getattr(token, 'content', f'TEXT_{i}')
                        token_display.append(f"📝 {content}")
                    else:
                        token_display.append(f"❓ [UNK_{i}]")
                else:
                    # Fallback if token structure is different
                    token_display.append(f"🔹 [TOKEN_{i}]")
            
            if token_display:
                token_info.append("**Generated Token Sequence**:")
                token_info.extend(token_display)
            else:
                token_info.append("**Token sequence not available in output**")
        else:
            token_info.append("**Token information not available in model output**")
            
        return "\n".join(token_info)
        
    except Exception as e:
        return f"Error visualizing tokens: {str(e)}"
    
    
def initialize_model(model_type="spirit-lm-base-7b"):
    """Initialize the Spirit LM model"""
    global spirit_lm
    
    # ... checkpoint validation logic unchanged ...
    
    start_time = time.time()
        
    with resource_monitor() as resources:
        try:
            print(f"Initializing {model_type} on {device}...")
            spirit_lm = Spiritlm(model_type)
            
        except Exception as e:
            logger.error(f"Initialization failed. Peak usage - CPU: {resources['peak_cpu']:.1f}%, "
                        f"Memory: {resources['peak_memory']:.1f}%, GPU: {resources['peak_gpu_memory']:.2f}GB")
            return f"Error loading model: {str(e)}"
    
    logger.info(f"Initialization successful. Peak usage - CPU: {resources['peak_cpu']:.1f}%, "
                f"Memory: {resources['peak_memory']:.1f}%, GPU: {resources['peak_gpu_memory']:.2f}GB")

    elapsed = time.time() - start_time

    return f"Model loaded successfully after {elapsed:.2f} seconds!"  # No errors.


def generate_text_to_speech(text, temperature=0.8, top_p=0.9, max_tokens=200, speaker_id=0):
    """Generate speech from text input"""
    if spirit_lm is None:
        return None, "Please initialize the model first."
    
    try:
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=True
        )
        
        interleaved_inputs = [GenerationInput(
            content=text,
            content_type=ContentType.TEXT
        )]
        
        outputs = spirit_lm.generate(
            interleaved_inputs=interleaved_inputs,
            output_modality=OutputModality.SPEECH,
            generation_config=generation_config,
            speaker_id=speaker_id
        )
                    
        # Save audio to temporary file
        if outputs and len(outputs) > 0:
            output = outputs[0]
            sample_rate = 16000  # Spirit LM default sample rate
            
            if len(output.content.shape) == 1:
                audio_data = torch.from_numpy(output.content).unsqueeze(0)
            else:
                # Stereo audio data
                audio_data = torch.from_numpy(output.content)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, audio_data, sample_rate)
                token_ciz = visualize_tokens(text, output)
                return tmp_file.name, "Audio generated successfully!", token_ciz
        else:
            return None, "No audio output generated.", ""
            
    except Exception as e:
        return None, f"Error generating audio: {str(e)}", ""

def generate_speech_to_text(audio_file, temperature=0.8, top_p=0.9, max_tokens=200):
    """Generate text from speech input"""
    if spirit_lm is None:
        return "Please initialize the model first."
    
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Resample if necessary (Spirit LM expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=True
        )
        
        interleaved_inputs = [GenerationInput(
            content=waveform.squeeze(0),
            content_type=ContentType.SPEECH
        )]
        
        outputs = spirit_lm.generate(
            interleaved_inputs=interleaved_inputs,
            output_modality=OutputModality.TEXT,
            generation_config=generation_config
        )
        
        if outputs and len(outputs) > 0:
            return outputs[0].content
        else:
            return "No text output generated."
            
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def get_system_info():
    """Get system information"""
    info = []
    info.append(f"PyTorch version: {torch.__version__}")
    info.append(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    return "\n".join(info)

# Create Gradio interface
with gr.Blocks(title="Spirit LM Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Spirit LM: Interleaved Spoken and Written Language Model")
    gr.Markdown("Interactive demo for Meta's Spirit LM - supporting both text-to-speech and speech-to-text generation.")
    
    with gr.Tab("Model Setup"):
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=["spirit-lm-base-7b", "spirit-lm-expressive-7b"],
                value="spirit-lm-base-7b",
                label="Model Type"
            )
            init_button = gr.Button("Initialize Model", variant="primary")
        
        init_output = gr.Textbox(label="Initialization Status", lines=3)
        system_info = gr.Textbox(label="System Information", value=get_system_info(), lines=5)
    
    with gr.Tab("Text to Speech"):
        with gr.Row():
            with gr.Column():
                tts_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to convert to speech...",
                    lines=3
                )
                
                with gr.Row():
                    tts_temp = gr.Slider(0.1, 2.0, 0.8, label="Temperature")
                    tts_top_p = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
                
                with gr.Row():
                    tts_tokens = gr.Slider(50, 500, 200, label="Max Tokens")
                    speaker_id = gr.Slider(0, 10, 0, step=1, label="Speaker ID")
                
                tts_button = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                tts_audio = gr.Audio(label="Generated Audio", type="filepath")
                tts_status = gr.Textbox(label="Status")
                tts_tokens = gr.Markdown(label="Token Visualization")
    
    with gr.Tab("Speech to Text"):
        with gr.Row():
            with gr.Column():
                stt_audio = gr.Audio(label="Upload Audio", type="filepath")
                
                with gr.Row():
                    stt_temp = gr.Slider(0.1, 2.0, 0.8, label="Temperature")
                    stt_top_p = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
                
                stt_tokens = gr.Slider(50, 500, 200, label="Max Tokens")
                stt_button = gr.Button("Generate Text", variant="primary")
            
            with gr.Column():
                stt_text = gr.Textbox(label="Generated Text", lines=5)
    
    # Event handlers
    init_button.click(
        initialize_model,
        inputs=[model_selector],
        outputs=[init_output]
    )
    
    tts_button.click(
        generate_text_to_speech,
        inputs=[tts_text, tts_temp, tts_top_p, tts_tokens, speaker_id],
        outputs=[tts_audio, tts_status, tts_tokens]
    )
    
    stt_button.click(
        generate_speech_to_text,
        inputs=[stt_audio, stt_temp, stt_top_p, stt_tokens],
        outputs=[stt_text]
    )

if __name__ == "__main__":
    print("Starting Spirit LM Gradio Demo...")
    print(f"System info:\n{get_system_info()}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )