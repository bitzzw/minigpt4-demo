import argparse

import torch
import gradio as gr

from minigpt4 import Config
from minigpt4.models import *
from minigpt4.processor import *
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, Prompt_Message_Keep


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args.cfg_path)

vis_processor = registry.get_processor_class("blip2_image_eval").from_config()

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

chat = Conversation(model, vis_processor)
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
            None, 
            gr.update(value=None, interactive=True), 
            gr.update(placeholder='Please upload your image first', interactive=False),
            gr.update(value="Upload & Start Chat", interactive=True), 
            chat_state, 
            img_list
            )
    
def gradio_upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = Prompt_Message_Keep.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return (
        gr.update(interactive=False), 
        gr.update(interactive=True, placeholder='Type and press Enter'), 
        gr.update(value="Start Chatting", interactive=False), 
        chat_state, 
        img_list
        )
    
def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


# ========================================
#             Demo Launching
# ========================================

title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
description = """<h3>Please upload your image and start chatting!</h3>"""
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )        
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='MiniGPT-4')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
    
    upload_button.click(gradio_upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)
