import os
from PIL import Image
import numpy as np
import pandas as pd
import peft
from peft import LoraConfig
from peft.peft_model import PeftModel
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Phi2wrapper


def feature_select(image_forward_outs):
    image_features = image_forward_outs.hidden_states[-1] # last layer
    # print(image_features.shape) # 1, 50, 768
    image_features = image_features[:, 1:, :]
    return image_features # 1, 49, 768


def do_infer(image_embedding, question, peft_path, res_path, proj_path):
    config = PeftConfig.from_pretrained(peft_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(model, peft_path)
    peft_model.to('cuda')
    peft_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    phi2_projection_model = Phi2wrapper(phi2_model=peft_model, tokenizer=tokenizer)
    phi2_projection_model.to('cuda')
    phi2_projection_model.projection_img.load_state_dict(torch.load(proj_path))
    phi2_projection_model.resblock.load_state_dict(torch.load(res_path))
    # phi2_projection_model.phi2_model.load_state_dict(torch.load(phi2_path))
    phi2_projection_model.eval()

    tokenizer = phi2_projection_model.tokenizer

    image_embedding = phi2_projection_model.projection_img(image_embedding)
    image_embedding = phi2_projection_model.resblock(image_embedding) # (1, 49, 2560)
    input_q_embedding = phi2_projection_model.phi2_model.get_input_embeddings()(question)  # (1, n, 2560)

    bos = phi2_projection_model.tokenizer("Context: ", return_tensors="pt", return_attention_mask=False)
    eoi = phi2_projection_model.tokenizer(" Question: ", return_tensors="pt", return_attention_mask=False)
    eoa = phi2_projection_model.tokenizer(" Answer: ", return_tensors="pt", return_attention_mask=False)

    bos_embedding = phi2_projection_model.phi2_model.get_input_embeddings()(bos.input_ids.to('cuda')).squeeze(0)
    eoi_embedding = phi2_projection_model.phi2_model.get_input_embeddings()(eoi.input_ids.to('cuda')).squeeze(0)
    eoa_embedding = phi2_projection_model.phi2_model.get_input_embeddings()(eoa.input_ids.to('cuda')).squeeze(0)
    
    batch_size=1
    x = torch.cat((bos_embedding.repeat(batch_size,1,1),  # (1, 3, 2560)
                   image_embedding, 
                   eoi_embedding.repeat(batch_size,1,1),  # (1, 3, 2560)
                   input_q_embedding, 
                   eoa_embedding.repeat(batch_size,1,1)   # (1, 3, 2560)
                   ),  
                  dim=1
                  )  # (1, n+9, 2560)
   
    out = phi2_projection_model.phi2_model.generate(inputs_embeds=x)
    # pred = tokenizer.decode(out[0]).replace('<|endoftext|>', '')
    pred = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return pred


if __name__ == "__main__":
    ver = 'v16'
    projection_path = f'/mnt/d/era1chk/projection_{ver}.pth'
    resblock_path = f'/mnt/d/era1chk/resblock_{ver}.pth'
    vision_tower_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
    _ = vision_tower.requires_grad_(False)
    vision_tower = vision_tower.to("cuda")

    img_pil = Image.open('car.jpg')
    image = image_processor(images=img_pil, return_tensors="pt")
    image_forward_out = vision_tower(image['pixel_values'].to(device=vision_tower.device), output_hidden_states=True)
    image_embedding = feature_select(image_forward_out) ## (1, 49, 768)
    # pred = do_infer(image_embedding, projection_path, resblock_path)
    # print(f'caption:{pred}')
    
    
