import os
from PIL import Image
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from model import PhiWrapper


def feature_select(image_forward_outs):
    image_features = image_forward_outs.hidden_states[-1] # last layer
    # print(image_features.shape) # 1, 50, 768
    image_features = image_features[:, 1:, :]
    return image_features # 1, 49, 768


def do_infer(image_embedding, projection_path, resblock_path):
    wrapper = PhiWrapper(0).to(device='cuda')
    wrapper.eval()

    if os.path.exists(projection_path) and os.path.exists(resblock_path):
        print(f'loading from checkpoint')
        wrapper.projection_img.load_state_dict(torch.load(projection_path))
        wrapper.resblock.load_state_dict(torch.load(resblock_path))
    tokenizer = wrapper.phi_tokenizer

    image_embedding = wrapper.projection_img(image_embedding) # (b, 49, 2560)
    image_embedding = wrapper.resblock(image_embedding)

    instruct_part1 = tokenizer('Image: ', return_tensors='pt')
    instruct_part1_embedding = wrapper.frozen_phi.get_input_embeddings()(
        instruct_part1.input_ids.to('cuda')
        ).squeeze(0)
    
    instruct_part2 = tokenizer(' Caption: ', return_tensors='pt')
    instruct_part2_embedding = wrapper.frozen_phi.get_input_embeddings()(
        instruct_part2.input_ids.to('cuda')
        ).squeeze(0)

    x = torch.cat(
            (
                instruct_part1_embedding.unsqueeze(0), 
                image_embedding,
                instruct_part2_embedding.unsqueeze(0),
            ), 
            dim=1
        )
   
    out = wrapper.frozen_phi.generate(inputs_embeds=x)
    pred = tokenizer.decode(out[0]).replace('<|endoftext|>', '')
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
    pred = do_infer(image_embedding, projection_path, resblock_path)
    print(f'caption:{pred}')
    
    
