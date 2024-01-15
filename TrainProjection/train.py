import os
from PIL import Image
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import COCO_CLIP_Dataset
from model import PhiWrapper

torch.set_default_device("cuda")

def file_exists(image_id, fpath = './embeddings'): 
    n = '0'*(12-len(str(image_id))) + str(image_id) + '.h5'
    fp = os.path.join(fpath, n)
    if os.path.exists(fp): 
        return True
    else: 
        return False

if __name__ == "__main__":

    
    wandb.login()
    wandb.init(project="Capstone, part 1", dir='./tmp', id='v1')

    ## get tokenizer and model ready
    max_token_len_data = 75
    wrapper = PhiWrapper(max_token_len_data)
    tokenizer = wrapper.phi_tokenizer

    ## get data ready
    captions_info_df = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017.csv')
    captions_info_df_subset = captions_info_df.drop_duplicates(subset='image_id', keep='first')
    captions_info_df_subset['image_embed_exists'] = captions_info_df_subset['image_id'].apply(lambda x: file_exists(x))
    
    dataset = COCO_CLIP_Dataset(
        captions_info_df_subset, 
        'embeddings', 
        tokenizer, 
        max_token_len_data
        )

    batch_size_train = 3
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, generator = torch.Generator(device='cuda'))
    num_batches_train_on = 1500    
    num_batches_train_on, len(train_dataloader)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=1e-5, eps=1e-9) 
    num_epochs = 10
    vocab_size = 51200

    wrapper.train()
    N_batches = len(train_dataloader)
                    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        for iteration, (x, gt) in enumerate(train_dataloader):
            if iteration == num_batches_train_on: 
                break 

            print(f"Iteration {iteration}/{num_batches_train_on}", end='\r')
            optimizer.zero_grad()
            
            ## gt is of form (batch, input caption tokenized and padded)
            ## x is clip image embed (batch, 49, 768)
            ## pass through wrapper, get loss from greedy strategy
            loss, word_output_pred_tokens = wrapper(x, gt)
            loss.requires_grad = True
            loss.backward()

            optimizer.step() 
            # optimizer.zero_grad(set_to_none=True) 
            epoch_loss += loss.item()

            if (iteration % 50) == 0: 
                print(f'Iteration: {iteration}, Loss: {loss.item()}')
                wandb.log({"loss": loss.item()})

                num_rows = gt.shape[0]
                batch_preds = word_output_pred_tokens.int()
                for i in range(num_rows):
                    gt_text = tokenizer.decode(gt[i])
                    pred_text = tokenizer.decode(batch_preds[i])
                print(f"Batch data {i}: \nCaption (gt): {gt_text}\nCaption (pred): {pred_text}\n")

        avg_loss = epoch_loss/(iteration+1) 
        wandb.log({"epoch loss": avg_loss})
        print(f"Epoch {epoch} finished")