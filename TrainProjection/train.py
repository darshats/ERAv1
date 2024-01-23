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


def file_exists(image_id, fpath = './embeddings'): 
    n = '0'*(12-len(str(image_id))) + str(image_id) + '.h5'
    fp = os.path.join(fpath, n)
    if os.path.exists(fp): 
        return True
    else: 
        return False

if __name__ == "__main__":

    wandb.login()
    wandb_run = wandb.init(project="Capstone, part 1", dir='./tmp', id='v9')
    ## v3 - lr=1e-5, full feature forcing
    ## v4 - lr=5e-3, full feature forcing
    ## v5 - lr=5e-3, part feature forcing, 'summarize this:'<image>'. Answer:'
    ## v6, same as above, no feature forcing
    ## v7, some feature forcing, hack loss prop with each generation
    ## v8, skip generate, use forward method
    ## v9, refactored loss loop completely

    ## get tokenizer and model ready
    max_token_len_data = 75
    wrapper = PhiWrapper(max_token_len_data).to(device='cuda')
    tokenizer = wrapper.phi_tokenizer

    ## get data ready
    # captions_info_df = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017.csv')
    # captions_info_df['token_size'] = captions_info_df['caption'].apply(lambda x: tokenizer(x, return_tensors="pt", 
    #                                            return_attention_mask=False).input_ids.shape[1])
    # captions_info_df_subset = captions_info_df.drop_duplicates(subset='image_id', keep='first')
    # captions_info_df_subset = captions_info_df_subset[captions_info_df_subset['token_size'] <=30]
    # captions_info_df_subset['image_embed_exists'] = captions_info_df_subset['image_id'].apply(lambda x: file_exists(x))

    # captions_info_df_subset.to_csv('TrainProjection/captions_images_map_COCO_train2017_processed.csv')
    captions_info_df_subset = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017_processed.csv')
    dataset = COCO_CLIP_Dataset(
        captions_info_df_subset, 
        'embeddings', 
        tokenizer, 
        max_token_len_data
        )

    batch_size_train = 8
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    num_batches_train_on = 8000    

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=5e-4, eps=1e-9) 
    num_epochs = 10
    N_batches = len(train_dataloader)

    wrapper.train()
    for name, param in wrapper.frozen_phi.named_parameters():
        param.requires_grad = False

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
                    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        for iteration, (x, gt) in enumerate(train_dataloader):
            x = x.to('cuda')        ### (b, 49, 768)
            gt = gt.to('cuda')      ### (b, 75)
            print(f"Iteration {iteration}/{num_batches_train_on}", end='\r')

            batch_size = x.shape[0] 
            max_output_len = gt.shape[1]
            batch_preds = None
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                ## gt is of form (batch, input caption tokenized and padded)
                ## x is clip image embed (batch, 49, 768)
                ## pass through wrapper, get loss from greedy strategy
                current_tokens = None
                for idx in range(max_output_len):
                    ## get the GT across batch at the idx th position of output
                    gt_token = gt[:, idx]
                     ## if across the batch we only have pads then break
                    num_pad = sum(torch.eq(torch.tensor([tokenizer.pad_token_id]*batch_size, device='cuda'), gt_token))
                    if num_pad.detach().item() == batch_size:
                        break 

                    pred_probs = wrapper(x, current_tokens)   ### (b, )
                    pred_token = torch.argmax(pred_probs, dim=-1) ### (b, )
                    loss = loss_fn(pred_probs, gt_token)/max_output_len
                    loss.backward(retain_graph=True)

                    ## feature forcing!, send in next GT to help generation along right track
                    append_token = gt_token if idx<=3 else pred_token
                    current_tokens = append_token.unsqueeze(1) if current_tokens is None else torch.cat((current_tokens, append_token.unsqueeze(1)), dim=1)
                    batch_preds = pred_token.unsqueeze(1) if batch_preds is None else torch.cat((batch_preds, pred_token.unsqueeze(1)), dim=1)
                
            optimizer.step() 
            epoch_loss += loss.detach().item()
            if (iteration % 10) == 0: 
                print(f'Iteration: {iteration}, Loss: {loss.item()}')
                wandb.log({"loss": loss.item()})
                gt_text = tokenizer.decode(gt[-1]).replace('<|endoftext|>', '')
                pred_text = tokenizer.decode(batch_preds[-1]).replace('<|endoftext|>', '')
                print(f"Sample Caption (gt): {gt_text}\nCaption (pred): {pred_text}\n")

            del x
            del gt

        avg_loss = epoch_loss/(iteration+1) 
        wandb.log({"epoch loss": avg_loss})
        print(f"Epoch {epoch} finished")