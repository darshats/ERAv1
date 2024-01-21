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
    wandb_run = wandb.init(project="Capstone, part 1", dir='./tmp', id='v8')
    ## v3 - lr=1e-5, full feature forcing
    ## v4 - lr=5e-3, full feature forcing
    ## v5 - lr=5e-3, part feature forcing, 'summarize this:'<image>'. Answer:'
    ## v6, same as above, no feature forcing
    ## v7, some feature forcing, hack loss prop with each generation
    ## v8, skip generate, use forward method

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

    batch_size_train = 2
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, generator = torch.Generator(device='cuda'))
    num_batches_train_on = 4000    
    num_batches_train_on, len(train_dataloader)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=5e-3, eps=1e-9) 
    num_epochs = 10
    N_batches = len(train_dataloader)

    wrapper.train()
    for name, param in wrapper.frozen_phi.named_parameters():
        param.requires_grad = False
                    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        for iteration, (x, gt) in enumerate(train_dataloader):

            print(f"Iteration {iteration}/{num_batches_train_on}", end='\r')
            loss = 0
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                ## gt is of form (batch, input caption tokenized and padded)
                ## x is clip image embed (batch, 49, 768)
                ## pass through wrapper, get loss from greedy strategy
                pred_logits = wrapper(x, gt)   ## (batch, max_predicted_len, vocab_size)
                ## add up losses at each predicted index
                max_predicted_len = pred_logits.shape[1]
                for idx in range(max_predicted_len):
                    ## get the GT across batch at the idx th position of output
                    gt_word_token = gt[:,idx]
                    pred_logits_idx = pred_logits[:,idx,:]
                    loss_at_idx = F.cross_entropy(
                        F.softmax(pred_logits_idx, dim=-1), 
                        gt_word_token, 
                        ignore_index=tokenizer.pad_token_id, 
                        label_smoothing=0.1
                        )
                loss += loss_at_idx

            loss.backward()
            optimizer.step() 
            # optimizer.zero_grad(set_to_none=True) 
            epoch_loss += loss.detach().item()

            if (iteration % 10) == 0: 
                print(f'Iteration: {iteration}, Loss: {loss.item()}')
                wandb.log({"loss": loss.item()})

                num_rows = gt.shape[0]
                batch_preds = torch.argmax(pred_logits, dim=-1)
                for i in range(num_rows):
                    gt_text = tokenizer.decode(gt[i]).replace('<|endoftext|>', '')
                    pred_text = tokenizer.decode(batch_preds[i]).replace('<|endoftext|>', '')
                print(f"Batch data {i}: \nCaption (gt): {gt_text}\nCaption (pred): {pred_text}\n")
            else:
                del pred_logits

        avg_loss = epoch_loss/(iteration+1) 
        wandb.log({"epoch loss": avg_loss})
        print(f"Epoch {epoch} finished")