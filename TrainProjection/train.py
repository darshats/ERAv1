import os
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import COCO_CLIP_Dataset
from model import PhiWrapper

torch.set_default_device("cuda")

def file_exists(image_id, fpath = '/media/App/amaranth/lavanya/Capstone_data/clip_features_base_patch32/'): 
    n = '0'*(12-len(str(image_id))) + str(image_id) + '.h5'
    fp = os.path.join(fpath, n)
    if os.path.exists(fp): 
        return True
    else: 
        return False

if __name__ == 'main':
    device = 'cuda'
    ## get tokenizer and model ready
    wrapper = PhiWrapper()
    tokenizer = wrapper.phi_tokenizer


    ## get data ready
    captions_info_df = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017.csv')
    captions_info_df_subset = captions_info_df.drop_duplicates(subset='image_id', keep='first')
    captions_info_df_subset['image_embed_exists'] = captions_info_df_subset['image_id'].apply(lambda x: file_exists(x))
    max_token_len_data = 75
    dataset = COCO_CLIP_Dataset(
        captions_info_df_subset, 
        '/media/App/amaranth/lavanya/Capstone_data/clip_features_base_patch32/', 
        tokenizer, 
        max_token_len_data
        )

    batch_size_train = 60 
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    num_batches_train_on = 1500    
    num_batches_train_on, len(train_dataloader)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=1e-5, eps=1e-9) 
    num_epochs = 10
    vocab_size = 50295

    wrapper.train()
    N_batches = len(train_dataloader)
                    
    for epoch in range(num_epochs):

        print(f"Working on epoch {epoch}")
        for iteration, batch in enumerate(train_dataloader):
            if iteration == num_batches_train_on: 
                break 

            print(f"Iteration {iteration}/{num_batches_train_on}", end='\r')
            optimizer.zero_grad()
            input = batch[0]
            gt = batch[1] 
            output = wrapper(input.to(device))

            ## need to map gt token_ids to one-hot enocding vocab_size
            gt_one_hot = torch.nn.functional.one_hot(gt, vocab_size).to(torch.float32)

            ## output in correct shape
            output_tensor_new = torch.empty(batch_size_train, max_token_len_data, vocab_size)
            for idx, s in enumerate(output.scores): 
                output_tensor_new[:, idx, :] = s
            
            output_tensor_new = F.softmax(output_tensor_new, dim=-1)
            
            ## ce loss between output_tensor_new and gt_one_hot
            loss = F.cross_entropy(output_tensor_new.view(-1, vocab_size), gt.view(-1))

            loss.requires_grad = True
            loss.backward()

            optimizer.step() 
            # optimizer.zero_grad(set_to_none=True) 

            if (iteration % 100) == 0: 
                print("")

                ## print gt and output decoded tokens for visual inspection for the 1st el of batch
                gt_input_ids = gt[0]
                output_input_ids = torch.argmax(output_tensor_new[0], dim=1)
                output_sequences_input_ids = output.sequences[0, :]

                gt_idx_0_decoded = tokenizer.decode(gt_input_ids).replace('<|endoftext|>', '')
                output_idx_0_decoded = tokenizer.decode(output_input_ids).replace('<|endoftext|>', '')
                output_sequences_idx_0_decoded = tokenizer.decode(output_sequences_input_ids).replace('<|endoftext|>', '')

                # print(f"Loss: {loss}\nInput_ids (gt): {gt_input_ids}\nInput_ids (pred): {output_input_ids}\nInput_ids (sequence): {output_sequences_input_ids}")
                ## print loss 
                print(f"Loss: {loss}\nCaption (gt): {gt_idx_0_decoded}\nCaption (pred): {output_idx_0_decoded}\nCaption (sequence): {output_sequences_idx_0_decoded}")
    
            
        print("")
        print(f"Epoch {epoch} finished")
        print("")