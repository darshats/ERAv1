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
from infer import do_infer

SAVE_N = 3
def save_checkpoint(module, checkpoint_folder, step):
    torch.save(module.state_dict(), os.path.join(checkpoint_folder, f'model_step_{step}.pth'))
    states = [x for x in os.listdir(checkpoint_folder)]
    states.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[2]))
    if len(states) > SAVE_N:
        print(f'There are {len(states)} checkpoints. Deleting...')
        for checkpoint in states[:-SAVE_N]:
            print(f'Deleting {checkpoint}')
            os.remove(os.path.join(checkpoint_folder, checkpoint))


def file_exists(image_id, fpath = './embeddings'): 
    n = '0'*(12-len(str(image_id))) + str(image_id) + '.h5'
    fp = os.path.join(fpath, n)
    if os.path.exists(fp): 
        return True
    else: 
        return False
    

def get_max_token_length(gt):
    token_len = 1
    for idx in range(max_output_len):
        ## get the GT across batch at the idx th position of output
        gt_token = gt[:, idx]
        ## if across the batch we only have pads then break
        num_pad = sum(torch.eq(torch.tensor([tokenizer.pad_token_id]*batch_size, device='cuda'), gt_token))
        if num_pad.detach().item() == batch_size:
            break 

        token_len += 1
    return token_len


if __name__ == "__main__":
    ver = 'v23'
    wandb.login()
    wandb_run = wandb.init(project="Capstone, part 1", dir='/mnt/d/wandb/tmp', id=ver)
    ## v3 - lr=1e-5, full feature forcing
    ## v4 - lr=5e-3, full feature forcing
    ## v5 - lr=5e-3, part feature forcing, 'summarize this:'<image>'. Answer:'
    ## v6, same as above, no feature forcing
    ## v7, some feature forcing, hack loss prop with each generation
    ## v8, skip generate, use forward method
    ## v9, refactored loss loop completely
    ## v10-11, loss addition instead of backward on each minibatch
    ## v13 softmax of logits instead of probs
    ## v14 attention masks and single caption batch with incremental output prediction
    ## v16 add wandb image logging
    ## v18 because no resume
    ## v19 because no resume
    ## v20 because no resume
    ## v21, v22 because no resume

    ## get tokenizer and model ready
    max_token_len_data = 0
    
    wrapper = PhiWrapper(max_token_len_data).to(device='cuda')
    if os.path.exists(f'/mnt/d/era1chk/projection_v22.pth') and os.path.exists(f'/mnt/d/era1chk/resblock_v22.pth'):
        print(f'loading from checkpoint')
        wrapper.projection_img.load_state_dict(torch.load(f'/mnt/d/era1chk/projection_v22.pth'))
        wrapper.resblock.load_state_dict(torch.load(f'/mnt/d/era1chk/resblock_v22.pth'))
    tokenizer = wrapper.phi_tokenizer

    ## get data ready
    # captions_info_df = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017.csv')
    # captions_info_df['token_size'] = captions_info_df['caption'].apply(lambda x: tokenizer(x, return_tensors="pt", 
    #                                            return_attention_mask=False).input_ids.shape[1])
    # captions_info_df_subset = captions_info_df.drop_duplicates(subset='image_id', keep='first')
    # captions_info_df_subset = captions_info_df_subset[captions_info_df_subset['token_size'] <=30]
    # captions_info_df_subset['image_embed_exists'] = captions_info_df_subset['image_id'].apply(lambda x: file_exists(x))

    # captions_info_df_subset.to_csv('TrainProjection/captions_images_map_COCO_train2017_processed.csv')
    ## load data preprocessed as per above
    captions_info_df_subset = pd.read_csv('TrainProjection/captions_images_map_COCO_train2017_processed.csv')
    dataset = COCO_CLIP_Dataset(
        captions_info_df_subset, 
        'embeddings', 
        tokenizer, 
        max_token_len_data
        )

    batch_size_train = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)
    num_batches_train_on = 64000    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrapper.parameters()), lr=5e-6, eps=1e-9) 
    num_epochs = 10
    step = 10500
    N_batches = len(train_dataloader)

    wrapper.train()
    for name, param in wrapper.frozen_phi.named_parameters():
        param.requires_grad = False

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        for iteration, (x, gt, imgpaths) in enumerate(train_dataloader):
            if iteration<step:
                continue

            x = x.to('cuda')        ### (b, 49, 768)
            gt = gt.to('cuda')      ### (b, 75)
            batch_size = x.shape[0] 
            max_output_len = gt.shape[1]

            ## determine the max length of any caption across the batch
            actual_len = get_max_token_length(gt)
            ## more than 20 caption length creates a large batch of 20 with OOM. Skip
            if actual_len>18:
                continue
            print(f"Iteration {iteration}/{num_batches_train_on}, max caption len {actual_len}", end='\r')

            pred_logits = wrapper(x, gt, actual_len)
            pred_tokens = torch.argmax(pred_logits, dim=-1) ### (b, )
            loss=0
            for i in range(actual_len):
                loss += loss_fn(pred_logits[i], gt.squeeze()[i])/actual_len
            loss.backward(retain_graph=True)

            optimizer.step() 
            epoch_loss += loss.detach().item()
            optimizer.zero_grad()
            if (iteration)%10 == 0: 
                print(f'Iteration: {iteration}, Loss: {loss.detach().item()}')
                wandb.log({"step": iteration, "loss": loss.detach().item()})
                gt_text = tokenizer.decode(gt[-1]).replace('<|endoftext|>', '')
                pred_text = tokenizer.decode(pred_tokens).replace('<|endoftext|>', '')
                print(f"Sample Caption (gt): {gt_text}\nCaption (pred): {pred_text}\n")

            if (iteration)%100 == 0:
                torch.save(wrapper.projection_img.state_dict(), f'/mnt/d/era1chk/projection_{ver}.pth')
                torch.save(wrapper.resblock.state_dict(), f'/mnt/d/era1chk/resblock_{ver}.pth')

                img_pil = Image.open(imgpaths[0])
                pred_text = do_infer(x, f'/mnt/d/era1chk/projection_{ver}.pth', f'/mnt/d/era1chk/resblock_{ver}.pth')
                image = wandb.Image(img_pil, caption=f'gt:{gt_text}, pred:{pred_text}')
                wandb.log({"example":image})

            del x
            del gt
            # del pred_logits

        avg_loss = epoch_loss/(iteration+1) 
        wandb.log({"epoch loss": avg_loss})
        print(f"Epoch {epoch} finished")