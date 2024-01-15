import h5py    
import numpy as np   
import os
import torch
from torch.utils.data import Dataset

class COCO_CLIP_Dataset(Dataset):
    def __init__(
        self, caption_file, embedding_path, tokenizer, max_token_len_data):
        self.embedding_path = embedding_path
        self.caption_file = caption_file
        self.tokenizer = tokenizer
        self.max_token_len_data = max_token_len_data

    def __len__(self):
        return len(self.caption_file)
    
    def __getitem__(self, index):
        row = self.caption_file.iloc[[index]]
        df_img = row['image_id'].values[0]
        img_base_name = '0'*(12-len(str(df_img))) + str(df_img)
        img_base_name = img_base_name.replace(' ', '0')
        img_clip_embedding_path = os.path.join(self.embedding_path, f'{img_base_name}.h5')

        clip_embed = h5py.File(img_clip_embedding_path,'r+')['image_features'][()]
        
        img_caption = row['caption'].values[0] ## Tokenize this 
        img_caption_tokenized = self.tokenizer(img_caption, return_tensors="pt", 
                                               return_attention_mask=False).input_ids
        input_tokenized = torch.cat((img_caption_tokenized, 
                               torch.tensor(self.tokenizer.eos_token_id).view((1,1))), dim=1)
        
        if (self.max_token_len_data - input_tokenized.shape[1]) > 0: 
            pad_tokenized = torch.tensor([self.tokenizer.pad_token_id]*(self.max_token_len_data - input_tokenized.shape[1])).unsqueeze(0)
            input_final =  torch.cat((input_tokenized, pad_tokenized), dim=1)
        else: 
            input_final = input_tokenized
        
        return torch.tensor(clip_embed).squeeze(0), input_final.squeeze(0)