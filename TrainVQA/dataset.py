from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import json
import os 
import torch
import h5py    
import ast
import numpy as np    

class LLAVA_150k_Dataset_Instruct(Dataset):

    def __init__(self, 
                 embedding_path, 
                 llava_json_df, 
                 tokenizer, 
                 max_token_len_q=15, 
                 max_token_len_a=40
                ):
        
        self.embedding_path = embedding_path
        self.llava_json_df = llava_json_df
        self.tokenizer = tokenizer
        self.max_token_len_q = max_token_len_q
        self.max_token_len_a = max_token_len_a

    def __len__(self):
        return len(self.llava_json_df)
    
    def __getitem__(self, index):

        row = self.llava_json_df.iloc[[index]]

        df_img = row['image_id'].values[0]
        img_base_name = '0'*(12-len(str(df_img))) + str(df_img)
        img_base_name = img_base_name.replace(' ', '0')
        img_clip_embedding_path = os.path.join(self.embedding_path, f'{img_base_name}.h5')
        imgpath = f'/mnt/d/train2017/{img_base_name}.jpg'

        np_array_embed_img = h5py.File(img_clip_embedding_path,'r+')['image_features'][()]
        
        Q_human_tokenized = ast.literal_eval(row['Q_human_tokenized'].values[0])
        A_gpt_tokenized = ast.literal_eval(row['A_GPT_tokenized'].values[0])
        
        return (torch.tensor(np_array_embed_img).squeeze(0), torch.tensor(Q_human_tokenized)), torch.tensor(A_gpt_tokenized), imgpath
