import torch
import torch.nn as nn
import torch.nn.functional as F

import peft
from peft.peft_model import PeftModelForCausalLM

class CustomGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x.clone())
    
class SimpleResBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_size)
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    

class Phi2wrapper(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, 
                 phi2_model:PeftModelForCausalLM, ## peft_model
                 tokenizer, 
                 input_dim_CLIP=768, 
                 input_dim_phi2=2560, 
                 ):
        
        super(Phi2wrapper, self).__init__()

        self.input_dim_CLIP = input_dim_CLIP
        self.input_dim_phi2 = input_dim_phi2
        self.projection_img = nn.Linear(self.input_dim_CLIP, self.input_dim_phi2, bias=False)
                                                                                                                                                           
        self.resblock = SimpleResBlock(self.input_dim_phi2)
        self.phi2_model = phi2_model
        self.tokenizer = tokenizer

        self.device = 'cuda'

        bos = self.tokenizer("Context: ", return_tensors="pt", return_attention_mask=False)
        eoi = self.tokenizer(" Question: ", return_tensors="pt", return_attention_mask=False)
        eoa = self.tokenizer(" Answer: ", return_tensors="pt", return_attention_mask=False)
    
        self.bos_embedding = self.phi2_model.base_model.get_input_embeddings()(bos.input_ids.to(self.device)).squeeze(0)
        self.eoi_embedding = self.phi2_model.base_model.get_input_embeddings()(eoi.input_ids.to(self.device)).squeeze(0)
        self.eoa_embedding = self.phi2_model.base_model.get_input_embeddings()(eoa.input_ids.to(self.device)).squeeze(0)
        
        self.eos_embedding = self.phi2_model.base_model.get_input_embeddings()(torch.tensor(self.tokenizer.eos_token_id).to(self.device)).unsqueeze(0)


    def create_input2(self, image_embedding, input_q_embedding, gt, gt_token_len):
        batch_size = image_embedding.shape[0]
        x = torch.cat((self.bos_embedding.repeat(batch_size,1,1),  # (1, 3, 2560)
                       image_embedding, 
                       self.eoi_embedding.repeat(batch_size,1,1),  # (1, 3, 2560)
                       input_q_embedding, 
                       self.eoa_embedding.repeat(batch_size,1,1)),  # (1, 3, 2560)
                      dim=1
                    )  # (1, n+9, 2560)
        inputs_len = x.shape[1]
        '''
        x is like xxxxxxxxxxx (1, inputs_len)
        '''
        x = x.repeat(gt_token_len, 1, 1)
        '''
        x is like below (b, inputs_len)
        xxxxxxxxxxx
        xxxxxxxxxxx
        xxxxxxxxxxx
        xxxxxxxxxxx
        '''

        '''
        gt is like 
        cccc
        
        need to generate
        pppp
        cppp
        ccpp
        cccp
        '''
        gt = gt.repeat(gt_token_len, 1)
        '''
        cccc
        cccc
        cccc
        cccc
        '''
        mid_mask = 1-torch.triu(torch.ones(gt_token_len, gt_token_len), diagonal=0)
        '''
        0000
        1000
        1100
        1110
        '''
        gt = gt * mid_mask.to('cuda').int()
        '''
        0000
        c000
        cc00
        ccc0
        '''
        gt = gt.masked_fill(gt==0, self.tokenizer.pad_token_id)
        '''
        pppp
        cppp
        ccpp
        cccp
        '''
        ## convert to embedding
        x = torch.cat((x, self.phi2_model.base_model.get_input_embeddings()(gt)), dim=1)
        '''
        attn mask as below
        11111111111|0000
        11111111111|1000
        11111111111|1100
        11111111111|1110

        x*attn_mask.masked_fill(x==0, p) like below each predicts one word of caption
        xxxxxxxxxxx0000
        xxxxxxxxxxxc000
        xxxxxxxxxxxcc00
        xxxxxxxxxxxccc0

        leftmask|midmask
        '''
        left_mask = torch.ones((gt_token_len, inputs_len))
        attn_mask = torch.cat((left_mask, mid_mask), dim=1)

        return x, attn_mask.to('cuda')


    def forward(self, image_embedding, input_q, gt, gt_token_len):
        image_embedding = self.projection_img(image_embedding)
        image_embedding = self.resblock(image_embedding) # (1, 49, 2560)
        input_q_embedding = self.phi2_model.base_model.get_input_embeddings()(input_q)  # (1, n, 2560)

        x, attn_mask = self.create_input2(image_embedding, input_q_embedding, gt, gt_token_len)
        x = x.to(dtype=torch.float16)
        attn_mask = attn_mask.to(dtype=torch.float16)
        pred = self.phi2_model(inputs_embeds=x, attention_mask=attn_mask)  ## (gt_token_len, x.shape[1], 51200)
        pred_logits = pred.logits[:,-1,:]  ## (gt_token_len, 51200)
        
        del x
        del pred
        del attn_mask
        return pred_logits
