import os
from PIL import Image
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, AutoProcessor, AutoConfig, PhiModel
from torchinfo import summary
import clip

torch.set_default_device("cuda")


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
    

class PhiWrapper(nn.Module):
    def __init__(self, max_token_len_data, input_dim_CLIP=768, input_dim_phi2=2560, ):
        super().__init__()

        self.max_token_len_data = max_token_len_data
        PHI = 'microsoft/phi-2'
        self.phi_tokenizer = AutoTokenizer.from_pretrained(PHI, trust_remote_code=True)
        self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token

        self.frozen_phi:PhiModel = AutoModelForCausalLM.from_pretrained(PHI, trust_remote_code=True)
        self.frozen_phi.config.eos_token_id = self.phi_tokenizer.eos_token_id
        self.frozen_phi.config.bos_token_id = self.phi_tokenizer.bos_token_id
        ## Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        for name, param in self.frozen_phi.named_parameters():
            param.requires_grad = False

        self.input_dim_CLIP = input_dim_CLIP
        self.input_dim_phi2 = input_dim_phi2
        self.projection_img = nn.Linear(self.input_dim_CLIP, self.input_dim_phi2, bias=False)
        self.resblock = SimpleResBlock(self.input_dim_phi2)
        self.clip_model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        self.clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')

        self.bos_embedding  = self.frozen_phi.get_input_embeddings()(
            torch.tensor(self.phi_tokenizer.bos_token_id).to(device='cuda')).unsqueeze(0)

        instruct_part1 = self.phi_tokenizer('Instruction:Caption this image:', return_tensors='pt')
        self.instruct_part1_embedding = self.frozen_phi.get_input_embeddings()(
            instruct_part1.input_ids.to(device='cuda')
            ).squeeze(0)
        
        instruct_part2 = self.phi_tokenizer('Answer:', return_tensors='pt')
        self.instruct_part2_embedding = self.frozen_phi.get_input_embeddings()(
            instruct_part2.input_ids.to(device='cuda')
            ).squeeze(0)
        

    def forward(self, x, caption_tokenized):
        x = self.projection_img(x)
        x = self.resblock(x)
        batch_size = x.shape[0]
        max_output_len = caption_tokenized.shape[1]
        vocab_size = 51200

        ## form a vertical vector: bos | instruction part1 | image emb | instruction part2. (b, 60, 2560)
        x = torch.cat(
            (
                self.bos_embedding.repeat(batch_size, 1, 1), 
                self.instruct_part1_embedding.repeat(batch_size, 1, 1), 
                x,
                self.instruct_part2_embedding.repeat(batch_size, 1, 1)
            ), 
            dim=1
        )

        loss = 0
        word_output_pred_tokens = torch.empty(batch_size, max_output_len)
        ## greedy loop, get output one token at a time
        for idx in range(max_output_len):
            pred_word = self.frozen_phi.generate(
                inputs_embeds=x,
                max_new_tokens = 1, 
                output_scores=True, 
                return_dict_in_generate = True, 
                pad_token_id=self.phi_tokenizer.pad_token_id, 
                bos_token_id=self.phi_tokenizer.bos_token_id, 
                eos_token_id=self.phi_tokenizer.eos_token_id
            )
            pred_word_probs = F.softmax(pred_word.scores[0], dim=-1)
            word_output_pred_tokens[:, idx] = pred_word.sequences[:, 1]
            vocab_size = pred_word.scores[0].shape[1]

            ## get the GT across batch at the idx th position of output
            gt_word_token = caption_tokenized[:,idx]
            # gt_word_one_hot = torch.nn.functional.one_hot(gt_word_token, vocab_size).to(torch.float32)
            if sum(torch.eq(torch.tensor([self.phi_tokenizer.pad_token_id]*batch_size), gt_word_token)) == torch.tensor(batch_size): 
                break 
            ## feature forcing!, send in next GT to help generation along right track
            ## to stop feature forcing, use embedding of pred_word.sequences[:, 1] instead of gt_word_token
            gt_word_embedding = self.frozen_phi.get_input_embeddings()(gt_word_token).unsqueeze(1)
            x = torch.cat((x, gt_word_embedding), dim=1)

            loss_at_idx = F.cross_entropy(
                pred_word_probs, 
                gt_word_token, 
                ignore_index=self.phi_tokenizer.pad_token_id, 
                label_smoothing=0.1
                )
            loss += loss_at_idx
            # print(f'loss at index {idx}: {loss_at_idx}')
            # loss_tosend = loss/batch_size

        return loss, word_output_pred_tokens

    def create_attn_mask(self, image_embeds:torch.Tensor, batch_captions:list):
        ## image features is 49,2560
        ## create embeddings like embedding of (SOS) + image_features + embedding of EOS + embedding of tokenid('Summarize this:')

        batch_size = image_embeds.shape[0]
        batched_sos = self.frozen_phi.get_input_embeddings()(torch.tensor(self.phi_tokenizer.eos_token_id).view((1,1)).repeat(batch_size, 1))
        batched_eos = self.frozen_phi.get_input_embeddings()(torch.tensor(self.phi_tokenizer.eos_token_id).view((1,1)).repeat(batch_size, 1))

        instruct_ids = self.phi_tokenizer('Summary is:', return_tensors='pt')
        batched_instruct_embeds = self.frozen_phi.get_input_embeddings()(instruct_ids.input_ids).repeat(batch_size, 1, 1)

        caption_ids = self.phi_tokenizer(batch_captions, return_tensors='pt')
        batched_caption_embeds = self.frozen_phi.get_input_embeddings()(caption_ids.input_ids)
        input = torch.cat(
            (
                batched_sos,
                image_embeds,
                batched_eos,
                batched_instruct_embeds,
                batched_caption_embeds
            ), dim=1
        )

        ## assuming caption has 10 tokens, and 'summary is:' has 2 tokens then we have a matrix of 11 rows x (1+49+1+2+11) 
        ## = 11x64
        input = input.repeat(len(caption_ids)+1)

        seq_len = input.shape[1] ## 53
        left = torch.zeros((len(caption_ids)+1, seq_len))
        right = torch.full((len(caption_ids)+1, len(caption_ids)+1), float('--inf'))
        attn_mask = torch.cat((left, right), dim=1)
        return attn_mask


    def forward_old(self, image_inputs:list=[], captions:[]=None):
        # img_inputs = self.clip_processor(images=image_inputs, return_tensors='pt', do_resize=False, do_center_crop=False, size={"shortest_edge": 512})
        img_inputs = self.clip_processor(images=image_inputs, return_tensors='pt')
        with torch.no_grad():
            image_features = self.clip_model(**img_inputs, output_hidden_states=True)
            ## skip cls token
            image_features = image_features.last_hidden_state[:, 1:]
            image_features = self.projection_img(image_features)
            image_features = self.resblock(image_features)

            attn_mask = self.create_attn_mask(image_features, captions)

            while (1):
                outputs = self.frozen_phi.generate(inputs_embeds=image_features, max_new_tokens=1)

                ## TODO to use this code the input_length should be max_new_tokens perhaps
                # outputs = self.frozen_phi.generate(inputs_embeds=image_features, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                # transition_scores = self.frozen_phi.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                # input_length = 1 if self.frozen_phi.config.is_encoder_decoder else inputs.input_ids.shape[1]
                # generated_tokens = outputs.sequences[:, input_length:]
                # for tok, score in zip(generated_tokens[0], transition_scores[0]):
                #     # | token | token string | logits | probability
                #     print(f"| {tok:5d} | {self.phi_tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

                ## this might be garbage as input embedding is not what phi2 might have seen
                
                text = self.phi_tokenizer.batch_decode(outputs)[0]
                print(text)
                new_embeds = self.frozen_phi.model.embed_tokens(outputs)
                ## take secondd output
                new_embeds = new_embeds[:, 1:, :]
                ## now we need to append this output back into image_features and loop
                image_features = torch.cat((image_features, new_embeds), dim=1)

            return text
            

if __name__ == "__main__":
    wrapper = PhiWrapper()

    ## this is all fake, just to see what tokenizer and summary emit about the model
    # summary(phi.frozen_phi)
    # prompt = 'Summarize this: an apple is very healthy fruit. Eating an apple keeps on healthy. No need for a doctor. We will not fall sick'
    # # prompt = phi.phi_tokenizer(f'Instruct: {prompt}\nOutput:', return_tensors="pt", return_attention_mask=False).to('cuda')
    # prompt = wrapper.phi_tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
    # with torch.no_grad():
    #     outputs = wrapper.frozen_phi.generate(**prompt, max_length=50)
    #     text = wrapper.phi_tokenizer.batch_decode(outputs)[0]
    #     print(text)

    ## Image input part
    '''
    phi2 fwd method inputs:
        def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

    either we can pass input_ids (b,n) from phi2-tokenizer or we can pass inputs_embeds (b,n,2560)
    position ids can be None, then its set to torch.arange(n)
    attention_mask can be None, its computed inside to be triangular matrix with lower triangle=0 and upper triangle=-65504
    32 decoder layers!
    '''
    img_pil = Image.open('TrainProjection/balloons.png')
    caption = 'painting of a woman in a purple dress walking in a field with hot air balloons'
    img_pil = img_pil.resize((512,512), resample=Image.LANCZOS)
    text = wrapper(image_inputs=[img_pil, img_pil], captions=[caption, caption])
    print(text)

        
