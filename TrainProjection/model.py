import os
from PIL import Image
import numpy as np

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, AutoProcessor, AutoConfig, GenerationMixin
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
    def __init__(self):
        super().__init__()

        PHI = 'microsoft/phi-2'
        self.phi_tokenizer = AutoTokenizer.from_pretrained(PHI, trust_remote_code=True)
        self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token

        self.frozen_phi:GenerationMixin = AutoModelForCausalLM.from_pretrained(PHI, trust_remote_code=True)
        self.frozen_phi.config.eos_token_id = self.phi_tokenizer.eos_token_id
        self.frozen_phi.config.bos_token_id = self.phi_tokenizer.bos_token_id
        ## Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        for name, param in self.frozen_phi.named_parameters():
            param.requires_grad = False

        self.input_dim_CLIP = 768
        self.input_dim_phi2 = 2560
        self.projection_img = nn.Linear(self.input_dim_CLIP, self.input_dim_phi2, bias=False)
        self.resblock = SimpleResBlock(self.input_dim_phi2)
        self.clip_model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')


    def forward(self, image_inputs:Image=None):
        # img_inputs = self.clip_processor(images=image_inputs, return_tensors='pt', do_resize=False, do_center_crop=False, size={"shortest_edge": 512})
        img_inputs = self.clip_processor(images=image_inputs, return_tensors='pt')
        with torch.no_grad():
            image_features = self.clip_model(**img_inputs, output_hidden_states=True)
            ## skip cls token
            image_features = image_features.last_hidden_state[:, 1:]
            image_features = self.projection_img(image_features)
            image_features = self.resblock(image_features)

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
            

# if __name__ == "__main__":
wrapper = PhiWrapper()

## this is all fake, just to see what tokenizer and summary emit about the model
# summary(phi.frozen_phi)
prompt = 'Summarize this: an apple is very healthy fruit. Eating an apple keeps on healthy. No need for a doctor. We will not fall sick'
# prompt = phi.phi_tokenizer(f'Instruct: {prompt}\nOutput:', return_tensors="pt", return_attention_mask=False).to('cuda')
prompt = wrapper.phi_tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
with torch.no_grad():
    outputs = wrapper.frozen_phi.generate(**prompt, max_length=50)
    text = wrapper.phi_tokenizer.batch_decode(outputs)[0]
    print(text)

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
text = wrapper(image_inputs=img_pil)
print(text)

    
