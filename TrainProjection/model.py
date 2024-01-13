import os
from PIL import Image

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary
import clip

class PhiWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        ## need to pip install flash_attn
        # self.frozen_phi = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, trust_remote_code=True)
        # self.frozen_phi = self.frozen_phi.to('cuda')
        # self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16)

        # for name, param in self.frozen_phi.named_parameters():
        #     param.requires_grad = False

        # self.input_dim_CLIP = 768
        # self.input_dim_phi2 = 2560
        # self.projection_img = nn.Linear(self.input_dim_CLIP, self.input_dim_phi2, bias=False)

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device='cuda')


    def forward(self, text_inputs=None, image_inputs:Image=None):
        if text_inputs is not None:
            with torch.no_grad():
                outputs = self.frozen_phi.generate(**inputs, max_length=200)
                text = self.phi_tokenizer.batch_decode(outputs)[0]
                return text
        
        if image_inputs is not None:
            img_t = self.preprocess(image_inputs).unsqueeze(0).to(device='cuda')
            with torch.no_grad():
                ## image_features.shape == (1, 512) ???
                image_features = self.clip_model.encode_image(img_t)
                image_proj = self.projection_img(image_features)
                outputs = self.frozen_phi.generate(inputs_embeds=image_proj, max_length=200)
                ## this might be garbage as input embedding is not what phi2 might have seen
                text = self.phi_tokenizer.batch_decode(outputs)[0]
                return text
            

if __name__ == "__main__":
    phi = PhiWrapper()

    ## this is all fake, just to see what tokenizer and summary emit about the model
    # summary(phi.frozen_phi)
    # prompt = 'hi model, whats up?'
    # inputs = phi.phi_tokenizer(f'Instruct: {prompt}\nOutput:', return_tensors="pt", return_attention_mask=False).to('cuda')
    # text = phi(text_inputs=inputs)
    # print(text)

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
    ## caption: painting of a woman in a purple dress walking in a field with hot air balloons
    img_pil = img_pil.resize((512,512), resample=Image.LANCZOS)
    text = phi(image_inputs=img_pil)
    print(text)

     
