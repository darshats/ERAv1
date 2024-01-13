import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary

class PhiWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        ## need to pip install flash_attn
        self.frozen_phi = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, trust_remote_code=True)
        self.frozen_phi = self.frozen_phi.to('cuda')
        self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16)


    def forward(self, x):
        outputs = self.frozen_phi.generate(**inputs, max_length=200)
        text = self.phi_tokenizer.batch_decode(outputs)[0]
        return text


if __name__ == "__main__":
    phi = PhiWrapper()

    ## this is all fake, just to see what tokenizer and summary emit about the model
    summary(phi.frozen_phi)
    prompt = 'hi model, whats up?'
    inputs = phi.phi_tokenizer(f'Instruct: {prompt}\nOutput:', return_tensors="pt", return_attention_mask=False).to('cuda')
    text = phi(inputs)


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

    print(text)
