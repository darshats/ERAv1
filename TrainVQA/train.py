import pandas as pd
import peft
from peft import LoraConfig
from PIL import Image
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import  AutoTokenizer
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from model import Phi2wrapper
from dataset import LLAVA_150k_Dataset_Instruct
from infer import do_infer

device = 'cuda'

def get_max_token_length(gt):
    token_len = 1
    for idx in range(gt.shape[1]):
        ## get the GT across batch at the idx th position of output
        gt_token = gt[:, idx]
        batch_size = gt.shape[0]
        ## if across the batch we only have pads then break
        num_pad = sum(torch.eq(torch.tensor([tokenizer.pad_token_id]*batch_size, device='cuda'), gt_token))
        if num_pad.detach().item() == batch_size:
            break 

        token_len += 1
    return token_len


def train(model:Phi2wrapper, num_epochs, train_dataloader, optimizer):
    model.train()
    ver = 'v103'
    wandb.login()
    wandb_run = wandb.init(project="Capstone, part 2", dir='/mnt/d/wandb/tmp', id=ver)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"Working on epoch {epoch}")
        for iteration, ((img_embedding, question), gt, imgpath) in enumerate(train_dataloader):
            img_embedding = img_embedding.to('cuda') # (1,49,768)
            question = question.to('cuda') # (1, n)
            gt = gt.to('cuda') #(1, m)
            if gt.shape[1] > 15:
                # print(f'Skip tokens>15: {gt.shape[1]}')
                continue

            optimizer.zero_grad()

            ## determine the max length of any caption across the batch
            actual_len = get_max_token_length(gt)-1

            pred_logits = model(img_embedding, question, gt, actual_len)
            pred_tokens = torch.argmax(pred_logits, dim=-1) ### (b, )
            loss = 0
            for i in range(actual_len):
                loss += loss_fn(pred_logits[i], gt.squeeze()[i]) / actual_len

            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.detach().item()

            if (iteration % 50) == 0: 
                print(f'Iteration: {iteration}, Loss: {loss.detach().item()}')
                wandb.log({"step": iteration, "loss": loss.detach().item()})

                question_text = tokenizer.batch_decode(question, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                gt_text = tokenizer.batch_decode(gt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                pred_text = tokenizer.batch_decode(pred_tokens.unsqueeze(dim=0), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print("Question:", question_text)
                print("Predictions:", pred_text)
                print("Gt answer:", gt_text)

            if iteration%100 == 0:
                peft_path = f'/mnt/d/era2chk/peft_{ver}.pth'
                res_path = f'/mnt/d/era2chk/resblock_{ver}.pth'
                proj_path = f'/mnt/d/era2chk/projection_{ver}.pth'
                # torch.save(model.phi2_model.state_dict(), phi2_path) 
                torch.save(model.resblock.state_dict(), res_path) 
                torch.save(model.projection_img.state_dict(), proj_path)
                model.phi2_model.save_pretrained(peft_path)

                img_pil = Image.open(imgpath[0])
                pred_text = do_infer(img_embedding, question, peft_path, res_path, proj_path)
                image = wandb.Image(img_pil, caption=f'q:{question_text}, gt:{gt_text}, pred:{pred_text}')
                wandb.log({"example":image})
                
            del img_embedding
            del question
            del gt
            # del pred_logits

        print("")
        print(f"Epoch {epoch} finished")
        print("")


if __name__ == '__main__':

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name = "microsoft/phi-2"
    phi2_model_pretrained = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,  
        quantization_config=bnb_config,
    )
    phi2_model_pretrained.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
        ]
    )

    torch.set_grad_enabled(True)  
    peft_model = peft.get_peft_model(phi2_model_pretrained, peft_config)
    peft_model.print_trainable_parameters()

    phi2_projection_model = Phi2wrapper(phi2_model=peft_model, tokenizer=tokenizer)
    phi2_projection_model.to('cuda')
    phi2_projection_model.projection_img.load_state_dict(torch.load('/mnt/d/era2chk/projection_v101.pth'))
    phi2_projection_model.resblock.load_state_dict(torch.load('/mnt/d/era2chk/resblock_v101.pth'))
    phi2_projection_model.phi2_model.load_state_dict(torch.load('/mnt/d/era2chk/phi2_v101.pth'))

    optimizer = torch.optim.Adam(phi2_projection_model.parameters(), lr=1e-5, eps=1e-9) 

    llava_json_df = pd.read_csv('/home/dshah/ERAv1/TrainVQA/llava_instruct_150k_df.csv')
    dataset = LLAVA_150k_Dataset_Instruct(
        embedding_path='/home/dshah/ERAv1/embeddings', 
        llava_json_df=llava_json_df,
        tokenizer=tokenizer
        )
    # (x1, x2), y = dataset[0]
    # print(x1.shape, x2.shape, y.shape)
    # phi2_projection_model(x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device), y.unsqueeze(0).to(device))

    batch_size_train = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=False, num_workers=8)
    print(f'Number of batches {len(train_dataloader)}')

    print(f'Len train_dataloader {len(train_dataloader)}')
    train(phi2_projection_model, num_epochs=1, train_dataloader=train_dataloader, optimizer=optimizer)