This repo captures part 1 of the capstone.

What has been done:
- projection of image to common embeddings space
- train projection + frozen phi2 with coco dataset of image+captions
- studied forward and generate methods of phi2 model and its super classes. Generate is marked torch.nograd hence cannot be used during training, but reusing forward is an option
- embeddings obtained using VIT and 224/32 = 7 so 49 patches each of length 768. This was extended to 49, 2568
- two variations were tried:
1) each batch has different (image,caption) as (x, gt). As captions are of different length, max lenght is chosen across all batches and padded accordingly. Data analysis showed most captions were within 30 length.
The input x was encoded as embeddng("Image:") + <image projected to 49,2560> + embedding(" Caption:"). Teacher forcing used for first 4 tokens. Loss computed was cross entropy. Looked at BLEU but not implemented. This was carried forward to part 2

2) second variation was where batchsize was one. But each caption was used to created incrementally increasing decoder lengths. So a caption of 10 tokens will give 10 inputs in them minibatch, each parallel decoding of a larger prefix of the caption. Attention mask was computed similar to triangular matrix but since image is not causal that part of mask was a full rectangle. Finally forward method is called with input_embeds and attention_mask. Loss decreased faster in this approach but couldnt carry forward to part2 due to lack of time

The e2e project is present here - https://github.com/lavanyanemani96/TSAI-ERA-Phase-I/tree/master/Capstone - as a group project.

Things to try:
- better loss function since there are multiple possible captions e.g. BLEU
- distributed training. We had 64k image-caption pairs from coco but barely got through 2 epochs
