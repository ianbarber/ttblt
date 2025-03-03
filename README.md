# ttblt

A simplified implementation of Byte Latent Transformers as a TorchTune recipe.

https://github.com/facebookresearch/blt is the original (and much more comprehensive!) repo. This project focuses just on fine-tuning an existing pretrained model, and is presented as a TorchTune recipe. 

```
tune run full_finetune_single_device.py --config qwen2_5_3B_blt_full_single_device.yaml
```

The implementation is:
* a simple tokenizer that just takes UTF-8 bytes
* a small local encoder transformer
* simplified patching logic that uses a local entropy measure within sequences to determine patching, with some very basic thresholding
* adding cross-attention onto existing layers from the pretrained model to attend to the patches

The example uses Qwen 2.5 3B (as the original paper experimented with Llama, so I figured some variety would be interesting). It uses the Alpaca dataset, in pretty much the standard Torchtune single device fine tune recipe.  

Note that the recipe is modified to:
* Allow non-strict loading of the Qwen checkpoint as we have the extra BLT params
* Specifically filter out the token embeddings, as we dump those in favor of learning a simple byte specific embedding
* Add a patch based generation function, and load the unusual checkpoint

For memory purposes the cross-attention is limited to the 1/3 of the layers of the Qwen model - it runs in 24GB of VRAM, but pretty slowly (there is a definitely a lot to optimize and no profiling has been done).  
