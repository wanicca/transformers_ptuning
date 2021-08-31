# transformers_ptuning
An unofficial p-tuning (``GPT understands, too'') wrapper for huggingface transformer models, supporting seq2seq models like T5. The wrapper can make additional token embeddings trainable while freezing parameters of the original model.

## Install

Clone this repo, and run `pip install .`

## Easy way to start

To create a prompt like `_ _ X _ _ _ Y _` (where `_` are trainable prompt tokens, `X` and `Y` are placeholder to fill text), we can the `PTuningWrapper.interval_prompt` function for convenience:

```python
from transformers_ptuning import PTuningWrapper

model = transformers.BertModel.from_pretrained("bert-base-uncased")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
# freeze parameters of original model
for param in model.parameters():
    param.requires_grad = False 
#get the wrapped model
wrapped_model,prompt_func = PTuningWrapper.interval_prompt(
    model,tokenizer,intervals=(2,3,1)
)
#input example
input_text = prompt_func("This is piece one","This is piece two")
#" <prompt_0> <prompt_1> This is piece one <prompt_2> <prompt_3> <prompt4> This is piece two <prompt_5>"
tokenized = tokenizer(input_text,return_tensors='pt')
out = wrapped_model(**tokenized)
```

Given the text content for placeholders, the `prompt_func` function can fill the blank in the prompt and return the result, which can be then tokenized. The `wrapped_model` will handle other things and make the gradients correctly computed. You should feed keyword parameters to `wrapped_model` for forward-pass.

### Changing the prompt encoder

The prompt encoder provides the embeddings of prompt tokens. The default prompt encoder is based on the vanilla `torch.nn.Embedding`. As paper ``GPT understands, too'' suggests, the prompt embeddings can be obtained from a lite neural network (e.g. LSTM) during training. So you can change the prompt encoder type.

```python
wrapped_model,prompt_func = PTuningWrapper.interval_prompt(
    model,tokenizer,intervals=(2,3,1),prompt_encoder_type="lstm"
)
```

You can also write your own prompt_encoder and pass the class to `prompt_encoder_type`

### Seq2seq model

For seq2seq models (e.g. T5), if you want to use only the encoder prompt, the case is just the same as aforesaid. If you want to use an additional decoder prompt, you should set `decoder_intervals`.

```python
model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = transformers.T5Tokenizer.from_pretrained("t5-small")
for param in model.parameters():
    param.requires_grad = False 
wrapped_model,prompt_func = PTuningWrapper.interval_prompt(
    model,tokenizer,intervals=(0,0),decoder_intervals=(1,2)
)
input_text, target_text = prompt_func("piece one","piece two")
tokenized = tokenizer(input_text,return_tensors="pt")
tokenized_decoder_labels = tokenizer(example[1],return_tensors="pt")
tokenized['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
    tokenized_decoder_labels['input_ids']
)
out = wrapped_model(**tokenized)
```

`(0,0)` is corresponding to null prompt (i.e. only a placeholder for input text, without any prompt token).

### Multiple prompts

The wrapper is only for one prompt. However, you can wrap the model in multiple times to get multiple wrappers, each for different prompt. **Be sure** to set different `special_prefix` so that the prompt tokens won't be duplicated.

```python
wrapped_model_A,prompt_func_A = PTuningWrapper.interval_prompt(
    model,tokenizer,intervals=(2,3),special_prefix="promptA"
)
wrapped_model_B,prompt_func_B = PTuningWrapper.interval_prompt(
    model,tokenizer,intervals=(2,3),special_prefix="promptB"
)
```

### Updating weights

You may want to add the prompt token embeddings into the original model after training. 

```python
model = wrapped_model.update_model_weight()
```

**Caution:** If you use the original output head of the model to get output, be sure that updating embedding weights may also influence the output head, which means it may output prompt tokens. To avoid that, a forward hook should be applied to the output layer so that the logits are trimmed.

```python
original_vocab_size = ... #
model.get_output_embeddings().register_forward_hook(lambda m,i,o:o[:,:,:original_vocab_size])
```

### More details

Please refer to docstring for more details.

## References

[Official Repo for ``GPT understands, too''](https://github.com/THUDM/P-tuning)