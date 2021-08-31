#Easy版本，wrapper只处理一个模板。
#暂时没考虑encoder和decoder的tokenizer不同的情况，以后可以给decoder全套的prompt_token_fn

import re
import transformers
import torch
import torch.nn.functional as F
def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)

class PTuningWrapper(torch.nn.Module):
    def __init__(self,model,prompt_encoder,decoder_prompt_encoder=None,
        prompt_token_fn=None, prompt_token_id=None, prompt_token_ids=None,
        replacing_token_id=0):
        """
        PTuningWrapper for Huggingface transformer models (Encoder Models).
        It will replace the prompt token embeddings with ones from prompt encoder.
        Therefore the origin embedding layer can be freezed.

        Parameters:
            model: 
                The huggingface transformer model to be wrapped.
            prompt_encoder: 
                A model that returns corresponding embeddings for prompt ids.
            decoder_prompt_encoder:
                Similar to prompt_encoder, but used for decoder part of the 
                model if the model is encoder-decoder (e.g. T5 and BART).
            prompt_token_fn:
                A function that receives the input_ids and returns a boolean
                tensor to tell which ones are prompt tokens. This parameter 
                conflicts with prompt_token_id and prompt_token_ids.
            prompt_token_id:
                If there is only one possible prompt token id, use this
                parameter.
            prompt_token_ids:
                If there is a list of prompt token ids, use this paramter.
            replacing_token_id:
                During processing, all prompt token will be replaced with this
                one so that the input_ids can be passed into the original 
                embedding layer of the transformer model.
        """
        super().__init__()
        self.underlying_model = model
        self.original_embedding = model.get_input_embeddings()
        self.prompt_encoder = prompt_encoder
        self.decoder_prompt_encoder = decoder_prompt_encoder
        self.replacing_token_id = replacing_token_id
        if self.decoder_prompt_encoder:
            self.decoder_original_embedding = model.decoder.embed_tokens
        if prompt_token_fn is not None:
            assert prompt_token_id is None and prompt_token_ids is None, \
                "Use one of prompt_token_fn, prompt_token_id, prompt_token_ids"
            self.prompt_token_fn = prompt_token_fn
        elif prompt_token_ids is not None:
            assert prompt_token_id is None, \
                "Use one of prompt_token_fn, prompt_token_id, prompt_token_ids"
            self.prompt_token_ids = torch.nn.parameter.Parameter(
                torch.tensor(prompt_token_ids,device=model.device))
            self.prompt_token_fn = lambda t:_isin(t,self.prompt_token_ids)
        elif prompt_token_id is not None:
            self.prompt_token_id = prompt_token_id
            self.prompt_token_fn = lambda t:(t==prompt_token_id)
        else:
            # self.original_embedding_size = self.model.get_input_embeddings()\
            #     .num_embeddings
            self.original_embedding_size = self.model.config.vocab_size
            self.prompt_token_fn = lambda t:(t>=self.original_embedding_size)
            
            #Default: All token ids beyond num_embeddings are seen as prompt token

    def forward(self,input_ids,decoder_input_ids=None,prompt_ids=None,**kwargs):
        prompt_masks = self.prompt_token_fn(input_ids)
        #由于本wrapper只处理单个prompt，所以长度应该是一样的，可以使用masked_select再reshape
        if prompt_masks.any():
            input_ids_ = input_ids.clone()
            if self.replacing_token_id is not None:
                input_ids_[prompt_masks]=self.replacing_token_id
            inputs_embeds = self.original_embedding(input_ids_)
            prompt_embeds = self.prompt_encoder(input_ids[prompt_masks],\
                prompt_ids).to(device=inputs_embeds.device)
            #不能用masked_select，这个是创建新的tensor，修改它不会改原先的变量
            #应该用masked_scatter，或者indexput
            # inputs_embeds.masked_scatter_(prompt_masks.unsqueeze(-1),
            #     prompt_embeds.expand(inputs_embeds.shape[0],-1,-1))
            #使用index_put,后面的expand其实可以换成repeat，反正reshape也会导致copy
            # inputs_embeds[prompt_masks]=prompt_embeds.expand(inputs_embeds.\
            #     shape[0],-1,-1).reshape(-1,inputs_embeds.shape[-1])
            #使用repeat
            # inputs_embeds[prompt_masks]=prompt_embeds.repeat(inputs_embeds.\
            #     shape[0],1)
            #把repeat交给prompt_encoder进行处理
            inputs_embeds[prompt_masks]=prompt_embeds
        else:
            inputs_embeds = self.original_embedding(input_ids)
        
        if decoder_input_ids is not None:
            if self.decoder_prompt_encoder is not None:
                decoder_prompt_masks = self.prompt_token_fn(decoder_input_ids)
                if decoder_prompt_masks.any():
                    decoder_input_ids_ = decoder_input_ids.clone()
                    if self.replacing_token_id is not None:
                        decoder_input_ids_[decoder_prompt_masks] = \
                            self.replacing_token_id
                    decoder_inputs_embeds = self.decoder_original_embedding(
                        decoder_input_ids_
                    )
                    decoder_prompt_embeds = self.decoder_prompt_encoder(
                        decoder_input_ids[decoder_prompt_masks],prompt_ids).to\
                        (device=decoder_inputs_embeds.device)
                    decoder_inputs_embeds[decoder_prompt_masks] = \
                        decoder_prompt_embeds
                else:
                    decoder_inputs_embeds = self.decoder_original_embedding(
                        decoder_input_ids
                    )
                return self.underlying_model(inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds,**kwargs
                )
            else: #decoder_prompt_encoder is not defined, so decoder_originical_embedding is not set.
                return self.underlying_model(inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids,**kwargs)
        else:
            return self.underlying_model(inputs_embeds=inputs_embeds,**kwargs)
    
    def update_model_weight(self):
        if self.prompt_encoder:
            if (self.original_embedding.num_embeddings < 
                self.prompt_encoder.id_offset+self.prompt_encoder.length):
                self.underlying_model.resize_token_embeddings(
                    self.prompt_encoder.id_offset+self.prompt_encoder.length
                )
            self.prompt_encoder.dump_embedding(self.original_embedding.weight)
        if self.decoder_prompt_encoder:
            if (self.decoder_original_embedding.num_embeddings < 
                self.decoder_prompt_encoder.id_offset + 
                self.decoder_prompt_encoder.length):
                self.underlying_model.resize_token_embeddings(
                    self.decoder_prompt_encoder.id_offset+
                    self.decoder_prompt_encoder.length
                )
            self.decoder_prompt_encoder.dump_embedding(
                self.decoder_original_embedding.weight)

    @classmethod
    def interval_prompt(cls,model,tokenizer,intervals,decoder_intervals=None,
        special_prefix="prompt",prompt_encoder_type="embedding",
        return_prompt_string=False,**kwargs):
        """
        Given intervals ,generate a wrapped model, a tokenizer, and a template function.
        
        Examples:
            For prompt "_ _ _ X _ _ Y ", the intervals should be (3,2,0)
            For prompt "X _ Y", the intervals should be (0,1,0)
            For prompt "_ X", the intervals should be (1,0)
            For null prompt "X", the intervals should be (0,0)

        Args:
            model:
                The huggingface transformer model to be wrapped.
            tokenizer:
                The tokenizer for the model. This function will add special 
                tokens in the tokenizer.
            intervals:
                The intervals should be a sequence of integers. 
            decoder_intervals:
                The intervals for decoder. For BERT, GPT and similar models,
                this parameter must be set to `None`. 
            special_prefix:
                The prefix for special tokens. To add more than one prompt, the 
                special_prefix parameters should be different for each prompt.
            prompt_encoder_type:
                Prompt encoder that generates embeddings for prompt tokens. It
                can be set to "embedding" or "lstm", or other subclass of 
                PromptEncoder.
            return_prompt_string:
                If set `True`, return the prompt string.
            kwargs:
                Other parameters for prompt_encoder.
        
        Returns:
            :obj:`PTuningWrapper`: 
                The wrapped model.
            :obj:`Callable`: 
                A function to fill the blank in the prompt and generate a 
                sequence that can be tokenized.
            :obj:`(str,str)`:
                (When set `return_prompt_string=True`) The 
                
        """
        #Assertion
        assert len(intervals)>=2, "intervals should have 2 elements at least."
        assert prompt_encoder_type in ("embedding","lstm") or \
            isinstance(prompt_encoder_type,type)
        #Processing tokenizer and prompt string
        prompt_string = ""
        decoder_prompt_string = ""
        counter = 0
        added_tokens = []
        id_offset = len(tokenizer)
        for interval_id,interval in enumerate(intervals):
            for i in range(interval):
                prompt_string+=f" <{special_prefix}_{counter}>"
                added_tokens.append(transformers.AddedToken(
                    f"<{special_prefix}_{counter}>", lstrip=True, rstrip=False
                ))
                counter+=1
            if interval_id < len(intervals) - 1:
                prompt_string+=" {}"
        prompt_length = counter
        if decoder_intervals is not None:
            for interval_id,interval in enumerate(decoder_intervals):
                for i in range(interval):
                    decoder_prompt_string += f" <{special_prefix}_{counter}>"
                    added_tokens.append(transformers.AddedToken(
                        f"<{special_prefix}_{counter}>", lstrip=True, rstrip=False
                    ))
                    counter+=1
                if interval_id < len(decoder_intervals) - 1:
                    decoder_prompt_string += " {}"
            decoder_prompt_length = counter - prompt_length
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
        if decoder_intervals is not None:
            def prompt_function(*args):
                return (
                    prompt_string.format(*args[:len(intervals)-1]),
                    decoder_prompt_string.format(*args[len(intervals)-1:\
                        len(intervals)+len(decoder_intervals)-2])
                )
        else:
            def prompt_function(*args):
                return prompt_string.format(*args)
        #Get the prompt encoder
        if prompt_encoder_type == "embedding":
            prompt_encoder_type = EmbeddingPromptEncoder
        elif prompt_encoder_type == "lstm":
            prompt_encoder_type = LSTMEmbeddingPromptEncoder
        if prompt_length > 0:
            prompt_encoder = prompt_encoder_type(prompt_length,
                model.config.hidden_size,id_offset,**kwargs)
        else:
            prompt_encoder = None
        if decoder_intervals is not None and decoder_prompt_length > 0:
            decoder_prompt_encoder = prompt_encoder_type(decoder_prompt_length,
                model.config.hidden_size,id_offset+prompt_length,**kwargs)
        else:
            decoder_prompt_encoder = None
        
        #Wrap the model
        id_end = len(tokenizer)
        wrapped_model = PTuningWrapper(model,prompt_encoder,
            decoder_prompt_encoder,
            prompt_token_fn=lambda x: (x>=id_offset)&(x<id_end)
        )
        #Return
        if return_prompt_string:
            return wrapped_model,prompt_function,\
                (prompt_string,decoder_prompt_string)
        else:
            return wrapped_model,prompt_function

class PromptEncoder(torch.nn.Module):
    def __init__(self,length,embedding_dim,id_offset,**kwargs) -> None:
        super().__init__()
        self.length = length
        self.embedding_dim = embedding_dim
        self.id_offset = id_offset
    
    def dump_embedding(self,weight):
        raise NotImplementedError

class EmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,length,embedding_dim,id_offset) -> None:
        super().__init__(length,embedding_dim,id_offset)
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        # self.input_ids = torch.nn.parameter.Parameter(torch.arange(length),
        #     requires_grad=False)
    
    def forward(self,prompt_token_ids,prompt_ids=None):
        prompt_token_ids = prompt_token_ids - self.id_offset
        return self.embedding(prompt_token_ids)

    def dump_embedding(self, weight):
        weight[self.id_offset:self.id_offset+self.length,:]=self.embedding.\
            weight.detach()

class LSTMEmbeddingPromptEncoder(PromptEncoder):
    def __init__(self,length,embedding_dim,id_offset) -> None:
        super().__init__(length,embedding_dim,id_offset)
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        self.input_ids = torch.nn.parameter.Parameter(torch.arange(length),
            requires_grad=False)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            dropout=0.0,
            bidirectional=True,
            batch_first=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self,prompt_token_ids,prompt_ids=None):
        embeds = self.embedding(self.input_ids)
        x = self.lstm(embeds.unsqueeze(0))
        running_weight = self.mlp(x[0]).squeeze(0)
        prompt_token_ids = prompt_token_ids - self.id_offset
        return F.embedding(prompt_token_ids,running_weight)
    def dump_embedding(self, weight):
        with torch.no_grad():
            embeddings = self.forward(self.input_ids+self.id_offset)
        weight[self.id_offset:self.id_offset+self.length,:]=embeddings.detach()


if __name__ == "__main__":
    #Unit Test
    ## 1. BERT
    model = transformers.BertModel.from_pretrained("bert-base-uncased")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"Test BERT")
    print(f"Original tokenizer size: {len(tokenizer)}")
    wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        interval_prompt(
            model,tokenizer,(2,3,1),return_prompt_string=True
        )
    print(f"Prompt length:{wrapped_model.prompt_encoder.length}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print("Prompt string:",prompt_string)
    example = prompt_func("piece one","piece two")
    print("Example:",example)
    tokenized = tokenizer(example,return_tensors="pt")
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    out = wrapped_model(**tokenized)
    print("Try backward")
    loss = torch.sum(out[1])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)
    ## 2. GPT2
    model = transformers.GPT2Model.from_pretrained("gpt2")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    print(f"Test GPT2")
    print(f"Original tokenizer size: {len(tokenizer)}")
    wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        interval_prompt(
            model,tokenizer,(2,3,1),return_prompt_string=True
        )
    print(f"Prompt length:{wrapped_model.prompt_encoder.length}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print("Prompt string:",prompt_string)
    example = prompt_func("piece one","piece two")
    print("Example:",example)
    tokenized = tokenizer(example,return_tensors="pt")
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    out = wrapped_model(**tokenized)
    print("Try backward")
    loss = torch.sum(out[0])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)
    ## 3. T5
    model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")
    print(f"Test T5")
    print(f"Original tokenizer size: {len(tokenizer)}")
    wrapped_model,prompt_func,prompt_string = PTuningWrapper.\
        interval_prompt(
            model,tokenizer,(2,3),(1,1),return_prompt_string=True
        )
    print(f"Prompt length:{wrapped_model.prompt_encoder.length+wrapped_model.decoder_prompt_encoder.length}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print("Prompt string:",prompt_string)
    example = prompt_func("piece one","piece two")
    print("Example:",example)
    tokenized = tokenizer(example[0],return_tensors="pt")
    tokenized_decoder_labels = tokenizer(example[1],return_tensors="pt")
    tokenized['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
        tokenized_decoder_labels['input_ids']
    )
    tokenized['decoder_attention_mask'] = tokenized_decoder_labels['attention_mask']
    print("Tokenized:",tokenized)
    for p in model.parameters():
        p.requires_grad = False
    wrapped_model.zero_grad()
    out = wrapped_model(**tokenized)
    print("Try backward")
    loss = torch.sum(out[0])
    loss.backward()
    print("Original embedding grads:",model.get_input_embeddings().weight.grad)
    print("Prompt embedding grads:", wrapped_model.prompt_encoder.embedding.weight.grad)
    wrapped_model.update_model_weight()
    print(model.get_input_embeddings().weight[wrapped_model.prompt_encoder.id_offset]==\
        wrapped_model.prompt_encoder.forward(torch.tensor([wrapped_model.prompt_encoder.id_offset])))