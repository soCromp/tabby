from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

#generation
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from random import shuffle
from copy import deepcopy

SUPERCLASS_FOR_HEADLESS_LM = {GPT2LMHeadModel:GPT2Model, LlamaForCausalLM:LlamaModel, GPTNeoXForCausalLM:GPTNeoXModel}

class Integer: # so I can have pointers to an int
    value = 0

class MultiLayer(nn.Module): 
    """Used for MOE MLPs and multiheads"""
    def __init__(self, col):
        super(MultiLayer, self).__init__()
        self.col = col
        
    def from_other(mlp, col, num_experts):
        multilayer = MultiLayer(col)
        layerlist = []
        for i in range(num_experts):
            layerlist.append(deepcopy(mlp))
        multilayer.layers = nn.ModuleList(layerlist)
        return multilayer
        
        
    def forward(self, hidden_states, **kwargs):
        # print('generate with mlp', self.col.value)
        return self.layers[self.col.value](hidden_states, **kwargs)
        
    
class MOECausalLMOutputWithPast(CausalLMOutputWithPast):
    collosses: []
    

class MHTabbyGPT2Config(GPT2Config):
    def __init__(self, num_experts=1, multihead=True, pad=None, eoc=None, **kwargs):
        super().__init__(**kwargs)  # Inherit standard GPT-2 config properties
        self.num_experts = num_experts
        self.multihead = multihead
        self.pad = pad
        self.eoc = eoc
    
    
class MHTabbyGPT2(GPT2LMHeadModel):
    # demo only works with gpt2 and dgpt2
    config_class = MHTabbyGPT2Config
    
    def __init__(self, config):
        super().__init__(config)  # Initialize GPT-2's standard attributes
        # Load extra MOE attributes from the config
        self.num_experts = config.num_experts
        self.multihead = config.multihead
        self.PAD = config.pad
        self.EOC = config.eoc
        self.col = Integer()
        self.col.value = 0
        
        del self.lm_head
        # Define expert layers
        head_like = nn.Linear(config.hidden_size, self.EOC+1, bias=False)
        self.lm_head = MultiLayer.from_other(
            head_like, self.col, self.num_experts
        )
        # self.lm_head = nn.ModuleList([
        #     nn.Linear(config.hidden_size, self.EOC+1, bias=False) for _ in range(self.num_experts)
        # ])
        
        
    def from_other(model, pad, eoc, num_experts=1, moe=False, multihead=False):
        # makes a tabby model out of a non-tabby model
        modeltype = type(model)
        moemodel = model
        moemodel.__class__ = MHTabbyGPT2
        moemodel.col = Integer()
        moemodel.num_experts = num_experts
        moemodel.PAD = pad 
        moemodel.EOC = eoc
        
        if moe:
            for i in range(len(moemodel.transformer.h)):
                moemodel.transformer.h[i].mlp = MultiLayer.from_other(
                    moemodel.transformer.h[i].mlp, moemodel.col, moemodel.num_experts)
        
        if multihead:
            moemodel.lm_head = MultiLayer.from_other(
                moemodel.lm_head, moemodel.col, moemodel.num_experts
            )
            
        return moemodel
    
    
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        """ Load the base GPT2 model and modify it for MOE """
        config = MHTabbyGPT2Config.from_pretrained(model_name)
        model = super().from_pretrained(model_name, config=config, *args, **kwargs)

        # Ensure token embeddings are resized after loading
        model.resize_token_embeddings(config.vocab_size)
        return model
    
    
    def set_train_mode(self):
        self.forward = self.multicol_forward
        
        
    def set_generation_mode(self, token_heads=None, column_names_tokens=None):
        self.forward = self.autocol_forward
        self.token_heads = token_heads
        self.column_names_tokens = column_names_tokens
        
    
    #generation forward
    def autocol_forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
        transformer, lm_head = self.children()
        
        prompt = deepcopy(input_ids) #bs x tokens
        mask = torch.ones_like(prompt)
        
        transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
        hidden_states = transformer_outputs[0]
        lm_logits = lm_head(hidden_states)
        
        
        return MOECausalLMOutputWithPast(
            loss = None,
            logits = lm_logits,
            past_key_values = transformer_outputs.past_key_values,
            hidden_states = transformer_outputs.hidden_states,
            attentions = transformer_outputs.attentions
        )
        
        
    #training forward
    def multicol_forward(self, input_ids = None, attention_mask = None, labels = None, cols_iterator=None, **kwargs):
        # input_ids, attention_mask, labels: batch x column x tokens

        transformer, lm_head = self.children()
        
        prompt = deepcopy(input_ids) #bs x tokens
        if labels is not None:
            prompt = torch.cat([prompt[prompt!=self.PAD].unsqueeze(0), 
                                labels[:,0,:][labels[:,0,:] != self.PAD].unsqueeze(0)], axis=1)
        
        if cols_iterator == None:
            cols_iterator = range(self.num_experts)
        elif len(cols_iterator.shape) == 2: # because huggingface trainer wraps cols_iterator into extra []
            cols_iterator = cols_iterator[0]
        
        collosses = []
        lossavg = None
        mask = torch.ones_like(prompt)
        for i in cols_iterator:
            self.col.value = i
            transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = lm_head(hidden_states)
            
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = prompt[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                collosses.append(loss)
                
            lossavg = sum(collosses) / len(collosses)
                
            # update prompt and mask
            if i < self.num_experts-1:
                if labels is not None: #in training mode, where labels are known
                    prompt = torch.cat([prompt, labels[:,i+1,:][labels[:,i+1,:] != self.PAD].unsqueeze(0)], axis=1)
                else: # in inference mode, where a column's prompt is the preds from the prior columns
                    return NotImplementedError
                    
                mask = torch.ones_like(prompt)
                
        return MOECausalLMOutputWithPast(
            loss = lossavg,
            logits = lm_logits,
            past_key_values = transformer_outputs.past_key_values,
            hidden_states = transformer_outputs.hidden_states,
            attentions = transformer_outputs.attentions
        )


    def _sample(
        self,
        input_ids,
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id = None,
        eos_token_id = None,
        output_attentions = None,
        output_hidden_states = None,
        output_scores = None,
        output_logits = None,
        return_dict_in_generate = None,
        synced_gpus = False,
        streamer = None,
        **model_kwargs,
    ):
        def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper):
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            return next_token_scores
        def select_next_token(next_token_scores):
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            return next_tokens
        
        return self.generation_loop(input_ids, logits_processor, stopping_criteria, logits_warper, max_length, pad_token_id, eos_token_id, output_attentions, 
            output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
            get_next_token_scores, select_next_token, **model_kwargs)
    
    
    def _greedy_search(
        self,
        input_ids,
        logits_processor = None,
        stopping_criteria = None,
        max_length = None,
        pad_token_id = None,
        eos_token_id = None,
        output_attentions = None,
        output_hidden_states = None,
        output_scores = None,
        output_logits = None,
        return_dict_in_generate = None,
        synced_gpus = False,
        streamer = None,
        **model_kwargs,
    ):
        def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper=None):
            next_token_scores = logits_processor(input_ids, next_token_logits)
            return next_token_scores
        def select_next_token(next_token_scores):
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            return next_tokens
            
        return self.generation_loop(input_ids, logits_processor, stopping_criteria, None, max_length, pad_token_id, eos_token_id, output_attentions, 
            output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
            get_next_token_scores, select_next_token, **model_kwargs)


    def generation_loop(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        logits_warper,
        max_length,
        pad_token_id,
        eos_token_id,
        output_attentions,
        output_hidden_states,
        output_scores,
        output_logits,
        return_dict_in_generate,
        synced_gpus,
        streamer,
        get_next_token_scores, 
        select_next_token,
        **model_kwargs,
    ):
        # init values
        expert = 0 # used to index into the list saying the order of cols/experts
        if self.token_heads is None: 
            token_heads = list(range(len(self.column_names_tokens)-1))
            shuffle(token_heads)
            token_heads = [len(self.column_names_tokens)-1] + token_heads
            # print(token_heads)
        else:
            token_heads = self.token_heads
            
        self.col.value = token_heads[expert]
        # print(self.col.value)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.PAD
        eoc_token_id = self.EOC # eos_token_id if eos_token_id is not None else self.EOC
        if isinstance(eoc_token_id, int):
            eoc_token_id = [eoc_token_id]
        eoc_token_id_tensor = torch.tensor(eoc_token_id).to(input_ids.device) #if eoc_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        
        insert_column_name = False
        column_names_tokens = deepcopy(self.column_names_tokens) # since we're popping and don't want to change original
        
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # choose next tokens (sample/argmax)
            next_tokens = select_next_token(next_token_scores)
            if input_ids[..., -1].item() == self.EOC and expert < self.num_experts-1:
                expert += 1
                self.col.value = token_heads[expert]
                next_tokens = torch.full_like(next_tokens, column_names_tokens[self.col.value].pop(0))
                if len(column_names_tokens[self.col.value]) > 0: # more tokens to keep inserting
                    insert_column_name = True 
            elif insert_column_name:
                next_tokens = torch.full_like(next_tokens, column_names_tokens[self.col.value].pop(0))
                if len(column_names_tokens[self.col.value]) == 0: # inserted this whole column name
                    insert_column_name = False
            elif input_ids[..., -1].item() == self.EOC and expert == self.num_experts-1: # this line is done
                break

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )


        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
    
def MOEModelForCausalLM(model, **kwargs):
    superclassForCausalLM = type(model)
    
    class MOEModelForCausalLM(superclassForCausalLM):
        def __init__(self):
            super(MOEModelForCausalLM, self).__init__()
            self.col = Integer()
            self.col.value = 0
            self.num_experts=1
            self.PAD = -1 # not initialized
            self.EOC = -1 # not initialized
            
            
        def from_other(model, pad, eoc, num_experts=1, moe=False, multihead=False, ma=False):
            # https://stackoverflow.com/questions/597199/converting-an-object-into-a-subclass-in-python
            # moemodel = deepcopy(model)
            modeltype = type(model)
            moemodel = model
            moemodel.__class__ = MOEModelForCausalLM
            moemodel.col = Integer()
            moemodel.num_experts = num_experts
            moemodel.PAD = pad 
            moemodel.EOC = eoc
            
            if moe or ma:
                if modeltype == GPT2LMHeadModel:
                    for i in range(len(moemodel.transformer.h)):
                        if moe:
                            moemodel.transformer.h[i].mlp = MultiLayer.from_other(
                                moemodel.transformer.h[i].mlp, moemodel.col, moemodel.num_experts)
                        if ma:
                            moemodel.transformer.h[i].attn = MultiLayer.from_other(
                                moemodel.transformer.h[i].attn, moemodel.col, moemodel.num_experts)
                elif modeltype == LlamaForCausalLM:
                    # moemodel = deepcopy(model)
                    print('deep copied model')
                    moemodel.__class__ = MOEModelForCausalLM
                    for i in range(len(moemodel.model.layers)):
                        if moe:
                            moemodel.model.layers[i].mlp = MultiLayer.from_other(
                                moemodel.model.layers[i].mlp, moemodel.col, moemodel.num_experts)
                        if ma:
                            moemodel.model.layers[i].self_attn = MultiLayer.from_other(
                                moemodel.model.layers[i].self_attn, moemodel.col, moemodel.num_experts)
                    print('added MOE MLPs')
                elif modeltype == GPTNeoXForCausalLM:
                    moemodel.__class__ = MOEModelForCausalLM  
                    for i in range(len(moemodel.gpt_neox.layers)):
                        if moe:
                            moemodel.gpt_neox.layers[i].mlp = MultiLayer.from_other(
                                moemodel.gpt_neox.layers[i].mlp, moemodel.col, moemodel.num_experts)
                        if ma:
                            moemodel.gpt_neox.layers[i].attention = MultiLayer.from_other(
                                moemodel.gpt_neox.layers[i].attention, moemodel.col, moemodel.num_experts)
                else:
                    raise NotImplementedError(f'Type {type(model)} not supported')
            
            if multihead:
                if modeltype == GPT2LMHeadModel or modeltype == LlamaForCausalLM:
                    moemodel.lm_head = MultiLayer.from_other(
                        moemodel.lm_head, moemodel.col, moemodel.num_experts
                    )
                elif modeltype == GPTNeoXForCausalLM:
                    moemodel.embed_out = MultiLayer.from_other(
                        moemodel.embed_out, moemodel.col, moemodel.num_experts
                    )
                else:
                    raise NotImplementedError(f'Type {type(model)} not supported')
            return moemodel
        
        
        def set_train_mode(self):
            self.forward = self.multicol_forward
            
            
        def set_generation_mode(self, token_heads=None, column_names_tokens=None):
            self.forward = self.autocol_forward
            self.token_heads = token_heads
            self.column_names_tokens = column_names_tokens
            
        
        #generation forward
        def autocol_forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            transformer, lm_head = self.children()
            
            prompt = deepcopy(input_ids) #bs x tokens
            mask = torch.ones_like(prompt)
            
            transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = lm_head(hidden_states)
            
            
            return MOECausalLMOutputWithPast(
                loss = None,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
            
            
        #training forward
        def multicol_forward(self, input_ids = None, attention_mask = None, labels = None, cols_iterator=None, **kwargs):
            # input_ids, attention_mask, labels: batch x column x tokens

            transformer, lm_head = self.children()
            
            prompt = deepcopy(input_ids) #bs x tokens
            if labels is not None:
                prompt = torch.cat([prompt[prompt!=self.PAD].unsqueeze(0), 
                                    labels[:,0,:][labels[:,0,:] != self.PAD].unsqueeze(0)], axis=1)
            
            if cols_iterator == None:
                cols_iterator = range(self.num_experts)
            elif len(cols_iterator.shape) == 2: # because huggingface trainer wraps cols_iterator into extra []
                cols_iterator = cols_iterator[0]
            
            collosses = []
            lossavg = None
            mask = torch.ones_like(prompt)
            for i in cols_iterator:
                self.col.value = i
                transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
                hidden_states = transformer_outputs[0]
                lm_logits = lm_head(hidden_states)
                
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = prompt[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    collosses.append(loss)
                    
                lossavg = sum(collosses) / len(collosses)
                    
                # update prompt and mask
                if i < self.num_experts-1:
                    if labels is not None: #in training mode, where labels are known
                        prompt = torch.cat([prompt, labels[:,i+1,:][labels[:,i+1,:] != self.PAD].unsqueeze(0)], axis=1)
                    else: # in inference mode, where a column's prompt is the preds from the prior columns
                        return NotImplementedError
                        
                    mask = torch.ones_like(prompt)
                    
            return MOECausalLMOutputWithPast(
                loss = lossavg,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
    
    
        def _sample(
            self,
            input_ids,
            logits_processor = None,
            stopping_criteria = None,
            logits_warper = None,
            max_length = None,
            pad_token_id = None,
            eos_token_id = None,
            output_attentions = None,
            output_hidden_states = None,
            output_scores = None,
            output_logits = None,
            return_dict_in_generate = None,
            synced_gpus = False,
            streamer = None,
            **model_kwargs,
        ):
            def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper):
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                return next_token_scores
            def select_next_token(next_token_scores):
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                return next_tokens
            
            return self.generation_loop(input_ids, logits_processor, stopping_criteria, logits_warper, max_length, pad_token_id, eos_token_id, output_attentions, 
                output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
                get_next_token_scores, select_next_token, **model_kwargs)
        
        
        def _greedy_search(
            self,
            input_ids,
            logits_processor = None,
            stopping_criteria = None,
            max_length = None,
            pad_token_id = None,
            eos_token_id = None,
            output_attentions = None,
            output_hidden_states = None,
            output_scores = None,
            output_logits = None,
            return_dict_in_generate = None,
            synced_gpus = False,
            streamer = None,
            **model_kwargs,
        ):
            def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper=None):
                next_token_scores = logits_processor(input_ids, next_token_logits)
                return next_token_scores
            def select_next_token(next_token_scores):
                next_tokens = torch.argmax(next_token_scores, dim=-1)
                return next_tokens
                
            return self.generation_loop(input_ids, logits_processor, stopping_criteria, None, max_length, pad_token_id, eos_token_id, output_attentions, 
                output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
                get_next_token_scores, select_next_token, **model_kwargs)
    
    
        def generation_loop(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            logits_warper,
            max_length,
            pad_token_id,
            eos_token_id,
            output_attentions,
            output_hidden_states,
            output_scores,
            output_logits,
            return_dict_in_generate,
            synced_gpus,
            streamer,
            get_next_token_scores, 
            select_next_token,
            **model_kwargs,
        ):
            # init values
            expert = 0 # used to index into the list saying the order of cols/experts
            if self.token_heads is None: 
                token_heads = list(range(len(self.column_names_tokens)-1))
                shuffle(token_heads)
                token_heads = [len(self.column_names_tokens)-1] + token_heads
                # print(token_heads)
            else:
                token_heads = self.token_heads
                
            self.col.value = token_heads[expert]
            # print(self.col.value)
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            if max_length is not None:
                warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                    UserWarning,
                )
                stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
            logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

            pad_token_id = pad_token_id if pad_token_id is not None else self.PAD
            eoc_token_id = self.EOC # eos_token_id if eos_token_id is not None else self.EOC
            if isinstance(eoc_token_id, int):
                eoc_token_id = [eoc_token_id]
            eoc_token_id_tensor = torch.tensor(eoc_token_id).to(input_ids.device) #if eoc_token_id is not None else None
            output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
            output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
            output_attentions = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            return_dict_in_generate = (
                return_dict_in_generate
                if return_dict_in_generate is not None
                else self.generation_config.return_dict_in_generate
            )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            if "inputs_embeds" in model_kwargs:
                cur_len = model_kwargs["inputs_embeds"].shape[1]
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
            
            insert_column_name = False
            column_names_tokens = deepcopy(self.column_names_tokens) # since we're popping and don't want to change original
            
            # print(self.PAD, self.EOC, pad_token_id, eoc_token_id)
            
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # print('self.col.value', self.col.value)
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # choose next tokens (sample/argmax)
                next_tokens = select_next_token(next_token_scores)
                # print(input_ids[..., -1].item())
                if input_ids[..., -1].item() == self.EOC and expert < self.num_experts-1:
                    expert += 1
                    self.col.value = token_heads[expert]
                    next_tokens = torch.full_like(next_tokens, column_names_tokens[self.col.value].pop(0))
                    if len(column_names_tokens[self.col.value]) > 0: # more tokens to keep inserting
                        insert_column_name = True 
                    # print('to expert', self.col.value, 'input ids shape', input_ids.shape)
                elif insert_column_name:
                    next_tokens = torch.full_like(next_tokens, column_names_tokens[self.col.value].pop(0))
                    if len(column_names_tokens[self.col.value]) == 0: # inserted this whole column name
                        insert_column_name = False
                elif input_ids[..., -1].item() == self.EOC and expert == self.num_experts-1: # this line is done
                    # print('done with line', 'input ids shape', input_ids.shape)
                    break

                # finished sentences should have their next token be a padding token
                # if eoc_token_id is not None:
                #     if pad_token_id is None:
                #         raise ValueError("If `eoc_token_id` is defined, make sure that `pad_token_id` is defined.")
                #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

                # if eoc_token was found in one sentence, set sentence to finished
                # if eoc_token_id_tensor is not None:
                #     unfinished_sequences = unfinished_sequences.mul(
                #         next_tokens.tile(eoc_token_id_tensor.shape[0], 1).ne(eoc_token_id_tensor.unsqueeze(1)).prod(dim=0)
                #     )

                # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                # this_peer_finished = unfinished_sequences.max() == 0

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids
     
     
    return MOEModelForCausalLM.from_other(model, **kwargs)
