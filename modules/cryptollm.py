import math
from typing import Optional
import torch
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from transformers import AutoModel, AutoTokenizer, AutoConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from utils.config_manager import ConfigManager

from modules.embedding import PatchEmbedding2, TokenEmbedding2
from modules.reprogramming import ReprogrammingLayer2
from modules.normalize import Normalize2
from modules.head import FlattenHead2

class CRYPTOLLM(BaseWindows):

    def __init__(self,
                 h,
                 input_size,
                 llm = None,
   
                 llm_num_hidden_layers = 32,
                 llm_output_attention: bool = True,
                 llm_output_hidden_states: bool = True,
                 prompt_prefix: Optional[str] = None,
                 loss = MAE(),
                 valid_loss = None,
                 learning_rate: float = 1e-4,
                 max_steps: int = 5,
                 val_check_steps: int = 100,
                 batch_size: int = 32,
                 valid_batch_size: Optional[int] = None,
                 windows_batch_size: int = 1024,
                 inference_windows_batch_size: int = 1024,
                 start_padding_enabled: bool = False,
                 step_size: int = 1,
                 num_lr_decays: int = 0,
                 early_stop_patience_steps: int = -1,
                 scaler_type: str = 'identity',
                 num_workers_loader: int = 0,
                 drop_last_loader: bool = False,
                 random_seed: int = 1,
                 optimizer = None,
                 optimizer_kwargs = None,
                 lr_scheduler = None,
                 lr_scheduler_kwargs = None,
                 step_callback = None,
                 **trainer_kwargs):
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config()
        llm_params_config = config_manager.get_llm_params_config()
        llm_params = llm_params_config.get('models', {})
        
        DEFAULT_MODEL = "openai-community/gpt2"
        if llm is None:
            llm = DEFAULT_MODEL 

        d_llm = llm_params[llm]['hidden_dimension']

        patch_len = model_config.get('patch_len', 16)
        stride = model_config.get('stride', 8)
        d_ff = model_config.get('d_ff', 128)
        top_k = model_config.get('top_k', 5)
        d_model = model_config.get('d_model', 32)
        n_heads = model_config.get('n_heads', 8)
        enc_in = model_config.get('enc_in', 7)
        dec_in = model_config.get('dec_in', 7)
        dropout = model_config.get('dropout', 0.1)
        learning_rate = model_config.get('learning_rate', 1e-4)
        max_steps = model_config.get('max_steps', 30)
        batch_size = model_config.get('batch_size', 24)
        windows_batch_size = model_config.get('windows_batch_size', 24)
        early_stop_patience_steps = model_config.get('early_stop_patience_steps', 10)
        val_check_steps = model_config.get('val_check_steps', 10)


        super(CRYPTOLLM, self).__init__(h=h,
                                      input_size=input_size,
                                      #hist_exog_list=hist_exog_list, # Removed
                                      #stat_exog_list=stat_exog_list, # Removed
                                      #futr_exog_list = futr_exog_list, # Removed
                                      loss=loss,
                                      valid_loss=valid_loss,
                                      max_steps=max_steps,
                                      learning_rate=learning_rate,
                                      num_lr_decays=num_lr_decays,
                                      early_stop_patience_steps=early_stop_patience_steps,
                                      val_check_steps=val_check_steps,
                                      batch_size=batch_size,
                                      valid_batch_size=valid_batch_size,
                                      windows_batch_size=windows_batch_size,
                                      inference_windows_batch_size=inference_windows_batch_size,
                                      start_padding_enabled=start_padding_enabled,
                                      step_size=step_size,
                                      scaler_type=scaler_type,
                                      drop_last_loader=drop_last_loader,
                                      random_seed=random_seed,
                                      optimizer=optimizer,
                                      optimizer_kwargs=optimizer_kwargs,
                                      lr_scheduler=lr_scheduler,
                                      lr_scheduler_kwargs=lr_scheduler_kwargs)
        
        # Store DataLoader configuration
        self.num_workers_loader = num_workers_loader
        self.trainer_kwargs = {
            'accelerator': 'auto',
            'devices': 1,
            'max_epochs': max_steps,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            **trainer_kwargs
        }

        self.hist_exog_list = [] # Initialize internally
        
    
        print("Initializing model architecture...")
        self.patch_len = patch_len 
        self.stride = stride 
        self.d_ff = d_ff
        self.top_k = top_k
        self.d_llm = d_llm
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.llm = llm

        model_name = self.llm
 
        try:
            self.llm_config = AutoConfig.from_pretrained(model_name)
            self.llm = AutoModel.from_pretrained(model_name, config=self.llm_config)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        except EnvironmentError as e:
            print(
                f"Failed to load {model_name}. Loading the default model ({DEFAULT_MODEL})..."
            )
            # raise error saying model failed to load
            raise RuntimeError(f"Failed to load {model_name} \n error: {e} ")

        self.llm_num_hidden_layers = llm_num_hidden_layers
        self.llm_output_attention = llm_output_attention
        self.llm_output_hidden_states = llm_output_hidden_states
        self.prompt_prefix = prompt_prefix
       
        if self.llm_tokenizer.eos_token:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.llm_tokenizer.add_special_tokens({'pad_token': pad_token})
            self.llm_tokenizer.pad_token = pad_token

        print("Freezing LLM parameters.")
        for param in self.llm.parameters():
            param.requires_grad = False

        print("Initializing Patch Embedding, Mapping Layer, and Reprogramming Layer...")
        self.patch_embedding = PatchEmbedding2(
            self.d_model, self.patch_len, self.stride, self.dropout)

        self.word_embeddings = self.llm.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1024
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer2(self.d_model, self.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((input_size - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        self.output_projection = FlattenHead2(self.enc_in, self.head_nf, self.h, head_dropout=self.dropout)
        self.normalize_layers = Normalize2(self.enc_in, affine=False)
        print("Model architecture initialized.")

        self.learning_rate = learning_rate # overwrite with config value if loaded from config
        self.max_steps = max_steps # overwrite with config value if loaded from config
        self.batch_size = batch_size # overwrite with config value if loaded from config
        self.windows_batch_size = windows_batch_size # overwrite with config value if loaded from config
        self.early_stop_patience_steps = early_stop_patience_steps # overwrite with config value if loaded from config
        self.val_check_steps = val_check_steps # overwrite with config value if loaded from config
        self.step_callback = step_callback

    def configure_trainer(self):
        """Configure PyTorch Lightning trainer with proper settings"""
        callbacks = []
        if self.early_stop_patience_steps > 0:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop_patience_steps,
                mode='min'
            )
            callbacks.append(early_stop_callback)

        return pl.Trainer(
            callbacks=callbacks,
            max_epochs=self.max_steps,
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            **self.trainer_kwargs
        )

    def _generate_prompt(self, x_enc):
        print("Starting _generate_prompt method...")
        B, T, N = x_enc.size()
        x_enc_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = self.calcute_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)

        print("Statistics computed for prompt generation.")
        prompt = []
        for b in range(x_enc_reshaped.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>{self.prompt_prefix}"
                f"Task description: forecast the next {str(self.h)} steps given the previous {str(self.input_size)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<||>"
            )
            print(f"Generated prompt for batch {b}: {prompt_}")
            prompt.append(prompt_)
        return prompt

    def forecast3(self, x_enc):
        print("Starting forecast3 method...")
        x_enc = self.normalize_layers(x_enc, 'norm')
        print("Normalization done.")
        B, T, N = x_enc.size()
        print(f"Input size after normalization: Batch={B}, Time={T}, Variables={N}")


        prompt = self._generate_prompt(x_enc)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)


        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        print("Tokenizing prompts...")
        prompt_tokenized = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        print("Prompt tokenization complete.")
        prompt_embeddings = self.llm.get_input_embeddings()(prompt_tokenized.to(x_enc.device))  # (batch, prompt_token, dim)
        print("Getting source embeddings...")
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        print(f"Patch embedding output shape: {enc_out.shape}, Number of variables: {n_vars}")
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        print("Reprogramming layer applied.")
        llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        print(f"Concatenated prompt embeddings and encoded output: {llm_enc_out.shape}")
        dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        print(f"LLM output shape after decoding: {dec_out.shape}")
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        print(f"Output projection shape: {dec_out.shape}")
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        #print('dec_out', dec_out)
        return dec_out

    def __repr__(self):
        return 'CRYPTOLLM'

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        #print('lags',lags)
        return lags

    def forward(self, windows_batch):
        insample_y = windows_batch['insample_y']

        x = insample_y.unsqueeze(-1)

        y_pred = self.forecast3(x)
        y_pred = y_pred[:, -self.h:, :]
        y_pred = self.loss.domain_map(y_pred)
        #print('y_pred',y_pred)
        return y_pred

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        loss = super().training_step(batch, batch_idx)
        
        if self.step_callback:
            self.step_callback(self.current_epoch, loss.item())
            
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        loss = super().validation_step(batch, batch_idx)
        
        if self.step_callback:
            self.step_callback(self.current_epoch, None, loss.item())
            
        return loss

    def configure_trainer(self):
        """Configure PyTorch Lightning trainer with proper settings"""
        callbacks = []
        if self.early_stop_patience_steps > 0:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop_patience_steps,
                mode='min'
            )
            callbacks.append(early_stop_callback)

        return pl.Trainer(
            callbacks=callbacks,
            max_epochs=self.max_steps,
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            **self.trainer_kwargs
        )

    def _generate_prompt(self, x_enc):
        print("Starting _generate_prompt method...")
        B, T, N = x_enc.size()
        x_enc_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = self.calcute_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)

        print("Statistics computed for prompt generation.")
        prompt = []
        for b in range(x_enc_reshaped.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>{self.prompt_prefix}"
                f"Task description: forecast the next {str(self.h)} steps given the previous {str(self.input_size)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<||>"
            )
            print(f"Generated prompt for batch {b}: {prompt_}")
            prompt.append(prompt_)
        return prompt

    def forecast3(self, x_enc):
        print("Starting forecast3 method...")
        x_enc = self.normalize_layers(x_enc, 'norm')
        print("Normalization done.")
        B, T, N = x_enc.size()
        print(f"Input size after normalization: Batch={B}, Time={T}, Variables={N}")


        prompt = self._generate_prompt(x_enc)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)


        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        print("Tokenizing prompts...")
        prompt_tokenized = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        print("Prompt tokenization complete.")
        prompt_embeddings = self.llm.get_input_embeddings()(prompt_tokenized.to(x_enc.device))  # (batch, prompt_token, dim)
        print("Getting source embeddings...")
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        print(f"Patch embedding output shape: {enc_out.shape}, Number of variables: {n_vars}")
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        print("Reprogramming layer applied.")
        llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        print(f"Concatenated prompt embeddings and encoded output: {llm_enc_out.shape}")
        dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        print(f"LLM output shape after decoding: {dec_out.shape}")
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        print(f"Output projection shape: {dec_out.shape}")
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        #print('dec_out', dec_out)
        return dec_out

    def __repr__(self):
        return 'CRYPTOLLM'

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        #print('lags',lags)
        return lags

    def forward(self, windows_batch):
        insample_y = windows_batch['insample_y']

        x = insample_y.unsqueeze(-1)

        y_pred = self.forecast3(x)
        y_pred = y_pred[:, -self.h:, :]
        y_pred = self.loss.domain_map(y_pred)
        #print('y_pred',y_pred)
        return y_pred
