import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm as bar
import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper

# The code to build the model is modified from:
# https://github.com/krasserm/perceiver-io


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


def cross_attention_layer(num_q_channels: int,
                          num_kv_channels: int,
                          num_heads: int,
                          dropout: float,
                          activation_checkpoint: bool = False):
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(num_channels: int,
                         num_heads: int,
                         dropout: float,
                         activation_checkpoint: bool = False):
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(num_layers: int,
                         num_channels: int,
                         num_heads: int,
                         dropout: float,
                         activation_checkpoint: bool = False
                         ):
    layers = [self_attention_layer(
        num_channels,
        num_heads,
        dropout,
        activation_checkpoint) for _ in range(num_layers)]

    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):

    def __init__(self, num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):

    def __init__(self,
                 num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels,
            num_kv_channels=num_channels,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class Encoder(nn.Module):

    def __init__(
            self,
            input_ch,
            preproc_ch,
            num_latents: int,
            num_latent_channels: int,
            num_layers: int = 3,
            num_cross_attention_heads: int = 4,
            num_self_attention_heads: int = 4,
            num_self_attention_layers_per_block: int = 6,
            dropout: float = 0.0,
            activation_checkpoint: bool = False,
    ):

        super().__init__()

        self.num_layers = num_layers
        if preproc_ch:
            self.preproc = nn.Linear(input_ch, preproc_ch)
        else:
            self.preproc = None
            preproc_ch = input_ch

        def create_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
            )

        self.layer_1 = create_layer()

        if num_layers > 1:
            self.layer_n = create_layer()
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape  # x, (64,8,129)；b=64

        if self.preproc:
            x = self.preproc(x)  # x, (64,8,64)

        # repeat initial latent vector along batch dimension  # self.latent, (num_latents, num_latent_channels)
        x_latent = repeat(self.latent, "... -> b ...", b=b)  # (64,4,16)
        x_latent = self.layer_1(x_latent, x, pad_mask)  # Multi-head cross-attention, x_latent (latent query), (64,4,16), regardless of obs num
        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent  # (64,4,16)


class Decoder(nn.Module):
    def __init__(
            self,
            ff_channels: int,
            preproc_ch,
            num_latent_channels: int,
            latent_size,
            num_output_channels,
            num_cross_attention_heads: int = 4,
            dropout: float = 0.0,
            activation_checkpoint: bool = False,
    ):

        super().__init__()
        q_chan = ff_channels + num_latent_channels
        if preproc_ch:
            q_in = preproc_ch
        else:
            q_in = q_chan

        self.postproc = nn.Linear(q_in, num_output_channels)

        if preproc_ch:
            self.preproc = nn.Linear(q_chan, preproc_ch)
        else:
            self.preproc = None

        self.cross_attention = cross_attention_layer(
            num_q_channels=q_in,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            activation_checkpoint=activation_checkpoint,
        )

        self.output = nn.Parameter(torch.empty(latent_size, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, coords):  # coords (64, 2048, 128)
        b, *_ = x.shape

        output = repeat(self.output, "... -> b ...", b=b)  # (64,1,16)
        output = torch.repeat_interleave(output, coords.shape[1], axis=1)  # (64,2048,16)

        output = torch.cat([coords, output], axis=-1)  # (64, 2048, 144)

        if self.preproc:
            output = self.preproc(output)  # (64, 2048, 144)

        output = self.cross_attention(output, x)  # (64,2048, 144)
        return self.postproc(output)  # (64,2048, 1)


class Senseiver(pl.LightningModule):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2
        out = self.hparams.im_ch + pos_encoder_ch,
        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch,
            preproc_ch = self.hparams.enc_preproc_ch,
            num_latents = self.hparams.num_latents,
            num_latent_channels = self.hparams.enc_num_latent_channels,
            num_layers = self.hparams.num_layers,
            num_cross_attention_heads = self.hparams.num_cross_attention_heads,
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
            dropout = self.hparams.dropout,
        )
        
       
        self.decoder_1 = Decoder(
            ff_channels = pos_encoder_ch,
            preproc_ch = self.hparams.dec_preproc_ch,  # latent bottleneck
            num_latent_channels = self.hparams.dec_num_latent_channels,  # hyperparam
            latent_size = self.hparams.latent_size,  # collapse from n_sensors to 1
            num_output_channels = self.hparams.im_ch,
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads,
            dropout = self.hparams.dropout,
        )

        
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')
        
        
        
    def forward(self, sensor_values, query_coords):
        out = self.encoder(sensor_values)  # (batch, num_latents, enc_num_latent_channels)
        return self.decoder_1(out, query_coords)

    def training_step(self, batch, batch_idx):
        sensor_values, coords, field_values = batch
        # forward
        pred_values = self(sensor_values, coords)
        
        # loss
        loss = F.mse_loss(pred_values, field_values, reduction='sum')
        
        self.log("train_loss", loss/field_values.numel(), 
                 on_step=True, on_epoch=True,prog_bar=True, logger=True,
                 batch_size=1)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def test(self, dataloader, num_pix=1024, split_time=0):
        
        #dataloader.dataset.Data_total = dataloader.dataset.Data_total[2382:2383,]
        im_num, *im_size, im_ch = dataloader.dataset.data.shape
        
        im_pix    = np.prod(im_size)
        pixels    = np.arange( 0, im_pix, num_pix )
        output_im = torch.zeros(im_num, im_pix, im_ch)
        
        # split the time steps to avoid OOM errors
        if im_num==1:
            times = [0,1]
        else:
            times = np.linspace(0, im_num, split_time, dtype=int)
        
        # Data_total
        im = dataloader.dataset.data
        sensors = dataloader.dataset.sensors
        pos_encodings = dataloader.dataset.pos_encodings

        t = 0
        for t_start in bar( times[1:] ):
            dt = t_start-t
            for pix in bar( pixels ):
                
                coords = pos_encodings[pix:pix+num_pix,][None,]
                coords = coords.repeat_interleave(dt, axis=0)
                
                sensor_values = im.flatten(start_dim=1, end_dim=-2)[t:t_start,sensors]
                
                sensor_positions = pos_encodings[sensors,][None,]
                sensor_positions = sensor_positions.repeat_interleave(sensor_values.shape[0], axis=0)
                
                sensor_values = torch.cat([sensor_values,sensor_positions], axis=-1)
           
                out = self(sensor_values, coords)
            
                output_im[t:t_start,pix:pix+num_pix] = out
            t += dt
            
        output_im = output_im.reshape(-1, *im_size, im_ch)
        output_im[dataloader.dataset.data==0]=0
        
        return output_im
    

    
    
    def histogram(self, path):
            import pickle
            
            results = dict()
            with torch.no_grad():
                
                self.im_num = 500
                self.im = self.im[:self.im_num]
                
                
                pixels = np.arange( 0, self.im_pix)
                coords = self.pos_encoder.position_encoding[:,][None,]
                
                for seed in bar( [123,1234,12345,9876,98765,666,777,11111] ):
                    results[str(seed)] = {}
                    for num_of_sensors in [25,50,100,150,200,250,500,750]:
                        
                        torch.manual_seed(seed)
                        rnd_sensor_ind = torch.randperm( 6144 )[:num_of_sensors]
                    
                        pred = torch.zeros(self.im_num, self.im_pix, 1) 
                        
                        sensor_positions = self.pos_encoder.position_encoding[self.sensors[rnd_sensor_ind],][None,]
                        for pix in range(self.im_num):
                            
                            sensor_values = self.im.flatten(start_dim=1, end_dim=2)[pix:pix+1,self.sensors[rnd_sensor_ind]]
                            sensor_values = torch.cat([sensor_values,sensor_positions], axis=-1)
                            pred[pix,:] = self(sensor_values, coords)
                            
                        pred = pred.reshape(-1, *self.im_dims, self.im_ch)
                        e = (self.im.cpu()-pred).norm(p=2, dim=(1,2))/(self.im.cpu()).norm(p=2, dim=(1,2))
                        results[str(seed)][str(num_of_sensors)] = e.mean()
                    print(results)
            with open(f'{path}/errors.pk', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
    
    
    
    
    
    
    
    
    
