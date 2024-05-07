#Code from:
#https://github.com/pytorch/examples/blob/main/word_language_model/model.py
#https://github.com/pytorch/examples/blob/main/vision_transformer/main.py

import torch
from torch import nn
from itertools import chain

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=500).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class ConvTransformerCustom(nn.Transformer):
    def __init__(self, n_in, n_out, n_heads=8, n_layers=4, n_in_tf=256, n_hidden=512, convs=2, 
                 dilation=(3,3), kernel_size=(3,3), dropout=0.1, **kwargs): 
        '''
            n_in is a tuple indicating channels for (t, xy, txy)
        '''
        #Set variables
        self.model_type = 'Transformer'
        self.src_mask = None
        self.n_in = n_in
        
        #Initialize transformer
        #d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
        #activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False,
        #norm_first=False, bias=True, device=None, dtype=None
        super().__init__(d_model=n_in_tf, nhead=n_heads, dim_feedforward=n_hidden, num_encoder_layers=n_layers)
        self.pos_encoder = PositionalEncoding(n_in_tf, dropout)
        
        #Initialize conv embedder
        assert len(kernel_size) == 2
        assert len(dilation) == 2
        assert len(n_in) == 3
        self.conv_channels_hidden= n_in_tf
        seq_list=[]
        for i in range(convs):
            seq_list.append( nn.Conv2d(
                  in_channels=sum(n_in) if i==0 else self.conv_channels_hidden, 
                  out_channels=self.conv_channels_hidden, kernel_size=kernel_size, 
                  padding=kernel_size[-1]//2 if dilation[-1]==1 or kernel_size[-1]==1 else dilation, dilation=dilation) )
            #if i < convs_per_layer -1: 
            seq_list.append(nn.SiLU())
        self.conv = nn.Sequential(*seq_list)

        #Initialize decoder
        self.decoder = nn.Linear(n_in_tf, n_out)

        #Initialize weights
        self.init_weights()
        
    def get_xy_params(self):
        return list(chain(list(m.parameters()) for m in [self.conv]))[0]
    
    def get_t_params(self):
        return list(chain(list(m.parameters()) for m in [self.encoder, self.pos_encoder, self.decoder]))[0]

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self, initrange=0.1):
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, t_in, xy_in, txy_in, train_mask, has_mask=True):
        #Process inputs: [b c t; b c h w ; b c t h w] -> [bt c h w]
        b, c_txy, t, h, w= txy_in.shape
        txy_joint= _remove_time(t_in, xy_in, txy_in, self.n_in)
        
        #Pass through conv layers
        conv_out= self.conv(txy_joint) #bt self.conv_channels_hidden h w
        conv_out= conv_out.reshape(b, t, self.conv_channels_hidden, h, w) #b t self.conv_channels_hidden h w
        
        #Build tensor of shape (b t c) by moving all other things to b
        #Transformer and pos_encoder are batch_first=False, so: bhw, t, c -> t bhw c
        #b t c h w -> b h w t c -> bhw t c -> t bhw c
        src= conv_out.permute(0,3,4,1,2).reshape(b*h*w, t, self.conv_channels_hidden).permute(1,0,2)
        
        #Pass through transformer
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        
        #Reformat output and return
        _, _, c_out= output.shape
        # t bhw c -> bhw t c -> b h w t c  -> b c t h w
        output= output.permute(1,0,2).reshape(b, h, w, t, c_out).permute(0,4,3,1,2)
        return output
    
def _remove_time(t_in, xy_in, txy_in, expected_channels):
    'Processor: [b c t; b c h w ; b c t h w] -> [bt c h w]'
    #Process inputs : b c t; b c h w ; b c t h w
    b, c_txy, t, h, w= txy_in.shape
    _, c_xy, _, _= xy_in.shape
    _, c_t, _= t_in.shape
    assert all([c1==c2 for c1, c2 in zip([c_t, c_xy, c_txy], expected_channels)]), \
        print(f'{[c_t, c_tx, c_txy]=} != {expected_channels=}')
    #b c t h w -> b t c h w -> bt c h w
    #b c h w -> b t c h w -> bt c h w
    #b c t -> b t c -> b t c h w -> bt c h w
    txy_joint= torch.concatenate([txy_in.permute(0,2,1,3,4).reshape(b*t, c_txy, h, w), 
                                  xy_in[:,None].repeat(1,t,1,1,1).reshape(b*t, c_xy, h, w),
                                  t_in.permute(0,2,1)[:,:,:,None,None].repeat(1,1,1,h,w).reshape(b*t, c_t, h, w)
                                 ], axis=1)
    return txy_joint
    
#From ViT
class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        batch_size, channels, height, width = input_data.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Input height ({height}) and width ({width}) must be divisible by patch size ({self.patch_size})"

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
            unfold(3, self.patch_size, self.patch_size). \
            permute(0, 2, 3, 1, 4, 5). \
            contiguous(). \
            view(batch_size, num_patches, -1)

        # Expected shape of a patch on default settings is (4, 196, 768)
        return patches

class InputEmbedding(nn.Module):
    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.LinearProjection = nn.Linear(self.input_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size).to(self.device))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size).to(self.device))

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(input_data)

        linear_projection = self.LinearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = self.pos_embedding[:, :n + 1, :]
        linear_projection += pos_embed

        return linear_projection

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.norm = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, emb_patches, mask=None):
        first_norm = self.norm(emb_patches)
        attention_out = self.attention(first_norm, first_norm, first_norm, attn_mask=mask)[0]
        first_added = attention_out + emb_patches
        second_norm = self.norm(first_added)
        mlp_out = self.enc_MLP(second_norm)
        output = mlp_out + first_added

        return output

#TODO: xytViT implementation is not finished yet. Stay tuned or send a PR!
class xytViT(nn.Module):
    def __init__(self, n_in, n_out, patch_size=16, num_encoders=4, latent_size=1024, 
                 dropout=0.1, num_heads=4):
        super().__init__()

        self.n_in = n_in
        # self.latent_size = latent_size
        # self.num_classes = n_out
        tf_input_size = patch_size**2 * sum(self.n_in) #Input size to the TF
        self.src_mask= None

        #No imput embedding over xy for now, every tile is independent
        #self.embedding = InputEmbedding(**kwargs)
        # Patchifying the Image
        self.patchify = PatchExtractor(patch_size=patch_size)
        self.LinearProjection = nn.Linear(tf_input_size, latent_size)
        self.pos_encoder = PositionalEncoding(latent_size, dropout)
        
        # Encoder Stack
        self.encoders = nn.ModuleList([EncoderBlock(latent_size=latent_size, 
                                dropout=dropout, num_heads=num_heads) for _ in range(self.num_encoders)])
        
        #Linear or non-linear decoder?
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, latent_size),
            nn.SiLU(),
            nn.Linear(latent_size, n_out),
        )
        
        #self.decoder = nn.Linear(latent_size, n_out)
    
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, t_in, xy_in, txy_in):
        #enc_output = self.embedding(test_input)
        b, c_txy, t, h, w= txy_in.shape
        txy_join= _remove_time(t_in, xy_in, txy_in, self.n_in) #out: bt c h w
        
        patches = patchify(input_data) #out: bt n_patches c(hp)(wp)
        _, np, chpwp= patches.shape
        patches2= patches.reshape(t*np, chpwp) #out: btnp chpwp
        
        enc_output = self.LinearProjection(patches2) #out: btnp l [l=latent_size]
        
        #btnp l -> b t np l -> t b np l -> t bnp l
        _, l= enc_output.shape
        src= enc_output.reshape(b, t, np, l).permute(1,0,2,3).reshape(t, b*np, l)
        
        #Pass through transformer
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        for enc_layer in self.encoders:
            enc_output = enc_layer(enc_output, mask=self.src_mask)

        output = self.decoder(output)
        
        #TODO: Recompose images from patches: t bnp c_out ->?
        _, _, c_out= output.shape
        
        #TODO: Return with proper dimensions
        # t bnp c -> bhw t c -> b h w t c  -> b c t h w
        output= output.permute(1,0,2).reshape(b, h, w, t, c_out).permute(0,4,3,1,2)
        
        return output