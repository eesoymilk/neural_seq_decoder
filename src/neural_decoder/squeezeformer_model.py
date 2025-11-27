import torch
from torch import nn
import torchaudio
from .augmentations import GaussianSmoothing

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeFormerBlock(nn.Module):
    def __init__(self, encoder_dim, num_attention_heads, feed_forward_expansion_factor, conv_expansion_factor, dropout, attention_dropout, conv_dropout):
        super(SqueezeFormerBlock, self).__init__()
        
        # 1. Attention Module (Pre-Norm)
        self.layer_norm1 = nn.LayerNorm(encoder_dim)
        self.self_attn = nn.MultiheadAttention(encoder_dim, num_attention_heads, dropout=attention_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Convolution Module (Pre-Norm)
        # SqueezeFormer uses a specific conv structure: LayerNorm -> Pointwise -> Glu -> Depthwise -> Swish -> Pointwise -> Dropout
        self.layer_norm2 = nn.LayerNorm(encoder_dim)
        self.conv_module = nn.Sequential(
            # Pointwise
            nn.Conv1d(encoder_dim, encoder_dim * conv_expansion_factor, kernel_size=1),
            nn.GLU(dim=1),
            # Depthwise
            nn.Conv1d(encoder_dim * conv_expansion_factor // 2, encoder_dim * conv_expansion_factor // 2, kernel_size=31, groups=encoder_dim * conv_expansion_factor // 2, padding=15), # padding=(kernel-1)/2
            nn.SiLU(), # Swish
            nn.BatchNorm1d(encoder_dim * conv_expansion_factor // 2), # SqueezeFormer often uses BN inside Conv module
            # Pointwise
            nn.Conv1d(encoder_dim * conv_expansion_factor // 2, encoder_dim, kernel_size=1),
            nn.Dropout(conv_dropout)
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. Feed Forward Module (Pre-Norm)
        self.layer_norm3 = nn.LayerNorm(encoder_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * feed_forward_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * feed_forward_expansion_factor, encoder_dim),
            nn.Dropout(dropout)
        )
        self.dropout3 = nn.Dropout(dropout)
        
        # Layer Scale (Optional, skipping for simplicity unless needed)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        
        # Attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + self.dropout1(x)
        
        # Convolution
        residual = x
        x = self.layer_norm2(x)
        x = x.transpose(1, 2) # [batch, dim, seq_len]
        x = self.conv_module(x)
        x = x.transpose(1, 2) # [batch, seq_len, dim]
        x = residual + self.dropout2(x)
        
        # Feed Forward
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout3(x)
        
        return x

class SqueezeFormerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0.1,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False, # Unused
    ):
        super(SqueezeFormerDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        
        # Preprocessing components (same as GRUDecoder)
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Input layers (Day-specific adaptation)
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # Input projection
        self.input_projection = nn.Linear(neural_dim * kernelLen, hidden_dim)
        
        # SqueezeFormer Encoder
        self.layers = nn.ModuleList([
            SqueezeFormerBlock(
                encoder_dim=hidden_dim,
                num_attention_heads=4,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                dropout=dropout,
                attention_dropout=dropout,
                conv_dropout=dropout
            ) for _ in range(layer_dim)
        ])
        
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # neuralInput: [batch, time, features]
        neuralInput = torch.permute(neuralInput, (0, 2, 1)) # [batch, features, time]
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1)) # [batch, time, features]

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        # Project to hidden dim
        x = self.input_projection(stridedInputs) # [batch, seq_len, hidden_dim]
        
        # Apply SqueezeFormer blocks
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_layer_norm(x)
        
        # Output projection
        seq_out = self.fc_decoder_out(x)
        
        return seq_out
