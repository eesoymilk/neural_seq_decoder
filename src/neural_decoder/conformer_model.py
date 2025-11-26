import torch
from torch import nn
import torchaudio
from .augmentations import GaussianSmoothing

class ConformerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False, # Kept for compatibility, but Conformer is inherently bidirectional context-wise usually
    ):
        super(ConformerDecoder, self).__init__()

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

        # Conformer specific components
        # Input projection to hidden_dim
        # The input to the core model after unfolding will have dimension: neural_dim * kernelLen
        self.input_projection = nn.Linear(neural_dim * kernelLen, hidden_dim)
        
        # Conformer Encoder
        # We use torchaudio's Conformer implementation
        # Note: Conformer expects input (batch, time, input_dim) if batch_first=True? 
        # torchaudio.models.Conformer args:
        # input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size
        
        self.conformer = torchaudio.models.Conformer(
            input_dim=hidden_dim,
            num_heads=4, # Hyperparameter to tune
            ffn_dim=hidden_dim * 4,
            num_layers=layer_dim,
            depthwise_conv_kernel_size=31, # Default suggestion from paper
            dropout=dropout,
        )
        
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
        # transformedNeural: [batch, time, features]
        # We need to unfold to create windows
        # Unfold expects [batch, channel, time]? No, Unfold is 2D.
        # Let's look at GRUDecoder implementation:
        # stridedInputs = torch.permute(self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)), (0, 2, 1))
        # transformedNeural permuted: [batch, features, time]
        # unsqueeze: [batch, features, time, 1] - treating as image with height=time, width=1?
        # Unfold((kernelLen, 1)) slides over time.
        
        # Replicating GRU logic exactly for preprocessing
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        # stridedInputs shape: [batch, num_windows, features*kernelLen]
        
        # Project to hidden dim
        x = self.input_projection(stridedInputs) # [batch, seq_len, hidden_dim]
        
        # Conformer expects [batch, seq_len, input_dim] and lengths
        # We don't have lengths easily available here as argument, but we can assume padded zeros are handled or just pass full sequence
        # torchaudio Conformer forward: (input, lengths) -> (output, lengths)
        # lengths is optional? No, it seems required for mask generation in some versions, but let's check.
        # Actually torchaudio Conformer forward signature is (input, lengths)
        
        lengths = torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.int32)
        
        x, _ = self.conformer(x, lengths)
        
        # Output projection
        seq_out = self.fc_decoder_out(x)
        
        return seq_out
