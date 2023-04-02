import torch
import torch.nn as nn
import torch.nn.functional as F
from .sublayer import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


class ConvolutionLayer(nn.Module):
    def __init__(
            self,
            embedding_shape,
            nchannels=20,
            nfilters=1,
            kernel_size=5,
            drop_rate=0.3,
            pool_size=2,
            output_embedding_len=256
    ):
        super(ConvolutionLayer, self).__init__()
        # setting stride to 1 for now for dimension calculation simplicity
        self.conv1 = nn.Conv2d(1, nchannels, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.ModuleList([
            nn.Conv2d(nchannels, nchannels, kernel_size=kernel_size, stride=1)
            for n in range(nfilters)
        ])
        self.conv2_drop = nn.ModuleList([nn.Dropout2d(p=drop_rate) for n in range(nfilters)])

        self.pool_size = pool_size

        input_dim = ConvolutionLayer.calc_shape(
            input_x=embedding_shape[0],
            input_y=embedding_shape[1],
            kernel_size=kernel_size,
            pool_size=pool_size,
            nfilters=nfilters,
            nchannels=nchannels
        )

        self.fc1 = nn.Linear(input_dim, output_embedding_len*2)
        self.fc2 = nn.Linear(output_embedding_len*2, output_embedding_len)

    @staticmethod
    def calc_dim(
            input_dim,
            kernel_size,
            pool_size,
            nfilters
    ):
        res = input_dim
        for i in range(nfilters + 1):
            res -= kernel_size - 1
            res = res // pool_size

        return res

    @staticmethod
    def calc_shape(
            input_x,
            input_y,
            kernel_size,
            pool_size,
            nfilters,
            nchannels
    ):

        dim_x = ConvolutionLayer.calc_dim(input_x, kernel_size, pool_size, nfilters)
        dim_y = ConvolutionLayer.calc_dim(input_y, kernel_size, pool_size, nfilters)

        res = nchannels * dim_x * dim_y
        return res

    def forward(self, x):

        # x.shape == [10, 26, 51, 51]

        nsamples = x.shape[0]
        nframes = x.shape[1]
        x = x.reshape(nsamples * nframes, 1, x.shape[2], x.shape[3])
        # x.shape == [260, 1, 51, 51]

        x = F.relu(F.max_pool2d(self.conv1(x), self.pool_size))
        # x.shape == [260, 20, 23, 23]

        for i in range(len(self.conv2)):
            x = F.relu(F.max_pool2d(self.conv2_drop[i](self.conv2[i](x)), self.pool_size))
        # x.shape == [260, 20, 9, 9]

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        x = x.reshape(nsamples, nframes, x.shape[-1])

        return x