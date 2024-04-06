import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channels=2):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, in_channels, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Bottleneck
        x2 = self.bottleneck(x1)

        # Decoder
        x3 = self.decoder(x2)

        return x3


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNetBlock, self).__init__()
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeneralUNet(nn.Module):
    def __init__(self, in_channels=2, kernel_size=3):
        super(GeneralUNet, self).__init__()

        # encoder
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(in_channels=in_channels, out_channels=32, kernel_size=kernel_size),
            UNetBlock(in_channels=32, out_channels=64, kernel_size=kernel_size),
            UNetBlock(in_channels=64, out_channels=128, kernel_size=kernel_size),
            UNetBlock(in_channels=128, out_channels=256, kernel_size=kernel_size)
        ])

        # max pooling layer that is used after each encoder block
        self.max_pool_1d = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        # bottleneck layer
        self.bottleneck_layer = UNetBlock(in_channels=256, out_channels=512, kernel_size=kernel_size)

        skip_conn_dims = [block.out_channels for block in self.encoder_blocks]

        # decoder
        self.decoder_blocks = nn.ModuleList(
            [UNetBlock(in_channels=512 + skip_conn_dims[-1], out_channels=256, kernel_size=kernel_size),
             UNetBlock(in_channels=256 + skip_conn_dims[-2], out_channels=128, kernel_size=kernel_size),
             UNetBlock(in_channels=128 + skip_conn_dims[-3], out_channels=64, kernel_size=kernel_size),
             UNetBlock(in_channels=64 + skip_conn_dims[-4], out_channels=32, kernel_size=kernel_size)]
        )

        # max unpooling layer that is used after each decoder block
        self.max_unpool_1d = nn.MaxUnpool1d(kernel_size=2, stride=2)

        # output layer
        self.out_conv = nn.Conv1d(in_channels=32, out_channels=in_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        # encoder pass
        encoder_outputs = []
        pooling_indices = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x, indices = self.max_pool_1d(x)
            encoder_outputs.append(x)
            pooling_indices.append(indices)

        x = self.bottleneck_layer(x)

        # decoder pass
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = torch.cat([x, encoder_outputs[-(i + 1)]], dim=1)
            x = decoder_block(x)
            x = self.max_unpool_1d(x, pooling_indices[-(i + 1)])

        return self.out_conv(x)
