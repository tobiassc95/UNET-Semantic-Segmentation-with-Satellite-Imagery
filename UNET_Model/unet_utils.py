import torch
import torch.nn as nn
import torch.nn.functional as tf

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False), # bias=False because it'd be canceled by the batch normalization.
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False), # bias=False because it'd be canceled by the batch normalization.
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False) # Two convolution layers.
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2) # Up convolution (transpose convolution).
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2) # It may be better than the nn.ConvTranspose2d (try it).

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = tf.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))

