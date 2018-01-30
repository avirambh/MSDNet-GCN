from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
import torch
import torch.nn as nn


class _DynamicInputDenseBlock(nn.Module):

    def __init__(self, conv_modules, debug):
        super(_DynamicInputDenseBlock, self).__init__()
        self.conv_modules = conv_modules
        self.debug = debug

    def forward(self, x):
        """
        Use the first element as raw input, and stream the rest of
        the inputs through the list of modules, then apply concatenation.
        expect x to be [identity, first input, second input, ..]
        and len(x) - len(self.conv_modules) = 1 for identity

        :param x: Input
        :return: Concatenation of the input with 1 or more module outputs
        """
        if self.debug:
            for i, t in enumerate(x):
                print("Current input size[{}]: {}".format(i,
                                                          t.size()))

        # Init output
        out = x[0]

        # Apply all given modules and return output
        for calc, m in enumerate(self.conv_modules):
            out = torch.cat([out, m(x[calc + 1])], 1)

            if self.debug:
                print("Working on input number: %s" % calc)
                print("Added: ", m(x[calc + 1]).size())
                print("Current out size {}".format(out.size()))

        return out


class MSDLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 in_scales, out_scales, orig_scales, args):
        """
        Creates a regular/transition MSDLayer. this layer uses DenseNet like concatenation on each scale,
        and performs spatial reduction between scales. if input and output scales are different, than this
        class creates a transition layer and the first layer (with the largest spatial size) is dropped.

        :param current_channels: number of input channels
        :param in_scales: number of input scales
        :param out_scales: number of output scales
        :param orig_scales: number of scales in the first layer of the MSDNet
        :param args: other arguments
        """
        super(MSDLayer, self).__init__()

        # Init vars
        self.current_channels = in_channels
        self.out_channels = out_channels
        self.in_scales = in_scales
        self.out_scales = out_scales
        self.orig_scales = orig_scales
        self.args = args
        self.bottleneck = args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor
        self.growth_factor = self.args.msd_growth_factor
        self.debug = self.args.debug

        # Define Conv2d/GCN params
        self.use_gcn = args.msd_all_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)

        # Calculate number of channels to drop and number of
        # all dropped channels
        self.to_drop = in_scales - out_scales
        self.dropped = orig_scales - out_scales # Use this as an offset
        self.subnets = self.get_subnets()

    def get_subnets(self):
        """
        Builds the different scales of the MSD network layer.

        :return: A list of scale modules
        """
        subnets = nn.ModuleList()

        # If this is a transition layer
        if self.to_drop:
            # Create a reduced feature map for the first scale
            # self.dropped > 0 since out_scales < in_scales < orig_scales
            in_channels1 = self.current_channels *\
                          self.growth_factor[self.dropped - 1]
            in_channels2 = self.current_channels *\
                           self.growth_factor[self.dropped]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped]
            bn_width1 = self.bottleneck_factor[self.dropped - 1]
            bn_width2 = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_down_densenet(in_channels1,
                                                    in_channels2,
                                                    out_channels,
                                                    self.bottleneck,
                                                    bn_width1,
                                                    bn_width2))
        else:
            # Create a normal first scale
            in_channels = self.current_channels *\
                          self.growth_factor[self.dropped]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped]
            bn_width = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_densenet(in_channels,
                                               out_channels,
                                               self.bottleneck,
                                               bn_width))


        # Build second+ scales
        for scale in range(1, self.out_scales):
            in_channels1 = self.current_channels *\
                          self.growth_factor[self.dropped + scale - 1]
            in_channels2 = self.current_channels *\
                           self.growth_factor[self.dropped + scale]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped + scale]
            bn_width1 = self.bottleneck_factor[self.dropped + scale - 1]
            bn_width2 = self.bottleneck_factor[self.dropped + scale]
            subnets.append(self.build_down_densenet(in_channels1,
                                                    in_channels2,
                                                    out_channels,
                                                    self.bottleneck,
                                                    bn_width1,
                                                    bn_width2))

        return subnets

    def build_down_densenet(self, in_channels1, in_channels2, out_channels,
                            bottleneck, bn_width1, bn_width2):
        """
        Builds a scale sub-network for scales 2 and up.

        :param in_channels1: number of same scale input channels
        :param in_channels2: number of upper scale input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width1: The first input width of the bottleneck factor
        :param bn_width2: The first input width of the bottleneck factor
        :return: A scale module
        """
        conv_module1 = self.convolve(in_channels1, int(out_channels/2), 'down',
                                    bottleneck, bn_width1)
        conv_module2 = self.convolve(in_channels2, int(out_channels/2), 'normal',
                                    bottleneck, bn_width2)
        conv_modules = [conv_module1, conv_module2]
        return _DynamicInputDenseBlock(nn.ModuleList(conv_modules),
                                       self.debug)

    def build_densenet(self, in_channels, out_channels, bottleneck, bn_width):
        """
        Builds a scale sub-network for the first layer

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A scale module
        """
        conv_module = self.convolve(in_channels, out_channels, 'normal',
                                    bottleneck, bn_width)
        return _DynamicInputDenseBlock(nn.ModuleList([conv_module]),
                                       self.debug)

    def convolve(self, in_channels, out_channels, conv_type,
                 bottleneck, bn_width=4):
        """
        Doing the main convolution of a specific scale in the
        MSD network

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param conv_type: convolution type
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A Sequential module of the main convolution
        """
        conv = nn.Sequential()
        tmp_channels = in_channels

        # Bottleneck before the convolution
        if bottleneck:
            tmp_channels = int(min([in_channels, bn_width * out_channels]))
            conv.add_module('Bottleneck_1x1', nn.Conv2d(in_channels,
                                                        tmp_channels,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0))
            conv.add_module('Bottleneck_BN', nn.BatchNorm2d(tmp_channels))
            conv.add_module('Bottleneck_ReLU', nn.ReLU(inplace=True))
        if conv_type == 'normal':
            conv.add_module('Spatial_forward', self.conv_l(tmp_channels,
                                                           out_channels,
                                                           kernel_size=self.ks,
                                                           stride=1,
                                                           padding=self.pad))
        elif conv_type == 'down':
            conv.add_module('Spatial_down', self.conv_l(tmp_channels, out_channels,
                                                        kernel_size=self.ks,
                                                        stride=2,
                                                        padding=self.pad))
        else: # Leaving an option to change the main conv type
            raise NotImplementedError

        conv.add_module('BN_out', nn.BatchNorm2d(out_channels))
        conv.add_module('ReLU_out', nn.ReLU(inplace=True))
        return conv

    def forward(self, x):
        cur_input = []
        outputs = []

        # Prepare the different scales' inputs of the
        # current transition/regular layer
        if self.to_drop: # Transition
            for scale in range(0, self.out_scales):
                last_same_scale = x[self.to_drop + scale]
                last_upper_scale = x[self.to_drop + scale - 1]
                cur_input.append([last_same_scale,
                                  last_upper_scale,
                                  last_same_scale])
        else: # Regular

            # Add first scale's input
            cur_input.append([x[0], x[0]])

            # Add second+ scales' input
            for scale in range(1, self.out_scales):
                last_same_scale = x[scale]
                last_upper_scale = x[scale - 1]
                cur_input.append([last_same_scale,
                                  last_upper_scale,
                                  last_same_scale])

        # Flow inputs in subnets and fill outputs
        for scale in range(0, self.out_scales):
            outputs.append(self.subnets[scale](cur_input[scale]))

        return outputs


class MSDFirstLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_scales, args):
        """
        Creates the first layer of the MSD network, which takes
        an input tensor (image) and generates a list of size num_scales
        with deeper features with smaller (spatial) dimensions.

        :param in_channels: number of input channels to the first layer
        :param out_channels: number of output channels in the first scale
        :param num_scales: number of output scales in the first layer
        :param args: other arguments
        """
        super(MSDFirstLayer, self).__init__()

        # Init params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.args = args
        self.use_gcn = args.msd_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)
        if self.use_gcn:
            print('|          First layer with GCN           |')
        else:
            print('|         First layer without GCN         |')

        self.subnets = self.create_modules()

    def create_modules(self):

        # Create first scale features
        modules = nn.ModuleList()
        if 'cifar' in self.args.data:
            current_channels = int(self.out_channels *
                                   self.args.msd_growth_factor[0])

            current_m = nn.Sequential(
                self.conv_l(self.in_channels,
                       current_channels, kernel_size=self.ks,
                       stride=1, padding=self.pad),
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True)
            )
            modules.append(current_m)
        else:
            raise NotImplementedError

        # Create second scale features and down
        for scale in range(1, self.num_scales):

            # Calculate desired output channels
            out_channels = int(self.out_channels *
                               self.args.msd_growth_factor[scale])

            # Use a strided convolution to create next scale features
            current_m = nn.Sequential(
                self.conv_l(current_channels, out_channels,
                       kernel_size=self.ks,
                       stride=2, padding=self.pad),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            # Use the output channels size for the next scale
            current_channels = out_channels

            # Append module
            modules.append(current_m)

        return modules

    def forward(self, x):
        output = [None] * self.num_scales
        current_input = x
        for scale in range(0, self.num_scales):

            # Use upper scale as an input
            if scale > 0:
                current_input = output[scale-1]
            output[scale] = self.subnets[scale](current_input)
        return output


class Transition(nn.Sequential):

    def __init__(self, channels_in, channels_out,
                 out_scales, offset, growth_factor, args):
        """
        Performs 1x1 convolution to increase channels size after reducing a spatial size reduction
        in transition layer.

        :param channels_in: channels before the transition
        :param channels_out: channels after reduction
        :param out_scales: number of scales after the transition
        :param offset: gap between original number of scales to out_scales
        :param growth_factor: densenet channel growth factor
        :return: A Parallel trainable array with the scales after channel
                 reduction
        """

        super(Transition, self).__init__()
        self.args = args

        # Define a parallel stream for the different scales
        self.scales = nn.ModuleList()
        for i in range(0, out_scales):
            cur_in = channels_in * growth_factor[offset + i]
            cur_out = channels_out * growth_factor[offset + i]
            self.scales.append(self.conv1x1(cur_in, cur_out))

    def conv1x1(self, in_channels, out_channels):
        """
        Inner function to define the basic operation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :return: A Sequential module to perform 1x1 convolution
        """
        scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        return scale

    def forward(self, x):
        """
        Propegate output through different scales.

        :param x: input to the transition layer
        :return: list of scales' outputs
        """
        if self.args.debug:
            print ("In transition forward!")

        output = []
        for scale, scale_net in enumerate(self.scales):
            if self.args.debug:
                print ("Size of x[{}]: {}".format(scale, x[scale].size()))
                print ("scale_net[0]: {}".format(scale_net[0]))
            output.append(scale_net(x[scale]))

        return output


class CifarClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """

        super(CifarClassifier, self).__init__()
        self.inner_channels = 128

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Linear(self.inner_channels, num_classes)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), self.inner_channels)
        x = self.classifier(x)
        return x


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=1):
        """
        Global convolutional network module implementation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of conv kernel
        :param stride: stride to use in the conv parts
        :param padding: padding to use in the conv parts
        :param share_weights: use shared weights for every side of GCN
        """
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(padding, 0), stride=(stride, 1))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, padding), stride=(1, stride))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, padding), stride=(1, stride))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(padding, 0), stride=(stride, 1))


    def forward(self, x):

        if GCN.share_weights:

            # Prepare input and state
            self.conv_l1.shared = 2
            self.conv_l2.shared = 2
            xt = x.transpose(2,3)

            # Left convs
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)

            # Right convs
            xrt = self.conv_l1(xt)
            xrt = self.conv_l2(xrt)
            xr = xrt.transpose(2,3)
        else:

            # Left convs
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)

            # Right convs
            xr = self.conv_r1(x)
            xr = self.conv_r2(xr)

        return xl + xr

def get_conv_params(use_gcn, args):
    """
    Calculates and returns the convulotion parameters

    :param use_gcn: flag to use GCN or not
    :param args: user defined arguments
    :return: convolution type, kernel size and padding
    """

    if use_gcn:
        GCN.share_weights = args.msd_share_weights
        conv_l = GCN
        ks = args.msd_gcn_kernel
    else:
        conv_l = nn.Conv2d
        ks = args.msd_kernel
    pad = int(math.floor(ks / 2))
    return conv_l, ks, pad