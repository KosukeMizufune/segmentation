import chainer
from chainer import links as L
from chainer import functions as F


def fixed_padding(inputs, kernel_size, dilate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeperableConv(chainer.Chain):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, dilate=1):
        super(SeperableConv, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_size, in_size, kernel_size, stride, 0, dilate=dilate,
                                         groups=in_size, nobias=True)
            self.bn = L.BatchNormalization(in_size)
            self.pointwise = L.Convolution2D(in_size, out_size, 1, 1, 0, dilate=1, groups=1, nobias=True)

    def __call__(self, x):
        h = fixed_padding(x, self.conv1.ksize, dilate=self.conv1.dilate[0])
        h = self.conv1(h)
        h = self.bn(h)
        h = self.pointwise(h)
        return h


class EntryFlowBlock(chainer.Chain):
    def __init__(self, in_size, out_size, stride=1):
        super(EntryFlowBlock, self).__init__()
        with self.init_scope():
            self.sep1 = SeperableConv(in_size, out_size, 3, 1)
            self.bn1 = L.BatchNormalization(out_size)
            self.sep2 = SeperableConv(out_size, out_size, 3, 1)
            self.bn2 = L.BatchNormalization(out_size)
            self.sep3 = SeperableConv(out_size, out_size, 3, 2)
            self.bn3 = L.BatchNormalization(out_size)
            self.conv = L.Convolution2D(in_size, out_size, 1, stride, nobias=True)
            self.bn_conv = L.BatchNormalization(out_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.sep1(x)))
        h = F.relu(self.bn2(self.sep2(h)))
        h = self.bn3(self.sep3(h))
        skip = self.bn_conv(self.conv(x))
        return F.relu(h + skip)


class MiddleFlowBlock(chainer.Chain):
    def __init__(self, in_size, out_size, dilate=1):
        super(MiddleFlowBlock, self).__init__()
        with self.init_scope():
            self.sep1 = SeperableConv(in_size, out_size, 3, 1, dilate=dilate)
            self.bn1 = L.BatchNormalization(out_size)
            self.sep2 = SeperableConv(out_size, out_size, 3, 1, dilate=dilate)
            self.bn2 = L.BatchNormalization(out_size)
            self.sep3 = SeperableConv(out_size, out_size, 3, 1, dilate=dilate)
            self.bn3 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.sep1(x)))
        h = F.relu(self.bn2(self.sep2(h)))
        h = self.bn3(self.sep3(h))
        return F.relu(h + x)


class ExitFlow(chainer.Chain):
    def __init__(self, dilate):
        super(ExitFlow, self).__init__()
        with self.init_scope():
            self.sep1 = SeperableConv(728, 728, 3, 1, dilate[0])
            self.bn1 = L.BatchNormalization(728)
            self.sep2 = SeperableConv(728, 1024, 3, 1, dilate[0])
            self.bn2 = L.BatchNormalization(1024)
            self.sep3 = SeperableConv(1024, 1024, 3, 2, dilate[0])
            self.bn3 = L.BatchNormalization(1024)
            self.sep4 = SeperableConv(1024, 1536, 3, 1, dilate[1])
            self.bn4 = L.BatchNormalization(1536)
            self.sep5 = SeperableConv(1536, 1536, 3, 1, dilate[1])
            self.bn5 = L.BatchNormalization(1536)
            self.sep6 = SeperableConv(1536, 2048, 3, 1, dilate[1])
            self.bn6 = L.BatchNormalization(2048)
            self.conv = L.Convolution2D(728, 1024, 1, 2)
            self.conv_bn = L.BatchNormalization(1024)

    def __call__(self, x):
        h = F.relu(self.bn1(self.sep1(x)))
        h = F.relu(self.bn2(self.sep2(h)))
        h = self.bn3(self.sep3(h))
        h += self.conv_bn(self.conv(x))
        h = F.relu(h)
        h = F.relu(self.bn4(self.sep4(h)))
        h = F.relu(self.bn5(self.sep5(h)))
        h = F.relu(self.bn6(self.sep6(h)))
        return h


class AlignedXception(chainer.Chain):
    def __init__(self, output_stride):
        super(AlignedXception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        with self.init_scope():
            # Entry Flow
            self.conv1 = L.Convolution2D(3, 32, 3, stride=2, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(32, 64, 3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(64)
            self.block1 = EntryFlowBlock(64, 128)
            self.block2 = EntryFlowBlock(128, 256)
            self.block3 = EntryFlowBlock(256, 728, stride=entry_block3_stride)

            # Middle Flow
            self.block4 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block5 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block6 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block7 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block8 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block9 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block10 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block11 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block12 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block13 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block14 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block15 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block16 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block17 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block18 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)
            self.block19 = MiddleFlowBlock(728, 728, dilate=middle_block_dilation)

            # Exit Flow
            self.exit = ExitFlow(exit_block_dilations)

    def __call__(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))  # relu may be unnecessary
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.exit(x)
        return x
