'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        x_in = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(x_in)
        x = nn.functional.relu(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # ?
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = [BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1)]
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        x = self.conv1(images)
        # x = self.maxpool(x) # ?
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

    def visualize(self, logdir):
        """ Visualize the kernel in the desired directory """

        # filters, biases = model.layers[1].get_weights()
        # f_min, f_max = filters.min(), filters.max()
        # filters = (filters - f_min) / (f_max - f_min)
        #
        # n_filters, ix = 6, 1
        # for i in range(n_filters):
        #     # get the filter
        #     f = filters[:, :, :, i]
        #     # plot each channel separately
        #     for j in range(3):
        #         # specify subplot and turn of axis
        #         ax = pyplot.subplot(n_filters, 3, ix)
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         # plot filter channel in grayscale
        #         pyplot.imshow(f[:, :, j], cmap='gray')
        #         ix += 1
        # # show the figure
        # pyplot.show()


        raise NotImplementedError
