import torch
import torch.nn as nn
import torchvision.models as models


## UTILITIES

def isnan(tensor):
    return True if torch.sum(torch.isnan(tensor)).item() >= 1 else False


def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def convert_relu_to_ELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ELU())
        else:
            convert_relu_to_ELU(child)


#####################################################


class CNNGeometric(nn.Module):
    def __init__(self, output_dim=1000,
                 feature_extraction_cnn='vgg',
                 feature_extraction_last_layer='',
                 return_correlation=False,
                 fr_kernel_sizes=[7, 5, 5],
                 fr_channels=[225, 128, 64],
                 image_size=240,
                 feature_self_matching=False,
                 normalize_features=True,
                 normalize_matches=True,
                 batch_normalization=True,
                 train_fe=False,
                 train_fr=True,
                 use_cuda=True,
                 matching_type='correlation',
                 computeH=True):
        super(CNNGeometric, self).__init__()
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=use_cuda,
                                                   image_size=image_size)

        self.FeatureCorrelation = FeatureCorrelation(shape='3D', normalization=normalize_matches,
                                                     matching_type=matching_type)
        self.FeatureRegression = FeatureRegression(output_dim, train_fr=train_fr,
                                                   use_cuda=use_cuda,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization,
                                                   image_size=image_size)

    def forward(self, tnf_batch):
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])

        correlation = self.FeatureCorrelation(feature_A, feature_B)

        theta = self.FeatureRegression(correlation)

        return theta


class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, last_layer='', use_cuda=True,
                 refNetV1=False, image_size=240):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization

        if feature_extraction_cnn == 'vgg':
            print("USING VGG")
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101':
            print("USING RESNET 101")
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])
        if feature_extraction_cnn == 'resnet18':  # maybe should use resnet50 for coarse net
            # keep AdapAvgPool layer and remove classif layer
            self.model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
            # convert_relu_to_ELU(self.model)
        if feature_extraction_cnn == 'resnet50':  # maybe should use resnet50 for coarse net
            # keep AdapAvgPool layer and remove  classif layer
            self.model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        #  convert_relu_to_ELU(self.model)

        if feature_extraction_cnn == 'wideresnet101':  # maybe should use resnet50 for coarse net
            # keep AdapAvgPool layer and remove  classif layer
            self.model = nn.Sequential(*list(models.wide_resnet101_2(pretrained=True).children())[:-3])
            self.model[6][0].conv2.stride = (1, 1)
            self.model[6][0].downsample[0].stride = (1, 1)
            for x in range(8, 23):
                self.model[6][x] = torch.nn.Identity()
        #  convert_relu_to_ELU(self.model)

        if not train_fe:
            print("FREEZING FE")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True, matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        if self.matching_type == 'correlation':
            if self.shape == '3D':
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
                feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B, feature_A)
                correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
            elif self.shape == '4D':
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b, c, h * w).transpose(1, 2)  # size [b,c,h*w]
                feature_B = feature_B.view(b, c, h * w)  # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A, feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b, h, w, h, w).unsqueeze(1)
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

            return correlation_tensor

        if self.matching_type == 'subtraction':
            return feature_A.sub(feature_B)

        if self.matching_type == 'concatenation':
            return torch.cat((feature_A, feature_B), 1)


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, batch_normalization=True, kernel_sizes=[7, 5, 5],
                 channels=[225, 128, 64], train_fr=True):
        super(FeatureRegression, self).__init__()

        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers - 1):
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i + 1]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear1 = nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], 5000)
        self.linear2 = nn.Linear(5000, 5000)
        self.linear3 = nn.Linear(5000, output_dim)
        self.relu = nn.ReLU(inplace=True)

        if not train_fr:
            print("FREEZING FR")
            for param in self.conv.parameters():
                param.requires_grad = False
            for param in self.linear.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # changed from view due to bug,  https://github.com/agrimgupta92/sgan/issues/22
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
