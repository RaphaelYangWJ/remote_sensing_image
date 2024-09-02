import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Resize
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


class zoom_rate_generator(object):

    # Constructor
    def __init__(self):
        # pretrained models
        self.vgg_model = models.vgg19(weights=True)
        self.resnet_model = models.resnet50(pretrained=True)
        self.densenet_model = models.densenet121(pretrained=True)


    def processing_pipeline(self, img, patch_size, output_dir):
        self.img = img
        self.patch_size = patch_size

        # pipeline running
        self.global_context()
        self.feature_embedding()
        processed_tensor = self.multi_scale_patch_generation()
        processed_tensor = processed_tensor.numpy()
        print(processed_tensor.shape)
        np.savez_compressed(output_dir, processed_tensor)
        # torch.save(processed_tensor, output_dir)


    # === Generate global context ===
    def global_context(self):



        # right_up_trans = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.RandomHorizontalFlip(p=1),
        #     transforms.ToTensor(),
        # ])
        #
        # # Transform 2: Left-Up Resize
        # left_up_trans = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.ToTensor(),
        # ])
        #
        # # Transform 3: Down Vertical Flip
        # down_trans = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomVerticalFlip(p=1),
        #     transforms.ToTensor(),
        # ])
        #
        # # Image Processing and Concatenation
        # left_up_img = left_up_trans(self.img)
        # right_up_img = right_up_trans(self.img)
        # upper_img = torch.cat((left_up_img, right_up_img),dim=2)
        # down_img = down_trans(upper_img)
        # self.img = torch.cat((upper_img, down_img),dim=1)
        self.img = transforms.ToTensor()(self.img)
        print(self.img.shape)
        if self.img.shape[1] != 1000:
            resizer = transforms.Compose([transforms.Resize((1000, 1000))])
            self.img = resizer(self.img)
        self.img = self.img.unsqueeze_(dim=0)
        print(self.img.shape)
        # compute zoom_coverage output dim
        self.zoom_matrix_dim = int(self.img.shape[2] / self.patch_size)
    # === Compute Feature Embedding ===
    def feature_embedding(self):
        self._vgg_feature_extractor()
        self._resnet_feature_extractor()
        self._densenet_feature_extractor()
        # normalization
        self.vgg_feature_embdding = self._embedding_normalization(self.vgg_feature_embdding)
        self.resnet_feature_embdding = self._embedding_normalization(self.resnet_feature_embdding)
        self.densenet_feature_embdding = self._embedding_normalization(self.densenet_feature_embdding)
        # aggregation with averaging
        self.feature_embed = np.round((self.vgg_feature_embdding +
                              self.resnet_feature_embdding +
                              self.densenet_feature_embdding)/3)
    # Feature Embedding: VGG
    def _vgg_feature_extractor(self):
        num_features = self.vgg_model.classifier[0].in_features
        new_classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.zoom_matrix_dim*self.zoom_matrix_dim)  # output size customization
        )
        # replace the classifier layer
        self.vgg_model.classifier = new_classifier
        # model evaluation mode
        self.vgg_model.eval()
        # get the feature embedding
        self.vgg_feature_embdding = self.vgg_model(self.img).view(1,self.zoom_matrix_dim,self.zoom_matrix_dim).permute(1,2,0).detach().numpy()
    # Feature Embedding: ResNet
    def _resnet_feature_extractor(self):
        num_features = self.resnet_model.fc.in_features
        new_classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.zoom_matrix_dim*self.zoom_matrix_dim)  # output size customization
        )
        # replace the classifier layer
        self.resnet_model.fc = new_classifier
        # model evaluation mode
        self.resnet_model.eval()
        # get the feature embedding
        self.resnet_feature_embdding = self.resnet_model(self.img).view(1,self.zoom_matrix_dim,self.zoom_matrix_dim).permute(1,2,0).detach().numpy()
    # Feature Embedding: ImageNet
    def _densenet_feature_extractor(self):
        num_features = self.densenet_model.classifier.in_features
        new_classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.zoom_matrix_dim*self.zoom_matrix_dim)  # output size customization
        )
        # replace the classifier layer
        self.densenet_model.classifier = new_classifier
        # model evaluation mode
        self.densenet_model.eval()
        # get the feature embedding
        self.densenet_feature_embdding = self.densenet_model(self.img).view(1,self.zoom_matrix_dim,self.zoom_matrix_dim).permute(1,2,0).detach().numpy()
    def _embedding_normalization(self,feature):
        # 1. Trim out outliers
        # compute the mean and standard deviation
        mean = np.mean(feature)
        std = np.std(feature)

        # trim out outliers
        threshold = 3
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        # replace the outliers with boundary values
        clipped_matrix = np.clip(feature, lower_bound, upper_bound)

        # find the max and min values
        min_value = np.min(clipped_matrix)
        max_value = np.max(clipped_matrix)

        # scaled
        normalized_matrix = ((clipped_matrix - min_value) / (max_value - min_value)) * (10 - 1) + 1
        return normalized_matrix

    # === Patch Extraction, Zooming, Generation ===
    def multi_scale_patch_generation(self):
        # create a container
        img_container = [self.img]
        zoom_matrix = np.squeeze(self.feature_embed,axis=2)

        for row in range(0,self.img.shape[2],self.patch_size):
            for col in range(0,self.img.shape[3],self.patch_size):
                extracted_patch = self._patch_extract(row_loc=row, col_loc=col)
                zoom_rate = zoom_matrix[int((row/self.patch_size)):int((row/self.patch_size))+1,
                                        int((col/self.patch_size)):int((col/self.patch_size))+1][0][0]
                zoomed_patch = self._patch_zoom(extracted_patch, int(zoom_rate))
                pad_patch = self._patch_padding(zoomed_patch)
                img_container.append(pad_patch)

        img_container = torch.cat(img_container,dim=0)
        return img_container
    def _patch_extract(self, row_loc, col_loc):
        return self.img[:,:,row_loc:(row_loc+self.patch_size),col_loc:(col_loc+self.patch_size)] # patch cut
    def _patch_zoom(self, extracted_patch, zoom_rate):
        torch_resize = Resize([self.patch_size*zoom_rate,self.patch_size*zoom_rate]) # define resize class
        return torch_resize(extracted_patch)
    def _patch_padding(self, zoomed_patch):
        padding_value = int((self.img.shape[2] - zoomed_patch.shape[2])/2)
        zero_padding = nn.ZeroPad2d(padding=(padding_value, padding_value, padding_value, padding_value))
        return zero_padding(zoomed_patch)


if __name__ == '__main__':

    source_data_dir = "../RP1_Mamba_Unet/dataset/OpenEarthMap/OpenEarthMap_wo_xBD/"
    target_data_dir = "../RP1_Mamba_Unet/dataset/OpenEarthMap/OpenEarthMap_Tensors/"
    patch_size = 100


    for item in os.listdir(source_data_dir):
        if item.endswith("txt") or item.endswith("csv"):
            None
        else:
            img_list = os.listdir(source_data_dir+item+"/images/")
            # path preparation
            if len(img_list):
                source_dir = source_data_dir + item + "/images/"
                target_dir = target_data_dir + item + "/images/"
                # Make path
                os.makedirs(target_dir) if os.path.exists(target_dir) == False else None

                for img_item in img_list:
                    if os.path.exists(target_dir + img_item + ".npz"):
                        None
                    else:
                        if img_item.endswith(".tif"):
                            processor = zoom_rate_generator()
                            img = Image.open(source_dir+img_item)
                            target_tensor_dir = target_dir + img_item + ".npz"
                            processor.processing_pipeline(img, patch_size, target_tensor_dir)
                            print("### Complete -> ", img_item)
        print("### Complete for the item of ", item)