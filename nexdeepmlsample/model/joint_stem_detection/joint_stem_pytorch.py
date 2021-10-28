# Translation from TensorFlow model to the current PyTorch model done by Raghav

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from nexdeepml.model.model import ModelPyTorch


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SoftMaxLayer(torch.nn.Module):
  def __init__(self, in_channels, n_classes, filter_multiple, kernel_width, kernel_height, name, **kwargs): 
    super(SoftMaxLayer, self).__init__( **kwargs) # set name=name 
    p_1 = (kernel_height-1) // 2
    p_2 = (kernel_width-1) // 2
    # for handling asymm padding i.e. if even filter size is used (with stride 1 and no dilution)
    pad_residual_right = 1 if kernel_width%2==0 else 0
    pad_residual_bottom = 1 if kernel_height%2==0 else 0 
    self.asymm_padding = torch.nn.ZeroPad2d((0,pad_residual_right,0,pad_residual_bottom))
    self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=filter_multiple*16, kernel_size=(kernel_height,kernel_width), padding=(p_1,p_2))
    self.swish = Swish()
    self.conv2 = torch.nn.Conv2d(in_channels=filter_multiple*16, out_channels=n_classes, kernel_size=1, padding=(0,0)) # set name="final"

  def forward(self, inputs):
    l = self.asymm_padding(inputs)
    l = self.conv1(l)
    l = self.swish(l)
    l = self.conv2(l)
    return l  


class ConvBlock(torch.nn.Module):
  def __init__(self, in_channels, filter_multiple, kernel_width, kernel_height, dropout, name, **kwargs):
    super(ConvBlock, self).__init__(**kwargs) #set name=name
    
    p_1 = (kernel_height-1) // 2
    p_2 = (kernel_width-1) // 2
    # for handling asymm padding i.e. if even filter size is used (with stride 1 and no dilution)
    pad_residual_right = 1 if kernel_width%2==0 else 0
    pad_residual_bottom = 1 if kernel_height%2==0 else 0 
    self.asymm_padding = torch.nn.ZeroPad2d((0,pad_residual_right,0,pad_residual_bottom))
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=filter_multiple*2, kernel_size=(kernel_height,kernel_width), padding=(p_1,p_2))	
    self.swish = Swish()
    self.batch_norm = torch.nn.BatchNorm2d(num_features=filter_multiple*2)
    self.dropout_p = dropout
    self.dropout = torch.nn.Dropout(self.dropout_p)

  def forward(self, inputs):
    l = self.asymm_padding(inputs)
    l = self.conv(l)
    l = self.swish(l)
    l = self.batch_norm(l)    
    l = self.dropout(l)

    return l


class BottleNeck(torch.nn.Module):
  def __init__(self, in_channels, n_filters, name, **kwargs):
    super(BottleNeck, self).__init__(**kwargs) # set name=name
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, stride=1)
    self.swish = Swish()

  def forward(self, inputs):
    x = self.conv(inputs)
    x = self.swish(x)
    return x


def calculate_same_padding_conv(in_size, out_size, filter_size, strides):
  in_width, in_height = in_size[1], in_size[0]
  out_width, out_height = out_size[1], out_size[0]
  filter_width, filter_height = filter_size[1], filter_size[0]
  
  out_height = np.ceil(float(in_height) / float(strides[0]))
  out_width  = np.ceil(float(in_width) / float(strides[1]))

  pad_along_height = max((out_height - 1) * strides[1] +
                      filter_height - in_height, 0)
  pad_along_width = max((out_width - 1) * strides[0] +
                    filter_width - in_width, 0)
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  return int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)


def calculate_same_padding_upconv(in_size, out_size, filter_size, strides):
  in_width, in_height = in_size[1], in_size[0]
  out_width, out_height = out_size[1], out_size[0]
  filter_width, filter_height = filter_size[1], filter_size[0]

  out_height = in_height * strides[0]
  out_width  = in_width * strides[1]

  pad_along_height = max((in_height - 1) * strides[0] +
                      filter_height - out_height, 0)
  pad_along_width = max((in_width - 1) * strides[1] +
                    filter_width - out_width, 0)
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  return int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)


class DownSample(torch.nn.Module):
  def __init__(self, feature_map_size, in_channels, n_filters, kernel_width, kernel_height, name, **kwargs):
    super(DownSample, self).__init__(**kwargs) # set name=name

    # TO-DO caution: feature_map_size if different for images can raise an error 
    
    # padding='same' with stride in pytorch 
    stride = 2
    pad_left, pad_right, pad_top, pad_bottom = calculate_same_padding_conv(feature_map_size,feature_map_size,(kernel_height,kernel_width),(stride,stride))
    self.asymm_padding = torch.nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=(kernel_height,kernel_width), stride=stride)
    self.swish = Swish()
    
    # for calculating the output_size of image (need it for VisualEncoder while stacking DownSamples)
    self.output_height = feature_map_size[0]+pad_top+pad_bottom
    self.output_height = ((self.output_height - kernel_height) // stride) + 1
    self.output_width = feature_map_size[1]+pad_left+pad_right
    self.output_width = ((self.output_width - kernel_width) // stride) + 1

  def forward(self, inputs):
    x = self.asymm_padding(inputs)
    x = self.conv(x)
    x = self.swish(x)
    return x


class UpSample(torch.nn.Module):
  def __init__(self, in_channels, n_filters, kernel, name, **kwargs):
    super(UpSample, self).__init__(**kwargs) # set name=name 

    # padding='same' with stride in pytorch
    # stride = 2
    # pad_left, pad_right, pad_top, pad_bottom = calculate_same_padding_upconv(feature_map_size,feature_map_size,kernel,(stride,stride))
    # self.asymm_padding = torch.nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
    
    # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#zero-padding-non-unit-strides-transposed
    # Pytorch won't allow aymmetric padding in their padding argument in ConvTranspose2d
    # Hence have to take symmetric padding of 1 and output will be different from Tf's and Tf uses asymm padding of (0,1,0,1)  
    
    self.conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel,stride=2,padding=1)
    self.swish = Swish()
    
  def forward(self, inputs):
    # l = self.asymm_padding(inputs) 
    # l = self.conv(inputs)
    l = self.conv(inputs,output_size=(inputs.shape[2]*2,inputs.shape[3]*2))
    l = self.swish(l)
    return l		


class DenseBlock(torch.nn.Module):

  def __init__(self, in_channels, filter_multiple, kernel_width, kernel_height, dropout, name, **kwargs):
    super(DenseBlock, self).__init__(**kwargs) # set name=name

    self.bneck1 = BottleNeck(in_channels=in_channels, name=name+"_bneck_1", n_filters=filter_multiple*4)
    self.conv_block1 = ConvBlock(in_channels=self.bneck1.conv.out_channels,name=name+"_conv_block_1", filter_multiple=filter_multiple, kernel_width=kernel_width, kernel_height=kernel_height, dropout=dropout)
    
    self.bneck2 = BottleNeck(in_channels=(in_channels+self.conv_block1.conv.out_channels),name=name+"_bneck_2", n_filters=filter_multiple*4)
    self.conv_block2 = ConvBlock(in_channels=self.bneck2.conv.out_channels,name=name+"_conv_block_2", filter_multiple=filter_multiple, kernel_width=kernel_width, kernel_height=kernel_height, dropout=dropout)
    
    self.bneck3 = BottleNeck(in_channels=(in_channels+self.conv_block1.conv.out_channels+self.conv_block2.conv.out_channels),name=name+"_bneck_3", n_filters=filter_multiple*4)
    self.conv_block3 = ConvBlock(in_channels=self.bneck3.conv.out_channels,name=name+"_conv_block_3", filter_multiple=filter_multiple, kernel_width=kernel_width, kernel_height=kernel_height, dropout=dropout)


  def forward(self, inputs):

    l = self.bneck1(inputs)
    l1 = self.conv_block1(l)
    c1 = torch.cat((inputs, l1), dim=1)
    
    l = self.bneck2(c1)
    l2 = self.conv_block2(l)
    c2 = torch.cat((c1, l2), dim=1)
    
    l = self.bneck3(c2)
    l3 = self.conv_block3(l)
    c3 = torch.cat((l1, l2, l3), dim=1)

    return c3


class VisualEncoder(torch.nn.Module):

  def __init__(self, feature_map_size, in_channels, filter_multiple, kernel_conv1_width, kernel_conv1_height, kernel_dense_block_width, kernel_dense_block_height, kernel_downsample_width, kernel_downsample_height, dropout, name="encoder", **kwargs):
    super(VisualEncoder, self).__init__(**kwargs) # set name=name 

    p_1 = (kernel_conv1_height-1) // 2
    p_2 = (kernel_conv1_width-1) // 2
    # for handling asymm padding i.e. if even filter size is used (with stride 1 and no dilution)
    pad_residual_right = 1 if kernel_conv1_width%2==0 else 0
    pad_residual_bottom = 1 if kernel_conv1_height%2==0 else 0 
    self.asymm_padding = torch.nn.ZeroPad2d((0,pad_residual_right,0,pad_residual_bottom))
    self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=filter_multiple*16, kernel_size=(kernel_conv1_height,kernel_conv1_width),padding=(p_1,p_2))

    self.swish_activation = Swish()

    self.dense1 = DenseBlock(in_channels=self.conv.out_channels, name="encoder_dense1", filter_multiple=filter_multiple, kernel_width=kernel_dense_block_width, kernel_height=kernel_dense_block_height, dropout=dropout)
    self.bneck1 = BottleNeck(
      in_channels=(
          self.conv.out_channels +
          self.dense1.conv_block1.conv.out_channels + 
          self.dense1.conv_block2.conv.out_channels + 
          self.dense1.conv_block3.conv.out_channels
          ),
      name="encoder_bottle_neck1", 
      n_filters=filter_multiple*11
      )
    self.downsample1 = DownSample(feature_map_size=feature_map_size, in_channels=self.bneck1.conv.out_channels, name="encoder_downsample1", n_filters=filter_multiple*11, kernel_width=kernel_downsample_width, kernel_height=kernel_downsample_height)
    
    self.dense2 = DenseBlock(in_channels=self.downsample1.conv.out_channels, name="encoder_dense2", filter_multiple=filter_multiple, kernel_width=kernel_dense_block_width, kernel_height=kernel_dense_block_height, dropout=dropout)
    self.bneck2 = BottleNeck(
        name="encoder_bottle_neck2", 
        n_filters=(filter_multiple*8)+1,
        in_channels=(
            self.downsample1.conv.out_channels +
            self.dense2.conv_block1.conv.out_channels + 
            self.dense2.conv_block2.conv.out_channels + 
            self.dense2.conv_block3.conv.out_channels
          ),
      )
    feature_map_size = (self.downsample1.output_height, self.downsample1.output_width)
    self.downsample2 = DownSample(feature_map_size=feature_map_size, in_channels=self.bneck2.conv.out_channels,name="encoder_downsample2", n_filters=(filter_multiple*8)+1, kernel_width=kernel_downsample_width, kernel_height=kernel_downsample_height)
    
    self.dense3 = DenseBlock(in_channels=self.downsample2.conv.out_channels, name="encoder_dense3", filter_multiple=filter_multiple, kernel_width=kernel_dense_block_width, kernel_height=kernel_dense_block_height, dropout=dropout)
    self.bneck3 = BottleNeck(
        name="encoder_bottle_neck3", 
        n_filters=filter_multiple*7,
        in_channels=(
            self.downsample2.conv.out_channels +
            self.dense3.conv_block1.conv.out_channels + 
            self.dense3.conv_block2.conv.out_channels + 
            self.dense3.conv_block3.conv.out_channels
          ),
      )
    feature_map_size = (self.downsample2.output_height, self.downsample2.output_width)
    self.downsample3 = DownSample(feature_map_size=feature_map_size, in_channels=self.bneck3.conv.out_channels,name="encoder_downsample3", n_filters=filter_multiple*7, kernel_width=kernel_downsample_width, kernel_height=kernel_downsample_height)
    
    self.dense4 = DenseBlock(in_channels=self.downsample3.conv.out_channels,name="encoder_dense4", filter_multiple=filter_multiple, kernel_width=kernel_dense_block_width, kernel_height=kernel_dense_block_height, dropout=dropout)

  def forward(self, inputs):
    x = self.asymm_padding(inputs)
    x = self.conv(x)
    x = self.swish_activation(x)

    dense1 = self.dense1(x)
    concat_1 = torch.cat((x, dense1), dim=1)
    x = self.bneck1(concat_1)
    x = self.downsample1(x)

    dense2 = self.dense2(x)
    concat_2 = torch.cat((x, dense2), dim=1)
    x = self.bneck2(concat_2)
    x = self.downsample2(x)

    dense3 = self.dense3(x)
    concat_3 = torch.cat((x, dense3), dim=1)
    x = self.bneck3(concat_3)
    x = self.downsample3(x)

    visual_code = self.dense4(x)

    return visual_code, concat_3, concat_2, concat_1


class VisualDecoder(torch.nn.Module):

  def __init__(self, in_channels_from_encoder, prefix, filter_multiple, kernel_width_dense_block, kernel_height_dense_block, kernel_width_upsample, kernel_height_upsample, dropout, name="decoder", **kwargs):
    super(VisualDecoder, self).__init__(**kwargs) # set name=name

    self.upsample1 = UpSample(in_channels=in_channels_from_encoder[0],name=prefix+"_decoder_upsample_1", n_filters=filter_multiple*6, kernel=(kernel_width_upsample, kernel_height_upsample))
    self.dense1 = DenseBlock(in_channels=(in_channels_from_encoder[0]+in_channels_from_encoder[1]),name=prefix+"_denseblock1", filter_multiple=filter_multiple, kernel_width=kernel_width_dense_block, kernel_height=kernel_height_dense_block, dropout=dropout)
    
    self.upsample2 = UpSample(
        in_channels=(
            self.dense1.conv_block1.conv.out_channels + 
            self.dense1.conv_block2.conv.out_channels + 
            self.dense1.conv_block3.conv.out_channels
          ),
        name=prefix+"_decoder_upsample_2", 
        n_filters=filter_multiple*6, 
        kernel=(kernel_width_upsample, kernel_height_upsample)
        )
    self.dense2 = DenseBlock(in_channels=(self.upsample2.conv.out_channels+in_channels_from_encoder[2]),name=prefix+"_denseblock2", filter_multiple=filter_multiple, kernel_width=kernel_width_dense_block, kernel_height=kernel_height_dense_block, dropout=dropout)
    
    
    self.upsample3 = UpSample(
        in_channels=(
            self.dense2.conv_block1.conv.out_channels + 
            self.dense2.conv_block2.conv.out_channels + 
            self.dense2.conv_block3.conv.out_channels
          ),
        name=prefix+"_decoder_upsample_3", 
        n_filters=filter_multiple*6, 
        kernel=(kernel_width_upsample, kernel_height_upsample)
        )
    self.dense3 = DenseBlock(in_channels=(self.upsample3.conv.out_channels+in_channels_from_encoder[3]),name=prefix+"_denseblock3", filter_multiple=filter_multiple, kernel_width=kernel_width_dense_block, kernel_height=kernel_height_dense_block, dropout=dropout)

  def forward(self, inputs, l1, l2, l3):

    l = self.upsample1(inputs)
    l = torch.cat((l, l1), dim=1)
    l = self.dense1(l)

    l = self.upsample2(l)
    l = torch.cat((l, l2), dim=1)
    l = self.dense2(l)

    l = self.upsample3(l)
    l = torch.cat((l, l3), dim=1)
    l = self.dense3(l)
    return l


class CropWeed(ModelPyTorch):
  def __init__(
        self, 
        name="cropweed_model", 
        input_shape=(3, 384, 512), 
        n_classes=3, 
        use_semantic_segmentation=True, 
        use_stem_detection=False,
        dropout_p=0.25, 
        filter_multiple_encoder=2, 
        filter_multiple_decoder_semantic=2, 
        filter_multiple_decoder_instance=2, 
        filter_multiple_decoder_stem=2, 
        # use_coordinate_conv_layer=False, 
        # use_instance_segmentation=True, 
        
        kernel_conv1_width=2, 
        kernel_conv1_height=2, 
        encoder_kernel_dense_block_width=3, 
        encoder_kernel_dense_block_height=3,
        encoder_kernel_downsample_width=3, 
        encoder_kernel_downsample_height=3, 

        sem_decoder_kernel_dense_block_width=3, 
        sem_decoder_kernel_dense_block_height=3, 
        sem_decoder_kernel_upsample_width=3, 
        sem_decoder_kernel_upsample_height=3, 

        # inst_decoder_kernel_dense_block_width=3, 
        # inst_decoder_kernel_dense_block_height=3, 
        # inst_decoder_kernel_upsample_width=3, 
        # inst_decoder_kernel_upsample_height=3, 

        stem_decoder_kernel_dense_block_width=3, 
        stem_decoder_kernel_dense_block_height=3, 
        stem_decoder_kernel_upsample_width=3,  # 2
        stem_decoder_kernel_upsample_height=3,  # 2

        kernel_softmax_layer_semantic=2,  # 1
        # kernel_softmax_layer_instance=2, 
        kernel_softmax_layer_stem=2,  # 1
        # gauss_stems=False,
        # just_weed_stem=False,
        # feature_space= 3, 
        # instance_loss="DL",
        **kwargs):

    super(CropWeed, self).__init__(**kwargs) # set name=name

    self.use_semantic_segmentation = use_semantic_segmentation
    # self.use_instance_segmentation = use_instance_segmentation
    self.use_stem_detection = use_stem_detection
    # self.use_coordinate_conv_layer=use_coordinate_conv_layer

    self.encoder = VisualEncoder(
      feature_map_size=(input_shape[1],input_shape[2]), 
      in_channels=input_shape[0],
      filter_multiple=filter_multiple_encoder, 
      kernel_conv1_width=kernel_conv1_width, 
      kernel_conv1_height=kernel_conv1_height,
      kernel_dense_block_width=encoder_kernel_dense_block_width, 
      kernel_dense_block_height=encoder_kernel_dense_block_height,
      kernel_downsample_width=encoder_kernel_downsample_width, 
      kernel_downsample_height=encoder_kernel_downsample_height,
      dropout=dropout_p, 
      name="encoder"
      )
    
    # if self.use_coordinate_conv_layer:
    # 	if self.use_coordinate_conv_layer == 1.0:
    # 		with_r = True
    # 	else:
    # 		with_r = False
      
    # 	self.encoded_coord_conv = CoordConv(x_dim=int(input_shape[1]/8), y_dim=int(input_shape[2]/8), with_r=with_r, out_channels=filter_multiple_encoder*6, kernel_size=(encoder_kernel_dense_block_width, encoder_kernel_dense_block_height), padding="same", kernel_initializer='he_uniform', data_format='channels_first', name='encoded_coord_conv2d')
    # 	self.l1_coord_conv = CoordConv(x_dim=int(input_shape[1]/4), y_dim=int(input_shape[2]/4), with_r=with_r, out_channels=filter_multiple_encoder*7, kernel_size=(encoder_kernel_dense_block_width, encoder_kernel_dense_block_height), padding="same", kernel_initializer='he_uniform', data_format='channels_first', name='skip3_coord_conv2d')
    # 	self.l2_coord_conv = CoordConv(x_dim=int(input_shape[1]/2), y_dim=int(input_shape[2]/2), with_r=with_r, out_channels=filter_multiple_encoder*8+1, kernel_size=(encoder_kernel_dense_block_width, encoder_kernel_dense_block_height), padding="same", kernel_initializer='he_uniform', data_format='channels_first', name='skip2_coord_conv2d')
    # 	self.l3_coord_conv = CoordConv(x_dim=int(input_shape[1]), y_dim=int(input_shape[2]), with_r=with_r, out_channels=filter_multiple_encoder*11, kernel_size=(encoder_kernel_dense_block_width, encoder_kernel_dense_block_height), padding="same", kernel_initializer='he_uniform', data_format='channels_first', name='skip1_coord_conv2d')

    if self.use_semantic_segmentation:
      self.semantic_decoder = VisualDecoder(
          in_channels_from_encoder=[3*2*filter_multiple_encoder, (filter_multiple_encoder*8)+1+3*2*filter_multiple_encoder, filter_multiple_encoder*11+3*2*filter_multiple_encoder, filter_multiple_encoder*16+3*2*filter_multiple_encoder],
          prefix="semantic", 
          filter_multiple=filter_multiple_decoder_semantic, 
          kernel_width_dense_block=sem_decoder_kernel_dense_block_width, 
          kernel_height_dense_block=sem_decoder_kernel_dense_block_height, 
          kernel_width_upsample=sem_decoder_kernel_upsample_width, 
          kernel_height_upsample=sem_decoder_kernel_upsample_height, 
          dropout=dropout_p, 
          name="semantic_decoder"
          )
                        
      self.semantic_softmax = SoftMaxLayer(
          in_channels=3*2*filter_multiple_decoder_semantic,
          n_classes=n_classes, 
          filter_multiple=filter_multiple_decoder_semantic, 
          kernel_width=kernel_softmax_layer_semantic, 
          kernel_height=kernel_softmax_layer_semantic, 
          name="semantic_softmax"
          )

    # if self.use_instance_segmentation:
    # 	self.instance_decoder = VisualDecoder(prefix="instance", 
    # 										filter_multiple=filter_multiple_decoder_instance, 
    # 										kernel_width_dense_block=inst_decoder_kernel_dense_block_width, 
    # 										kernel_height_dense_block=inst_decoder_kernel_dense_block_height, 
    # 										kernel_width_upsample=inst_decoder_kernel_upsample_width, 
    # 										kernel_height_upsample=inst_decoder_kernel_upsample_height, 
    # 										dropout=dropout_p, 
    # 										name="instance_decoder")
    # 	if instance_loss == "VECTOR_LOSS":
    # 		self.instance_softmax = SoftMaxLayer(n_classes=2, 
    # 											filter_multiple=filter_multiple_decoder_instance, 
    # 											kernel_width=kernel_softmax_layer_instance, 
    # 											kernel_height=kernel_softmax_layer_instance, 
    # 											name="instance_softmax")
    # 	elif instance_loss == "DL":
    # 		self.instance_softmax = SoftMaxLayer(n_classes=feature_space, 
    # 											filter_multiple=filter_multiple_decoder_instance, 
    # 											kernel_width=kernel_softmax_layer_instance, 
    # 											kernel_height=kernel_softmax_layer_instance, 
    # 											name="instance_softmax")
      
    if self.use_stem_detection:
      self.stem_decoder = VisualDecoder(
          in_channels_from_encoder=[3*2*filter_multiple_encoder, (filter_multiple_encoder*8)+1+3*2*filter_multiple_encoder, filter_multiple_encoder*11+3*2*filter_multiple_encoder, filter_multiple_encoder*16+3*2*filter_multiple_encoder],
          prefix="stem", 
          filter_multiple=filter_multiple_decoder_stem, 
          kernel_width_dense_block=stem_decoder_kernel_dense_block_width, 
          kernel_height_dense_block=stem_decoder_kernel_dense_block_height, 
          kernel_width_upsample=stem_decoder_kernel_upsample_width, 
          kernel_height_upsample=stem_decoder_kernel_upsample_height, 
          dropout=dropout_p, 
          name="stem_decoder"
          )

      self.stem_softmax = SoftMaxLayer(
          in_channels=3*2*filter_multiple_decoder_stem,
          n_classes=3, 
          filter_multiple=filter_multiple_decoder_stem, 
          kernel_width=kernel_softmax_layer_stem, 
          kernel_height=kernel_softmax_layer_stem, 
          name="stem_softmax"
          )

  def forward(self, inputs):
    
    encoded, l1, l2, l3 = self.encoder(inputs)

    # if self.use_coordinate_conv_layer:
    # 	encoded = self.encoded_coord_conv(encoded)
    # 	l1 = self.l1_coord_conv(l1)
    # 	l2 = self.l2_coord_conv(l2)
    # 	l3 = self.l3_coord_conv(l3)

    outputs=[]

    if self.use_semantic_segmentation: 
      self.semantic_features = self.semantic_decoder(encoded, l1=l1, l2=l2, l3=l3)
      self.semantic_features = self.semantic_softmax(self.semantic_features)
      outputs.append(self.semantic_features)
    
    # if self.use_instance_segmentation: 
    #   self.instance_features = self.instance_decoder(encoded, l1=l1, l2=l2, l3=l3)
    #   self.instance_features = self.instance_softmax(self.instance_features)
    #   outputs.append(self.instance_features)

    
    if self.use_stem_detection: 
      self.stem_features = self.stem_decoder(encoded, l1=l1, l2=l2, l3=l3)
      self.stem_features = self.stem_softmax(self.stem_features)
      outputs.append(self.stem_features)

    return outputs
    