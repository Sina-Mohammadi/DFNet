from keras import backend as K
K.set_image_data_format("channels_last")
from Sharpenning_Loss_Function import Sharpenning_Loss
from Modules_and_Upsample import MAG_Module, AMI_Module, BilinearUpsampling
from keras.layers import Conv2D, BatchNormalization, Activation  
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile, NASNetLarge
import os
from urllib.request import urlretrieve
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
from keras.utils import to_categorical
from CustomImageDataGenerator import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from config import cfg



class DFNet():

  def __init__(self, batch_size, learning_rate, epochs, Lambda, train_set_directory,
               save_directory, Backbone_model, use_multiprocessing, show_ModelSummary):

    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.Lambda = Lambda                            #The balance parameter of the Sharpenning loss
    self.train_set_directory = train_set_directory  #Path to folder containing Images folder and Masks folder
    self.save_directory = save_directory            #Path to save folder
    self.Backbone_model = Backbone_model            #VGG16 or ResNet50 or NASNetMobile or NASNetLarge
    self.use_multiprocessing = use_multiprocessing  #True or False
    self.show_ModelSummary = show_ModelSummary      #True or False

    self.Build_model()                              #Building and compiling DFNet
    self.PrepareData()                              #Preparing the images and masks for training

########################################################

  def Build_model(self):

    ############# Feature Extraction Network ##################

    if self.Backbone_model == "VGG16":

      Backbone = VGG16(input_shape=(352,352,3), include_top=False, weights='imagenet')

      Stages=[Backbone.get_layer("block5_conv3").output, Backbone.get_layer("block4_conv3").output,
              Backbone.get_layer("block3_conv3").output]

              # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                              
    elif self.Backbone_model == "ResNet50":

      Backbone = ResNet50(input_shape=(352,352,3), include_top=False, weights='imagenet')

      Stages=[Backbone.output, Backbone.get_layer("conv4_block6_out").output,
              Backbone.get_layer("conv3_block4_out").output, Backbone.get_layer("conv2_block3_out").output]

              # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
    elif self.Backbone_model == "NASNetMobile":

      Backbone = NASNetMobile(input_shape=(352,352,3), include_top=False, weights=None)
      if os.path.exists('./NASNet-mobile-no-top.h5'):
          pass
      else:
          urlretrieve('https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-mobile-no-top.h5', './NASNet-mobile-no-top.h5')
      Backbone.load_weights('./NASNet-mobile-no-top.h5')

      Stages=[Backbone.output, Backbone.get_layer("activation_131").output,
              Backbone.get_layer("activation_72").output, Backbone.get_layer("activation_13").output]

              # # # # # # # # # # # # # # # # # # # # # # # # # # # #
          
    elif self.Backbone_model == "NASNetLarge":

      Backbone = NASNetLarge(input_shape=(352,352,3), include_top=False, weights=None)
      if os.path.exists('./NASNet-large-no-top.h5'):
          pass
      else:
          urlretrieve('https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5', './NASNet-large-no-top.h5')
      Backbone.load_weights('./NASNet-large-no-top.h5')

      Stages=[Backbone.output, Backbone.get_layer("activation_319").output,
              Backbone.get_layer("activation_260").output, Backbone.get_layer("activation_201").output]

              # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    else:
      raise ValueError("Enter the name of the model correctly! It must be one of the following options: VGG16, ResNet50, NASNetMobile, NASNetLarge")

              # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ExtractedFeatures = []
    Number_of_Filters = [192,128,96,64]

    for Stage,Num in zip(Stages,Number_of_Filters):
      MAG_output=MAG_Module(Stage, Num)
      extr=Conv2D(Num, (1, 1), padding='same')(MAG_output) 
      ExtractedFeatures.append(extr)    #Extracted features from the Feature Extraction Network


    ############### Feature Integration Network ##################

    z = BilinearUpsampling(output_size=(ExtractedFeatures[0]._keras_shape[1]*2, ExtractedFeatures[0]._keras_shape[2]*2))(ExtractedFeatures[0])

    for i in range(len(ExtractedFeatures)-1):
      z = AMI_Module(z, ExtractedFeatures[i+1], ExtractedFeatures[i+1]._keras_shape[-1])
      z = BilinearUpsampling(output_size=(z._keras_shape[1]*2,z._keras_shape[2]*2))(z)

    z = Conv2D(64, (3, 3), padding='same', strides=(1,1))(z)
    z = BatchNormalization(axis=-1)(z)
    z = Activation('relu')(z)
    z = BilinearUpsampling(output_size=(352,352))(z)
    z = Conv2D(2, (1, 1), padding='same')(z)
    z = Activation('softmax')(z)


    self.model = Model(Backbone.input,z)
    self.model.compile(optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss=Sharpenning_Loss(self.Lambda), metrics=["accuracy"])

    if self.show_ModelSummary == True:
      self.model.summary()

 ########################################################################################  

  def PrepareData(self):

    def preimage(image):
        image = image / 255
        image = image.astype(np.float32)
        return image
      
    def premask(mask):
        label = mask / 255
        label = (label >= 0.5).astype(np.bool)
        label = to_categorical(label, num_classes=2)
        return label


    image_datagen_train = ImageDataGenerator(rotation_range=12,horizontal_flip=True,preprocessing_function=preimage)
    
    mask_datagen_train = ImageDataGenerator(x_categorical=2,rotation_range=12,horizontal_flip=True,preprocessing_function=premask)


    image_generator_train = image_datagen_train.flow_from_directory(self.train_set_directory+'Images/',target_size=(352, 352),
                                                                    class_mode=None,seed=1,batch_size=self.batch_size)

    mask_generator_train = mask_datagen_train.flow_from_directory(self.train_set_directory+'Masks/',target_size=(352, 352),
                                                                  class_mode=None,color_mode="grayscale",seed=1,batch_size=self.batch_size)

    self.train_generator = zip(image_generator_train, mask_generator_train)
    self.steps_per_epoch = len(image_generator_train) 

##################################################################################

  def train(self):

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.0001)
    #If the training loss does not decrease for 10 epochs, the learning rate is divided by 10.
    
    check_point = ModelCheckpoint(self.save_directory+'ModelCheckpoint-{epoch:03d}-{loss:.4f}.hdf5',monitor='loss', verbose=1, save_best_only=True , mode='min')
    
    logger = CSVLogger(self.save_directory+'logger.log',append=True)
    
    callbacks = [reduce_lr ,check_point , logger]

    self.model.fit_generator(self.train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                             use_multiprocessing=self.use_multiprocessing, verbose=1, callbacks=callbacks, shuffle=True)  
    
    
    
if __name__ == "__main__":
    
  DFNetModel=DFNet(batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, epochs=cfg.epochs, train_set_directory=cfg.train_set_directory,
        save_directory=cfg.save_directory, Backbone_model=cfg.Backbone_model, use_multiprocessing=cfg.use_multiprocessing, show_ModelSummary=cfg.show_ModelSummary)    
  
  DFNetModel.train()  
   
    