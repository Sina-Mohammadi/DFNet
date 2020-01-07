from keras import backend as K
from tensorflow.image import resize_bilinear
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
from keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Activation  
from keras.layers import Reshape, multiply, add, GlobalAveragePooling2D, Concatenate




def CA_Block(x,r=8):
    
    ''' Creates the Channel Attention Block
    Args:
        x: input tensor
        r: reduction ratio
    Returns: 
        a keras tensor
        
    This block is similar to [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)'''
    
    num_ch = x._keras_shape[-1]
    y = GlobalAveragePooling2D()(x)
    y = Reshape((1,1,int(num_ch)))(y)
    y = Conv2D(int(num_ch/r), (1, 1), activation='relu', use_bias=False)(y)
    y = Conv2D(int(num_ch), (1, 1), activation='sigmoid', use_bias=False)(y)
    y = multiply([x, y])

    return y




def LargeKernel(x,num_f,k):
    
    '''Args:
        x: input tensor
        num_f: number of filters
        k: size of the LargeKernel
      Returns: 
        a keras tensor'''
        
    y1 = Conv2D(num_f,(1,k),strides=(1,1),padding='same')(x)
    y1 = Conv2D(num_f,(k,1),strides=(1,1),padding='same')(y1)
    y2 = SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=int(((k-3)/2)+1))(x)
    y = add([y1,y2])
    
    return y



  
def MAG_Module(x,num_f):
    
    ''' Creates the MAG Module"
    Args:
        x: input tensor
        num_f: number of filters
    Returns: 
        a keras tensor'''    
  
    y0 = Conv2D(num_f, (1, 1), padding='same')(x)    
    y0 = BatchNormalization(epsilon=1e-5)(y0)
    y0 = Activation('relu')(y0)  
  
    y1 = Conv2D(num_f, (3, 3), padding='same')(x)    
    y1 = BatchNormalization(epsilon=1e-5)(y1)
    y1 = Activation('relu')(y1)
        
    y2 = LargeKernel(x,num_f,5)
    y2 = BatchNormalization(epsilon=1e-5)(y2)
    y2 = Activation('relu')(y2)
    
    y3 = LargeKernel(x,num_f,7)
    y3 = BatchNormalization(epsilon=1e-5)(y3)
    y3 = Activation('relu')(y3)
    
    y4 = LargeKernel(x,num_f,9)
    y4 = BatchNormalization(epsilon=1e-5)(y4)
    y4 = Activation('relu')(y4)
    
    y5 = LargeKernel(x,num_f,11)
    y5 = BatchNormalization(epsilon=1e-5)(y5)
    y5 = Activation('relu')(y5)

    y = Concatenate()([y0,y1,y2,y3,y4,y5])
    y = CA_Block(y)

    return y 



  
def AMI_Module(x1,x2,num_f):
    
    ''' Creates the AMI Module"
    Args:
        x1, x2: input tensors
        num_f: number of filters
    Returns: 
        a keras tensor''' 
        
    y = Concatenate()([x1,x2])
    y = CA_Block(y)
    y = Conv2D(num_f, (3, 3), padding='same', strides=(1,1))(y)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)

    return y




class BilinearUpsampling(Layer):
    
    """Bilinear upsampling layer
       Args:
           upsampling: the upsampling factors for rows and columns.
           output_size: use this arg instead of upsampling arg if
           your desired size is not an integer factor of the input size"""

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))