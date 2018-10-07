import numpy as np
import tensorflow as tf
import scipy.io

import pdb

FLAGS = tf.app.flags.FLAGS

import myParams

import srez_model

def is_empty(any_structure):
    try:
        if any_structure:
            # print('Structure is not empty.')
            return False
        else:
            # print('Structure is empty.')
            return True
    except:
        return False

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def add_PutInOutput(self,NewOut):
        """ggg: Puts as last output. """
        
        # assert len(self.get_output().get_shape()) == 5, "add_Mult2DMCxSharedOverFeat: Previous layer must be 5-dimensional (batch, H, W, Features,C)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out=NewOut

        self.outputs.append(out)
        return self

    def print_shape(self,message=""):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            IShape=self.get_output().get_shape()
            Len=len(IShape)

            out=self.get_output()
            out = tf.Print(out,[],message=message + ' shape: '+str(IShape))

            print(message + ' shape: ' + str(IShape))

        self.outputs.append(out)
        return self

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        
        return '%s_L%03d' % (self.name, layer+1)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    # ggg
    def _glorot_initializer_g(self, units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(np.prod(units)))
        return tf.truncated_normal(units,mean=0.0, stddev=stddev)

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    # ggg
    def add_dropout(self, keep_prob=0.5):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.dropout(self.get_output(),keep_prob=keep_prob)

        self.outputs.append(out)
        return self

    def add_clip_by_norm(self, clip_norm=3):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.clip_by_norm(self.get_output(),clip_norm=clip_norm)

        self.outputs.append(out)
        return self

    # ggg
    def add_reshapeTo4D(self, height, width):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, height, width, -1])

        self.outputs.append(out)
        return self

    # ggg
    def add_CombineWandFeatures(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            height = int(self.get_output().get_shape()[1])
            width = int(self.get_output().get_shape()[2])
            nFeatures = int(self.get_output().get_shape()[3])
            out = tf.reshape(self.get_output(), [batch_size, height, width*nFeatures, -1])

        self.outputs.append(out)
        return self

    def add_Mult2DMCx(self, num_units, numOutChannels, stddev_factor=1.0):
        """ggg: Fully connected on X (Height) """
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, Features)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            
            init   = self._glorot_initializer(H*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            weight  = tf.get_variable('M2D_MC_weight', initializer=init)
            
            InputToThisLayer=self.get_output()
            InputToThisLayer = tf.transpose(InputToThisLayer, perm=[0,2,1,3])
            InputReshaped=tf.reshape(InputToThisLayer, [batch_size*W,H*channels])

            OutBeforeReshape=tf.matmul(InputReshaped,weight)
            out=tf.reshape(OutBeforeReshape, [batch_size,W,num_units,numOutChannels])

            # Bias term
            initb   = tf.constant(0.0, shape=[1,1,num_units,numOutChannels])
            bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

            # Output of this layer
            out     = out + bias
            out = tf.transpose(out, perm=[0,2,1,3])

        self.outputs.append(out)
        return self

    def add_Mult2DMCy(self, num_units, numOutChannels, stddev_factor=1.0):
        """ggg: Fully connected on Y (Width) """
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, Features)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            
            init   = self._glorot_initializer(W*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            weight  = tf.get_variable('M2D_MC_weight', initializer=init)
            
            InputToThisLayer=self.get_output()
            InputReshaped=tf.reshape(InputToThisLayer, [batch_size*H,W*channels])

            OutBeforeReshape=tf.matmul(InputReshaped,weight)
            out=tf.reshape(OutBeforeReshape, [batch_size,H,num_units,numOutChannels])

            # Bias term
            initb   = tf.constant(0.0, shape=[1,1,num_units,numOutChannels])
            bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

            # Output of this layer
            out     = out + bias

        self.outputs.append(out)
        return self

    def add_5thDim(self):
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, Features)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            
            out=self.get_output()
            out=tf.reshape(out,[batch_size,H,W,channels,1])

        self.outputs.append(out)
        return self

    def add_Split4thDim(self,num_units):
        assert len(self.get_output().get_shape()) >= 4, "Previous layer must be at least 4-dimensional (batch, H, W, Features,...)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            if(len(self.get_output().get_shape())>4):
                Cchannels = int(self.get_output().get_shape()[4])

                out=self.get_output()
                out=tf.reshape(out,[batch_size,H,W,num_units,int(channels/num_units),Cchannels])
                out=tf.transpose(out, perm=[0,1,2,4,3,5])

            else:
                out=self.get_output()
                out=tf.reshape(out,[batch_size,H,W,num_units,int(channels/num_units)])
                out=tf.transpose(out, perm=[0,1,2,4,3])

        self.outputs.append(out)
        return self

    def print_size(self,Message):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out=self.get_output()
            Shape = self.get_output().get_shape()
            print(Message + 'Shape: ' + str(Shape))
            out = tf.Print(out,[Shape],message='Shape:'+str(Shape)+' ' + Message)

        self.outputs.append(out)
        return self
        
    def remove_5thDim(self):
        assert len(self.get_output().get_shape()) == 5, "remove_5thDim: Previous layer must be 5-dimensional (batch, H, W, Dim4,Dim5)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            
            out=self.get_output()
            out=tf.reshape(out,[batch_size,H,W,channels*Cchannels])

        self.outputs.append(out)
        return self
    
    def add_Reshape(self,shp):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            out=self.get_output()
            out=tf.reshape(out, shp)

        self.outputs.append(out)
        return self

    def add_Permute(self,perm):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            out=self.get_output()
            out=tf.transpose(out, perm=perm)

        self.outputs.append(out)
        return self

    def add_Permute34(self):
        assert len(self.get_output().get_shape()) == 5, "add_Permute45: Previous layer must be 5-dimensional (batch, H, W, Dim4,Dim5)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out=self.get_output()
            if(len(self.get_output().get_shape())==4):
                out=tf.transpose(out, perm=[0,1,3,2])
            if(len(self.get_output().get_shape())==5):
                out=tf.transpose(out, perm=[0,1,3,2,4])
            if(len(self.get_output().get_shape())==6):
                out=tf.transpose(out, perm=[0,1,3,2,4,5])

        self.outputs.append(out)
        return self

    def add_Permute45(self):
        assert len(self.get_output().get_shape()) == 5, "add_Permute45: Previous layer must be 5-dimensional (batch, H, W, Dim4,Dim5)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            out=self.get_output()
            out=tf.transpose(out, perm=[0,1,2,4,3])

        self.outputs.append(out)
        return self


    def add_Combine34(self,squeeze=False):
        assert len(self.get_output().get_shape()) >= 5, "add_Combine34: Previous layer must be at least 5-dimensional (batch, H, W, Dim4,Dim5,...)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            if(len(self.get_output().get_shape())>5):
                Dim6 = int(self.get_output().get_shape()[5])
                out=self.get_output()
                if squeeze:
                    out=tf.reshape(out,[batch_size,H,W*channels,Cchannels,Dim6])
                else:
                    out=tf.reshape(out,[batch_size,H,W*channels,1,Cchannels,Dim6])
            else:
                out=self.get_output()
                if squeeze:
                    out=tf.reshape(out,[batch_size,H,W*channels,Cchannels])
                else:
                    out=tf.reshape(out,[batch_size,H,W*channels,1,Cchannels])
            
        self.outputs.append(out)
        return self

    def add_Combine45(self,squeeze=False):
        assert len(self.get_output().get_shape()) >= 5, "add_Combine45: Previous layer must be at least 5-dimensional (batch, H, W, Dim4,Dim5,...)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            if(len(self.get_output().get_shape())>5):
                Dim6 = int(self.get_output().get_shape()[5])
                out=self.get_output()
                if squeeze:
                    out=tf.reshape(out,[batch_size,H,W,channels*Cchannels,Dim6])
                else:
                    out=tf.reshape(out,[batch_size,H,W,channels*Cchannels,1,Dim6])
            else:
                out=self.get_output()
                if squeeze:
                    out=tf.reshape(out,[batch_size,H,W,channels*Cchannels])
                else:
                    out=tf.reshape(out,[batch_size,H,W,channels*Cchannels,1])
            
        self.outputs.append(out)
        return self

    def add_einsumC(self, equation, Wshape, stddev_factor=1.0):
        """ggg: einsum with Complex """
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            IShape=self.get_output().get_shape()

            Len=len(IShape)

            

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            if Len==5:
                assert Cchannels==2, "add_einsumC: Only 2 channel here at 5th dim"
            else:
                Dim6 = int(self.get_output().get_shape()[5])
                assert Dim6==2, "add_einsumC: Only 2 channel here at 6th dim"
                
            
            initR   = self._glorot_initializer_g(Wshape, stddev_factor=stddev_factor)
            initI   = self._glorot_initializer_g(Wshape, stddev_factor=stddev_factor)
            weightR  = tf.get_variable('einsum_weightR', initializer=initR)
            weightI  = tf.get_variable('einsum_weightI', initializer=initI)
            
            InputToThisLayer=self.get_output()

            # InputToThisLayer = tf.Print(InputToThisLayer,[],message='add_einsumC:'+str(InputToThisLayer.get_shape()))

            if Len==5:
                IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
                IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])
            else:
                IReal=tf.slice(InputToThisLayer,[0,0,0,0,0,0],[-1,-1,-1,-1,-1,1])
                IImag=tf.slice(InputToThisLayer,[0,0,0,0,0,1],[-1,-1,-1,-1,-1,1])

            IReal=tf.reshape(IReal,IShape[:-1])
            IImag=tf.reshape(IImag,IShape[:-1])

            print('add_einsumC start:')
            print('------------')
            print('IShape: ' + str(IShape))

            OurRR=tf.einsum(equation,IReal,weightR)
            OurRI=tf.einsum(equation,IReal,weightI)
            OurIR=tf.einsum(equation,IImag,weightR)
            OurII=tf.einsum(equation,IImag,weightI)

            OutR=tf.subtract(OurRR,OurII)
            OutI=tf.add(OurRI,OurIR)

            OutXShape=OutR.get_shape()

            print('OutXShape: ' + str(OutXShape))

            OutXShapep1=np.hstack((OutXShape,1))

            print('OutXShapep1: ' + str(OutXShapep1))

            OutR=tf.reshape(OutR,OutXShapep1)
            OutI=tf.reshape(OutI,OutXShapep1)


            out=tf.concat([OutR, OutI], Len-1)

            # out = tf.Print(out,[],message='add_einsumC:'+str(out.get_shape()))

            """if add_bias:
                # Bias term
                initb   = tf.constant(0.0, shape=[1,1,num_units,numOutChannels,2])
                bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

                # Output of this layer
                out     = out + bias"""

            print('add_einsumC End')

        self.outputs.append(out)
        return self

    def add_conv2dC(self, num_units, mapsize=1, stride=1, stddev_factor=1.0,padding='SAME',add_bias=False,NamePrefix=''):
        """ggg: Adds a 2D convolutional layer - complex."""

        assert len(self.get_output().get_shape()) == 5 and "add_conv2dC: Previous layer must be 5-dimensional (batch, width, height, channels,C)"
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            assert Cchannels==2, "Only 2 channel here at 5th dim"
            
            # Weight term and convolution
            initR  = self._glorot_initializer_conv2d(channels, num_units,mapsize,stddev_factor=stddev_factor)
            initI  = self._glorot_initializer_conv2d(channels, num_units,mapsize,stddev_factor=stddev_factor)

            NameR = '%sC2D_weightR' % (NamePrefix)
            NameI= '%sC2D_weightI' % (NamePrefix)
            weightR = tf.get_variable(NameR, initializer=initR)
            weightI = tf.get_variable(NameI, initializer=initI)

            InputToThisLayer=self.get_output()

            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            IReal=tf.reshape(IReal,[batch_size,H,W,channels])
            IImag=tf.reshape(IImag,[batch_size,H,W,channels])

            outRR    = tf.nn.conv2d(IReal, weightR,strides=[1, stride, stride, 1],padding=padding)
            outRI    = tf.nn.conv2d(IReal, weightI,strides=[1, stride, stride, 1],padding=padding)
            outIR    = tf.nn.conv2d(IImag, weightR,strides=[1, stride, stride, 1],padding=padding)
            outII    = tf.nn.conv2d(IImag, weightI,strides=[1, stride, stride, 1],padding=padding)

            outR=tf.subtract(outRR,outII)
            outI=tf.add(outRI,outIR)

            if add_bias:
                # Bias term
                initbR  = tf.constant(0.0, shape=[num_units])
                initbI  = tf.constant(0.0, shape=[num_units])
                
                biasR   = tf.get_variable('C2D_biasR', initializer=initbR)
                biasI   = tf.get_variable('C2D_biasI', initializer=initbI)
                outR    = tf.nn.bias_add(outR, biasR)
                outI    = tf.nn.bias_add(outI, biasI)
            
            Hnew=int(outR.get_shape()[1])
            Wnew=int(outR.get_shape()[2])

            outR=tf.reshape(outR,[batch_size,Hnew,Wnew,num_units,1])
            outI=tf.reshape(outI,[batch_size,Hnew,Wnew,num_units,1])

            out=tf.concat([outR, outI], 4)


        self.outputs.append(out)
        return self


    def add_Mult2DMCyC(self, num_units, numOutChannels, stddev_factor=1.0, add_bias=False):
        """ggg: Fully connected on Y (Width): Complex """
        
        assert len(self.get_output().get_shape()) == 5, "add_Mult2DMCyC: Previous layer must be 5-dimensional (batch, H, W, Features,2)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])

            # print("AAAAAAAAAA %d %d" % (channels, Cchannels))

            assert Cchannels==2, "Only 2 channel here at 5th dim"
            
            initR   = self._glorot_initializer(W*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            initI   = self._glorot_initializer(W*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            weightR  = tf.get_variable('M2D_MC_weightR', initializer=initR)
            weightI  = tf.get_variable('M2D_MC_weightI', initializer=initI)
            
            InputToThisLayer=self.get_output()
            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            IRealReshaped=tf.reshape(IReal, [batch_size*H,W*channels])
            IImagReshaped=tf.reshape(IImag, [batch_size*H,W*channels])

            OutBeforeReshapeRR=tf.matmul(IRealReshaped,weightR)
            OutBeforeReshapeRI=tf.matmul(IRealReshaped,weightI)
            OutBeforeReshapeIR=tf.matmul(IImagReshaped,weightR)
            OutBeforeReshapeII=tf.matmul(IImagReshaped,weightI)

            OutBeforeReshapeR=tf.subtract(OutBeforeReshapeRR,OutBeforeReshapeII)
            OutBeforeReshapeI=tf.add(OutBeforeReshapeRI,OutBeforeReshapeIR)

            outR=tf.reshape(OutBeforeReshapeR, [batch_size,H,num_units,numOutChannels,1])
            outI=tf.reshape(OutBeforeReshapeI, [batch_size,H,num_units,numOutChannels,1])
            out=tf.concat([outR, outI], 4)

            if add_bias:
                # Bias term
                initb   = tf.constant(0.0, shape=[1,1,num_units,numOutChannels,2])
                bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

                # Output of this layer
                out     = out + bias

        self.outputs.append(out)
        return self

    def add_Mult2DMCyCSharedOverFeat(self, num_units, numOutChannels, stddev_factor=1.0, add_bias=False,Trainable=True,InitC=[],NamePrefix=''):
        """ggg: Fully connected on Y (Width): Complex """
        
        assert len(self.get_output().get_shape()) == 5, "add_Mult2DMCyCSharedOverFeat: Previous layer must be 5-dimensional (batch, H, W, Features,2)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            assert Cchannels==2, "Only 2 channel here at 5th dim"
            
            NameR = '%sadd_Mult2DMCyCSharedOverFeat_weightR' % (NamePrefix)
            NameI = '%sadd_Mult2DMCyCSharedOverFeat_weightI' % (NamePrefix)

            if Trainable:
                if is_empty(InitC):
                    initR   = self._glorot_initializer(W, num_units*numOutChannels, stddev_factor=stddev_factor)
                    initI   = self._glorot_initializer(W, num_units*numOutChannels, stddev_factor=stddev_factor)
                else:
                    initR=np.float32(np.real(InitC))
                    initI=np.float32(np.imag(InitC))
            
                weightR  = tf.get_variable(NameR, initializer=initR)
                weightI  = tf.get_variable(NameI, initializer=initI)
                
            else:
                weightR = tf.constant(np.float32(np.real(InitC)),name=NameR)
                weightI = tf.constant(np.float32(np.imag(InitC)),name=NameI)

            # initR   = self._glorot_initializer(W, num_units*numOutChannels, stddev_factor=stddev_factor)
            # initI   = self._glorot_initializer(W, num_units*numOutChannels, stddev_factor=stddev_factor)
            # weightR  = tf.get_variable('M2D_MC_weightR', initializer=initR)
            # weightI  = tf.get_variable('M2D_MC_weightI', initializer=initI)

            InputToThisLayer=self.get_output()

            # InputToThisLayer = tf.Print(InputToThisLayer,[],message='add_Mult2DMCyCSharedOverFeat:'+str(InputToThisLayer.get_shape()))


            InputToThisLayer = tf.transpose(InputToThisLayer, perm=[0,1,3,2,4]) # now batch, H, Features, W ,2
            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            IRealReshaped=tf.reshape(IReal, [batch_size*H*channels,W])
            IImagReshaped=tf.reshape(IImag, [batch_size*H*channels,W])

            OutBeforeReshapeRR=tf.matmul(IRealReshaped,weightR)
            OutBeforeReshapeRI=tf.matmul(IRealReshaped,weightI)
            OutBeforeReshapeIR=tf.matmul(IImagReshaped,weightR)
            OutBeforeReshapeII=tf.matmul(IImagReshaped,weightI)

            OutBeforeReshapeR=tf.subtract(OutBeforeReshapeRR,OutBeforeReshapeII)
            OutBeforeReshapeI=tf.add(OutBeforeReshapeRI,OutBeforeReshapeIR)

            outR=tf.reshape(OutBeforeReshapeR, [batch_size,H,channels,num_units,numOutChannels,1])
            outI=tf.reshape(OutBeforeReshapeI, [batch_size,H,channels,num_units,numOutChannels,1])

            out=tf.concat([outR, outI], 5)
            out = tf.transpose(out, perm=[0,1,3,2,4,5])
            out=tf.reshape(out, [batch_size,H,num_units,channels*numOutChannels,2])

            if add_bias:
                # Bias term
                initb   = tf.constant(0.0, shape=[1,1,num_units,channels*numOutChannels,2])
                bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

                # Output of this layer
                out     = out + bias

        self.outputs.append(out)
        return self

    def add_Mult2DMCxCSharedOverFeat(self, num_units, numOutChannels, stddev_factor=1.0,add_bias=False,Trainable=True,InitC=[],NamePrefix=''):
        """ggg: Fully connected on X (Height): Complex """
        
        assert len(self.get_output().get_shape()) == 5, "add_Mult2DMCxCSharedOverFeat: Previous layer must be 5-dimensional (batch, H, W, Features,2)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            assert Cchannels==2, "Only 2 channel here at 5th dim"

            NameR = '%sadd_Mult2DMCxCSharedOverFeat_weightR' % (NamePrefix)
            NameI = '%sadd_Mult2DMCxCSharedOverFeat_weightI' % (NamePrefix)
                        
            if Trainable:
                if is_empty(InitC):
                    initR   = self._glorot_initializer(H, num_units*numOutChannels, stddev_factor=stddev_factor)
                    initI   = self._glorot_initializer(H, num_units*numOutChannels, stddev_factor=stddev_factor)
                else:
                    initR=np.float32(np.real(InitC))
                    initI=np.float32(np.imag(InitC))

                weightR  = tf.get_variable(NameR, initializer=initR)
                weightI  = tf.get_variable(NameI, initializer=initI)
                
            else:
                weightR = tf.constant(np.float32(np.real(InitC)),name=NameR)
                weightI = tf.constant(np.float32(np.imag(InitC)),name=NameI)

            InputToThisLayer=self.get_output()
            InputToThisLayer = tf.transpose(InputToThisLayer, perm=[0,2,3,1,4]) # now batch, W, Features, H ,2

            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            IRealReshaped=tf.reshape(IReal, [batch_size*W*channels,H])
            IImagReshaped=tf.reshape(IImag, [batch_size*W*channels,H])

            OutBeforeReshapeRR=tf.matmul(IRealReshaped,weightR)
            OutBeforeReshapeRI=tf.matmul(IRealReshaped,weightI)
            OutBeforeReshapeIR=tf.matmul(IImagReshaped,weightR)
            OutBeforeReshapeII=tf.matmul(IImagReshaped,weightI)

            OutBeforeReshapeR=tf.subtract(OutBeforeReshapeRR,OutBeforeReshapeII)
            OutBeforeReshapeI=tf.add(OutBeforeReshapeRI,OutBeforeReshapeIR)

            outR=tf.reshape(OutBeforeReshapeR, [batch_size,W,channels,num_units,numOutChannels,1])
            outI=tf.reshape(OutBeforeReshapeI, [batch_size,W,channels,num_units,numOutChannels,1])
            out=tf.concat([outR, outI], 5)

            out = tf.transpose(out, perm=[0,3,1,2,4,5])
            out=tf.reshape(out, [batch_size,num_units,W,channels*numOutChannels,2])

            if add_bias:
                # Bias term
                initb   = tf.constant(0.0, shape=[1,num_units,1,channels*numOutChannels,2])
                bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

                # Output of this layer
                out     = out + bias

        self.outputs.append(out)
        return self

    def add_Mult2DMCxC(self, num_units, numOutChannels, stddev_factor=1.0,add_bias=False):
        """ggg: Fully connected on X (Height): Complex """
        
        assert len(self.get_output().get_shape()) == 5, "add_Mult2DMCxC: Previous layer must be 5-dimensional (batch, H, W, Features,2)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            assert Cchannels==2, "Only 2 channel here at 5th dim"
            
            initR   = self._glorot_initializer(H*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            initI   = self._glorot_initializer(H*channels, num_units*numOutChannels, stddev_factor=stddev_factor)
            weightR  = tf.get_variable('M2D_MC_weightR', initializer=initR)
            weightI  = tf.get_variable('M2D_MC_weightI', initializer=initI)
            
            InputToThisLayer=self.get_output()
            InputToThisLayer = tf.transpose(InputToThisLayer, perm=[0,2,1,3,4])

            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            IRealReshaped=tf.reshape(IReal, [batch_size*W,H*channels])
            IImagReshaped=tf.reshape(IImag, [batch_size*W,H*channels])

            OutBeforeReshapeRR=tf.matmul(IRealReshaped,weightR)
            OutBeforeReshapeRI=tf.matmul(IRealReshaped,weightI)
            OutBeforeReshapeIR=tf.matmul(IImagReshaped,weightR)
            OutBeforeReshapeII=tf.matmul(IImagReshaped,weightI)

            OutBeforeReshapeR=tf.subtract(OutBeforeReshapeRR,OutBeforeReshapeII)
            OutBeforeReshapeI=tf.add(OutBeforeReshapeRI,OutBeforeReshapeIR)

            outR=tf.reshape(OutBeforeReshapeR, [batch_size,W,num_units,numOutChannels,1])
            outI=tf.reshape(OutBeforeReshapeI, [batch_size,W,num_units,numOutChannels,1])
            out=tf.concat([outR, outI], 4)

            out = tf.transpose(out, perm=[0,2,1,3,4])

            if add_bias:
                # Bias term
                initb   = tf.constant(0.0, shape=[1,num_units,1,numOutChannels,2])
                bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

                # Output of this layer
                out     = out + bias

        self.outputs.append(out)
        return self

    def add_PixelwiseMultC(self, numOutChannels, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=[]):
        """ggg: add dense 1x1 data->image (flattened), same matrix on all channels.
        Multiply to numOutChannels features (supposedly real and imag and others?) """
        
        #print("add add_PixelwiseMultC %d %d %d %d" % (int(self.get_output().get_shape()[0]),int(self.get_output().get_shape()[1]),int(self.get_output().get_shape()[2]),int(self.get_output().get_shape()[3])))
        assert len(self.get_output().get_shape()) == 5, "add_PixelwiseMultC: Previous layer must be 5-dimensional (batch, H, W, featIn,2)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            Cchannels = int(self.get_output().get_shape()[4])
            assert Cchannels==2, "Only 2 channel here at 5th dim"
            
            NameR = '%sPixelwiseMultC_weightR' % (NamePrefix)
            NameI= '%sPixelwiseMultC_weightI' % (NamePrefix)

            if Trainable:
                if is_empty(InitC):
                    initR   = self._glorot_initializer(H*W, channels*numOutChannels, stddev_factor=stddev_factor)
                    initI   = self._glorot_initializer(H*W, channels*numOutChannels, stddev_factor=stddev_factor)
                else:
                    initR = np.float32(np.real(InitC))
                    initI = np.float32(np.imag(InitC))

                weightR = tf.get_variable(NameR, initializer=initR)
                weightI = tf.get_variable(NameI, initializer=initI)
                
            else:
                weightR = tf.constant(np.float32(np.real(InitC)),name=NameR)
                weightI = tf.constant(np.float32(np.imag(InitC)),name=NameI)

            # initR   = self._glorot_initializer(H*W, channels*numOutChannels, stddev_factor=stddev_factor)
            # initI   = self._glorot_initializer(H*W, channels*numOutChannels, stddev_factor=stddev_factor)
            # # weightR  = tf.get_variable('PixelswiseMult_weightR', initializer=initR)
            # # weightI  = tf.get_variable('PixelswiseMult_weightI', initializer=initI)
            # weightR  = tf.get_variable(NameR, initializer=initR)
            # weightI  = tf.get_variable(NameI, initializer=initI)
            
            InputToThisLayer=self.get_output()

            IReal=tf.slice(InputToThisLayer,[0,0,0,0,0],[-1,-1,-1,-1,1])
            IImag=tf.slice(InputToThisLayer,[0,0,0,0,1],[-1,-1,-1,-1,1])

            InputToThisLayerRshpR=tf.reshape(IReal,[batch_size,H,W,channels,1])
            InputToThisLayerRshpI=tf.reshape(IImag,[batch_size,H,W,channels,1])
            
            weightRshpR=tf.reshape(weightR,[1,H,W,channels,numOutChannels])
            weightRshpI=tf.reshape(weightI,[1,H,W,channels,numOutChannels])
            
            outMulRR=tf.multiply(InputToThisLayerRshpR,weightRshpR)
            outMulRI=tf.multiply(InputToThisLayerRshpR,weightRshpI)
            outMulIR=tf.multiply(InputToThisLayerRshpI,weightRshpR)
            outMulII=tf.multiply(InputToThisLayerRshpI,weightRshpI)

            outMulR=tf.subtract(outMulRR,outMulII)
            outMulI=tf.add(outMulRI,outMulIR)

            outR=tf.reduce_sum(outMulR, 3)
            outI=tf.reduce_sum(outMulI, 3)

            outR=tf.reshape(outR,[batch_size,H,W,numOutChannels,1])
            outI=tf.reshape(outI,[batch_size,H,W,numOutChannels,1])

            out=tf.concat([outR, outI], 4)

            # Bias term
            # initb   = tf.constant(0.0, shape=[1,H,W,numOutChannels,2])
            # bias    = tf.get_variable('PixelswiseMult_bias', initializer=initb)

            # # Output of this layer
            # out     = out + bias

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult2D(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])

            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(H, W,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('M2D_weight', initializer=initw)

            # Bias term
            #initb   = tf.constant(0.0, shape=[num_units])
            #bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            #out     = tf.matmul(self.get_output(), weight) + bias

            weight = tf.reshape(weight, [1,H,W,1])
            weight = tf.tile(weight,[batch_size,1,1,channels])
            out    = tf.multiply(self.get_output(), weight)

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult2DComplexRI(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            HalfChannels = int(channels/2)

            prev_units = self._get_num_inputs()
            
            initr   = self._glorot_initializer(H, W, stddev_factor=stddev_factor)
            initi   = self._glorot_initializer(H, W, stddev_factor=stddev_factor)
            weightr  = tf.get_variable('M2D_CRIR_weight', initializer=initr)
            weighti  = tf.get_variable('M2D_CRII_weight', initializer=initi)

            weightr = tf.reshape(weightr, [1,H,W,1])
            weighti = tf.reshape(weighti, [1,H,W,1])
            
            weightr = tf.tile(weightr,[batch_size,1,1,channels])
            weighti = tf.tile(weighti,[batch_size,1,1,channels])

            a = np.array([np.arange(2,channels+1,2),np.arange(1,channels,2)])
            Idxs=a.transpose().flatten()

            IdxsT = tf.constant(Idxs, shape=[channels])

            PlusMinus = tf.constant([1.0, -1.0], shape=[1,1,1,2])
            PlusMinus = tf.tile(PlusMinus,[batch_size,H,W,HalfChannels])

            A   = tf.multiply(self.get_output(), weightr)
            B   = tf.multiply(self.get_output(), weighti)

            B=tf.gather(B,IdxsT,validate_indices=None,name=None,axis=3)
            B=tf.multiply(B, PlusMinus)

            out= A + B

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult3DComplexRI(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            HalfChannels = int(channels/2)

            prev_units = self._get_num_inputs()
            
            initr   = self._glorot_initializer(H, W*HalfChannels, stddev_factor=stddev_factor)
            initi   = self._glorot_initializer(H, W*HalfChannels, stddev_factor=stddev_factor)
            weightr  = tf.get_variable('M3D_CRIR_weight', initializer=initr)
            weighti  = tf.get_variable('M3D_CRII_weight', initializer=initi)

            weightr = tf.reshape(weightr, [1,H,W,HalfChannels])
            weighti = tf.reshape(weighti, [1,H,W,HalfChannels])
            
            weightr = tf.tile(weightr,[batch_size,1,1,1])
            weighti = tf.tile(weighti,[batch_size,1,1,1])

            Idxs1122=np.int32(np.kron(np.arange(0,HalfChannels),np.ones(2)))
            Idxs1122T = tf.constant(Idxs1122, shape=[channels])

            weightr=tf.gather(weightr,Idxs1122T,validate_indices=None,name=None,axis=3)            
            weighti=tf.gather(weighti,Idxs1122T,validate_indices=None,name=None,axis=3)            
            
            A   = tf.multiply(self.get_output(), weightr)
            B   = tf.multiply(self.get_output(), weighti)

            PlusMinus = tf.constant([-1.0, 1.0], shape=[1,1,1,2])
            PlusMinus = tf.tile(PlusMinus,[batch_size,H,W,HalfChannels])

            Idxs = np.array([np.arange(1,channels,2),np.arange(0,channels-1,2)])
            Idxs = Idxs.transpose().flatten()
            IdxsT = tf.constant(Idxs, shape=[channels])
            
            B=tf.gather(B,IdxsT,validate_indices=None,name=None,axis=3)
            B=tf.multiply(B, PlusMinus)

            out= A + B

        self.outputs.append(out)
        return self

    def add_Mult2DMC(self, num_units, numOutChannels, stddev_factor=1.0):
        """ggg: add dense matrix data->image (flattened), same matrix on all channels.
        Multiply to numOutChannels features (supposedly real and imag and others?) """
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, 1)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            assert channels==1, "Only 1 channel here at 4th dim"

            init   = self._glorot_initializer(W, num_units*numOutChannels, stddev_factor=stddev_factor)
            weight  = tf.get_variable('M2D_MC_weight', initializer=init)
            
            InputToThisLayer=self.get_output()
            InputReshaped=tf.reshape(InputToThisLayer, [batch_size*H,W])

            OutBeforeReshape=tf.matmul(InputReshaped,weight)
            out=tf.reshape(OutBeforeReshape, [batch_size,H,num_units,numOutChannels])

            # Bias term
            initb   = tf.constant(0.0, shape=[1,H,num_units,numOutChannels])
            bias    = tf.get_variable('M2D_MC_bias', initializer=initb)

            # Output of this layer
            out     = out + bias

            OutP = tf.transpose(out, perm=[0, 2,1, 3])
            OutPR = tf.reshape(OutP, [batch_size,num_units,1,H*numOutChannels])

        self.outputs.append(OutPR)
        return self

    def add_PixelwiseMult(self, numOutChannels, stddev_factor=1.0):
        """ggg: add dense 1x1 data->image (flattened), same matrix on all channels.
        Multiply to numOutChannels features (supposedly real and imag and others?) """
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, featIn)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            
            init   = self._glorot_initializer(H*W, channels*numOutChannels, stddev_factor=stddev_factor)
            weight  = tf.get_variable('PixelwiseMult_weight', initializer=init)
            
            InputToThisLayer=self.get_output()
            InputToThisLayerR=tf.reshape(InputToThisLayer,[batch_size,H,W,channels,1])
            
            weightR=tf.reshape(weight,[1,H,W,channels,numOutChannels])
            
            outMul=tf.multiply(InputToThisLayerR,weightR)
            out=tf.reduce_sum(outMul, 3)

            # Bias term
            initb   = tf.constant(0.0, shape=[1,H,W,numOutChannels])
            bias    = tf.get_variable('PixelswiseMult_bias', initializer=initb)

            # Output of this layer
            out     = out + bias

        self.outputs.append(out)
        return self

    def add_denseFromM(self, name):
        """ggg: Fully-connected layer from given matrix """

        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        ConstS=scipy.io.loadmat('/home/a/TF/Consts.mat')
        #CurM=ConstS[name]
        CurM=ConstS['piMDR']
        ConstMf=np.float32(CurM)

        prev_units = self._get_num_inputs()
        num_units=ConstMf.shape[1]
        
        ConstMtf = tf.constant(ConstMf, shape=[prev_units,num_units])
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            
            
            # Weight term
            weight  = tf.get_variable('DM_weight', initializer=ConstMtf)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('DM_bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('Dweight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('Dbias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    # ggg
    def add_tanh(self):
        """Adds a TanH activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.tanh(self.get_output())

        self.outputs.append(out)
        return self      

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('C2D_weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('C2D_bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2dWithName(self, num_units, name=None, mapsize=1, stride=1, stddev_factor=1.0):
        """ggg: Adds a 2D convolutional layer with name"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('C2D_weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('C2D_bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias,name=name)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('C2DT_weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [myParams.myDict['batch_size'],
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('C2DT_bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_max_pool(self, pool_size=[2,2],strides=2):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            pool = tf.layers.max_pooling2d(inputs=self.get_output(), pool_size=pool_size, strides=strides)

        self.outputs.append(pool)
        return self

    # ggg
    def add_Unet1Step(self, num_units, mapsize=3, stride=2, num_layers=2, stddev_factor=1e-3):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        self.add_conv2d(num_units, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        self.add_relu()

        self.add_conv2d_transpose(num_units, mapsize=mapsize, stride=stride, stddev_factor=1.)
        self.add_relu()


        # Residual block
        #for _ in range(num_layers):
        #   self.add_batch_norm()
        #  self.add_relu()
        
        #out = tf.concat( [bypass, self.get_output()], 3)
        #self.outputs.append(out)

        self.add_sum(bypass)

        return self

    def add_UnetKsteps(self, num_units, mapsize=3, stride=2, stddev_factor=1e-3):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        nLines=num_units.shape[0]
        CurLine=num_units[0]
        BeforeD=CurLine[0:3]
        AfterD=CurLine[5:8]


        for i in BeforeD:
            if i>0:
                print("add conv2D+relu %d" % (i))
                self.add_conv2d(i, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
                self.add_elu()

        # call recursive
        if nLines>1:
            bypass = self.get_output()
            
            # go down
            self.add_conv2d(CurLine[3], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)
            self.add_elu()
            print("add conv2D+relu %d stride %d" % (CurLine[3],stride))
            # or alternatively add_max_pool(self, pool_size=[2,2],strides=2):

            self.add_UnetKsteps(num_units=num_units[1:], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)

            # go up
            self.add_conv2d_transpose(CurLine[4], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)
            self.add_elu()
            print("add conv2DT+relu %d stride %d" % (CurLine[3],stride))

            # concat
            out = tf.concat( [bypass, self.get_output()], 3)
            self.outputs.append(out)

        for i in AfterD:
            if i>0:
                print("add conv2D+relu %d" % (i))
                self.add_conv2d(i, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
                self.add_elu()
        
        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
            #out = tf.reduce_mean(self.get_output(), axis=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

    # ggg
    def add_constMatMul(self):
        ConstS=scipy.io.loadmat('/home/a/TF/AdjMRB.mat')
        ConstM=ConstS['AdjMRB']
        nChosen=FLAGS.nChosen*2 # ConstS['nChosen']
        ConstMf=np.float32(ConstM)
        #ConstMf=np.repeat(ConstMf, 16, axis=0)
        #ConstM = np.array(np.random.choice([0, 1], size=(1,64,64,3)), dtype='float32')
        ConstMtf1 = tf.constant(ConstMf, shape=[1,1,nChosen,64*64*2])
        ConstMtf=   tf.tile(ConstMtf1,[16,1,1,1])
        #ConstMtf = tf.constant(ConstMf, shape=[16,1,64*64*2,64*64*2])

        #W_filtered = tf.mul(connectivity, W)
        #out     = tf.multiply(self.get_output(), connectivity) # + bias
        P=tf.transpose(self.get_output(), perm=[0, 1,3,2])
        #out = tf.reshape(P, [16,64,64,2])
        #out = tf.reshape(self.get_output(), [16,64,64,2])
        M=tf.matmul(P,ConstMtf) # now its [Nbatch 1 1 8192] real and then image
        out = tf.reshape(M, [16,2,64,64])
        out=tf.transpose(out, perm=[0, 3,2,1])
        #out=tf.transpose(x, perm=[1, 0])
        self.outputs.append(out)
        return self
        #"""Adds a 2D convolutional layer."""

        #assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"       
        #with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
        #    prev_units = self._get_num_inputs()
        #    
        #    # Weight term and convolution
        #    initw  = self._glorot_initializer_conv2d(prev_units, num_units,
        #                                             mapsize,
        #                                             stddev_factor=stddev_factor)
        #    weight = tf.get_variable('weight', initializer=initw)
        #    out    = tf.nn.conv2d(self.get_output(), weight,
        #                          strides=[1, stride, stride, 1],
        #                          padding='SAME')

        #    # Bias term
        #    initb  = tf.constant(0.0, shape=[num_units])
        #    bias   = tf.get_variable('bias', initializer=initb)
        #    out    = tf.nn.bias_add(out, bias)
            
        #self.outputs.append(out)
        #return self

def _discriminator_model(sess, features, disc_input):
    # Fully convolutional model
    mapsize = 3
    layers  = [64, 128, 256, 512]

    old_vars = tf.global_variables()

    model = Model('DIS', 2*disc_input - 1)

    #RunDiscriminator= FLAGS.gene_l1_factor < 0.999
    RunDiscriminator= FLAGS.gene_l1_factor < 2

    if myParams.myDict['InputMode'] == 'RegridTry3FMB':
        RunDiscriminator=False

    if RunDiscriminator:
        for layer in range(len(layers)):
            nunits = layers[layer]
            stddev_factor = 2.0

            model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
            model.add_batch_norm()
            model.add_relu()

        # Finalization a la "all convolutional net"
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        # Linearly map to real/fake and return average score
        # (softmax will be applied later)
        model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    else:
        stddev_factor = 2.0
        model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
        # model.add_flatten()

    model.add_mean()

    # pdb.set_trace()

    new_vars  = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), disc_vars

def create_model(sess, features, labels):
    # Generator
    rows      = int(features.get_shape()[1])
    cols      = int(features.get_shape()[2])
    channelsIn  = int(features.get_shape()[3])
    channelsOut  = int(labels.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[myParams.myDict['batch_size'], rows, cols, channelsIn])

    # TBD: Is there a better way to instance the generator?
    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list = srez_model._generator_model(sess, features, labels, channelsOut)

        scope.reuse_variables()

        gene_moutput, _ = srez_model._generator_model(sess, gene_minput, labels, channelsOut)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = _discriminator_model(sess, features, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _ = _discriminator_model(sess, features, gene_output)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]

#def _downscale(images, K):
#    """Differentiable image downscaling by a factor of K"""
#    arr = np.zeros([K, K, 3, 3])
#    arr[:,:,0,0] = 1.0/(K*K)
#    arr[:,:,1,1] = 1.0/(K*K)
#    arr[:,:,2,2] = 1.0/(K*K)
#    dowscale_weight = tf.constant(arr, dtype=tf.float32)
#    
#    downscaled = tf.nn.conv2d(images, dowscale_weight,
#                              strides=[1, K, K, 1],
#                              padding='SAME')
#    return downscaled

def create_generator_loss(disc_output, gene_output, features, labels,varsForL1,varsForL2,varsForPhaseOnly):
    # I.e. did we fool the discriminator?
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_output, logits=tf.ones_like(disc_output))
    #gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    ls_loss_real = tf.square(disc_output - tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(ls_loss_real, name='gene_ce_loss')

    WL1_Lambda=myParams.myDict['WL1_Lambda']
    WL2_Lambda=myParams.myDict['WL2_Lambda']

    WPhaseOnly=myParams.myDict['WPhaseOnly']
    
    
    # I.e. does the result look like the feature?
    #K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    #assert K == 2 or K == 4 or K == 8    
    #downscaled = _downscale(gene_output, K)
    #downscaled = gene_output
    
    #gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - features), name='gene_l1_loss')

    # Wl2_loss = tf.add_n([tf.nn.l2_loss(v) for v in varsForL2]) * WL2_Lambda
    if WL1_Lambda==0:
        Wl1_loss=0
    else:
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=WL1_Lambda, scope=None)
        Wl1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, varsForL1)

    if WL2_Lambda==0:
        Wl2_loss=0
    else:
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=WL2_Lambda, scope=None)
        Wl2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, varsForL2)

    if WPhaseOnly==0:
        PhaseOnly_loss=0
    else:
        A=tf.sqrt(tf.add(tf.square(varsForPhaseOnly[0]),tf.square(varsForPhaseOnly[1])))
        B=tf.sqrt(tf.add(tf.square(varsForPhaseOnly[2]),tf.square(varsForPhaseOnly[3])))
        A1=tf.abs(tf.subtract(A,1))
        B1=tf.abs(tf.subtract(B,1))
        PhaseOnly_loss=tf.multiply(WPhaseOnly,tf.reduce_mean(A1) + tf.reduce_mean(B1))

    

    if myParams.myDict['NetMode'] == 'RegridTry1' or myParams.myDict['NetMode'] == 'RegridTry1C':
        # channelsOut=myParams.myDict['channelsOut']
        # LabelsH=myParams.myDict['LabelsH']
        # LabelsW=myParams.myDict['LabelsW']

        # Msk=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/Msk.mat')
        # Msk=Msk['Msk']
        # Msk=np.reshape(Msk,[1,LabelsH, LabelsW, channelsOut])

        # Msk = tf.constant(Msk)

        gene_l1_loss  = tf.reduce_mean(tf.multiply(Msk,tf.abs(gene_output - labels)), name='gene_l1_loss')
    else:
        gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
        # gene_l1_loss  = tf.reduce_mean(tf.square(gene_output - labels), name='gene_l1_loss')

    if myParams.myDict['NetMode'] == 'Conv_v1_ForB0':
        print('Using masked diff!!!')
        # gene_l1_loss  = tf.reduce_mean(tf.multiply(tf.abs(gene_output - labels),labels>0), name='gene_l1_loss')
        B0L=tf.slice(labels,[0,0,0,0],[-1,-1,-1,1])
        B0G=tf.slice(gene_output,[0,0,0,0],[-1,-1,-1,1])

        # IL=tf.slice(labels,[0,0,0,1],[-1,-1,-1,1])
        # IG=tf.slice(gene_output,[0,0,0,1],[-1,-1,-1,1])

        Msk=tf.logical_and(tf.not_equal(B0L,0), tf.abs(B0L)>myParams.myDict['MinAbsB0ForComparison'] )
        gene_l1_loss  = tf.reduce_mean(tf.boolean_mask(tf.abs(B0G - B0L), Msk ), name='gene_l1_lossL')

        # gene_l1_lossI  = tf.reduce_mean(tf.abs(IG - IL), name='gene_l1_lossO')

        # gene_l1_loss  = gene_l1_lossB+gene_l1_lossI


    # X = np.concatenate([np.arange(0,1), np.arange(0,myParams.myDict['LabelsH']-1)])
    # Y = tf.constant(X) #, dtype=tf.int32, shape=[LabelsH])
    # gene_output1x=tf.gather(gene_output,Y,validate_indices=None,name=None,axis=1)
    # gene_output1y=tf.gather(gene_output,Y,validate_indices=None,name=None,axis=2)
    # gene_outputDx=gene_output1x-gene_output
    # gene_outputDy=gene_output1y-gene_output

    # labels1x=tf.gather(labels,Y,validate_indices=None,name=None,axis=1)
    # labels1y=tf.gather(labels,Y,validate_indices=None,name=None,axis=2)
    # labelsDx=labels1x-labels
    # labelsDy=labels1y-labels

    # DLoss=5*tf.reduce_mean(tf.abs(gene_outputDx - labelsDx) + tf.abs(gene_outputDy - labelsDy), name='DLoss')
    
    #gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')


    # gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_ce_loss,FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')
    gene_loss     = tf.identity(gene_l1_loss, name='gene_loss')

    MoreOut3 = gene_loss
    MoreOut  = tf.reduce_mean(gene_output)
    MoreOut2 = tf.reduce_mean(labels)

    # gene_lossWithWL2=gene_loss+Wl2_loss
    gene_lossWithL1L2=gene_loss+Wl1_loss+Wl2_loss+PhaseOnly_loss
    return gene_lossWithL1L2, MoreOut, MoreOut2, MoreOut3
    # return gene_loss

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    #cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_real_output, logits=tf.ones_like(disc_real_output))
    #cross_entropy_real = tf.nn.softmax_cross_entropy_with_logits_v2(labels=disc_real_output, logits=tf.ones_like(disc_real_output))
    #cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output), logits=disc_real_output)
    #disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    #cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_fake_output, logits=tf.zeros_like(disc_fake_output))
    #disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    ls_loss_real = tf.square(disc_real_output - tf.ones_like(disc_real_output))
    disc_real_loss = tf.reduce_mean(ls_loss_real, name='disc_real_loss')

    ls_loss_fake = tf.square(disc_fake_output)
    disc_fake_loss = tf.reduce_mean(ls_loss_fake, name='disc_fake_loss')

    return disc_real_loss, disc_fake_loss

def create_optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='gene_optimizer')
    # gene_opti = tf.train.AdamOptimizer(learning_rate=0.00001,
    #                                    beta1=0.9,
    #                                    beta2=0.999,
    #                                    name='gene_optimizer')
    
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    
    disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)
