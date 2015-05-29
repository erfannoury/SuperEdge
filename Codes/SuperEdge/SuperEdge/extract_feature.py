import theano.tensor as T
import theano
import numpy as np
import caffe_pb2
import os
import cv2
import cPickle
from datetime import datetime
from collections import OrderedDict
from sklearn_theano.base import (Convolution, Relu, MaxPool, FancyMaxPool,
                                 LRN, Feedforward, ZeroPad,
                                 CaffePool)
from sklearn.externals import joblib
from sklearn_theano.utils import check_tensor
from sklearn_theano.feature_extraction.overfeat import get_overfeat_class_label
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import zoom

class VGG16Extractor(object):
    def __init__(self, width=480, height=320, use_mean_pixel=True, *args, **kwargs):
        self.image_width = width
        self.image_height = height
        self.use_mean_pixel = use_mean_pixel
        # BGR ordering
        self.mean_pixel = np.asarray([116.779, 103.939, 123.68], dtype=np.float32)
        self.mean_image = '../../../Models/VGG_CNN_F_mean.pkl'
        ######## Loading mean image for VGG CNN F (BGR ordering) ##########
        f = file(self.mean_image, 'rb')
        self.mean_image = cPickle.load(f)
        f.close()
        self.mean_image = cv2.resize(self.mean_image, (self.image_width, self.image_height))
        ######## Loading mean image for VGG CNN F ##########
#        self.model_address = '../../../Models/VGG_ILSVRC_16_layers.caffemodel'
#        self.model_pickle = '../../../Models/VGG_ILSVRC_16_layers.pkl'
        self.model_address = '../../../Models/VGG_CNN_F.caffemodel'
        self.model_pickle = '../../../Models/VGG_CNN_F.pkl'
        self.model_trans = '../../../Models/Transformation Pickles/'
        self.caffemodel = None
        self.parsedmodel = None
        self.LAYER_PROPERTIES = dict(
                                    DATA=None,
                                    CONVOLUTION=('blobs',
                                                 ('convolution_param', 'stride'),
                                                 ('convolution_param', 'stride_h'),
                                                 ('convolution_param', 'stride_w'),
                                                 ('convolution_param', 'pad'),
                                                 ('convolution_param', 'pad_h'),
                                                 ('convolution_param', 'pad_w')),
                                    RELU=None,
                                    POOLING=(('pooling_param', 'kernel_size'),
                                             ('pooling_param', 'kernel_h'),
                                             ('pooling_param', 'kernel_w'),
                                             ('pooling_param', 'stride'),
                                             ('pooling_param', 'stride_h'),
                                             ('pooling_param', 'stride_w'),
                                             ('pooling_param', 'pad'),
                                             ('pooling_param', 'pad_h'),
                                             ('pooling_param', 'pad_w'),
                                             ('pooling_param', 'pool')
                                             ),
                                    SPLIT=None,
                                    LRN=(('lrn_param', 'local_size'),
                                         ('lrn_param', 'alpha'),
                                         ('lrn_param', 'beta'),
                                         ('lrn_param', 'norm_region')),
                                    CONCAT=(('concat_param', 'concat_dim'),),
                                    INNER_PRODUCT=('blobs',),
                                    SOFTMAX_LOSS=None,
                                    DROPOUT=None,
                                    SOFTMAX=None
                                    )

        print 'loading caffemodel'
        if not os.path.exists(self.model_pickle):
            now = datetime.now()
            self.caffemodel = self._open_caffe_model(self.model_address)
            print 'parsing using protobuf took ', (datetime.now() - now)
            f = file(self.model_pickle, 'wb')
            cPickle.dump(self.caffemodel, f, protocol=2)
            print 'dumped parsed caffemodel to pickle file'
            f.close()
        else:
            now = datetime.now()
            f = file(self.model_pickle, 'rb')
            self.caffemodel = cPickle.load(f)
            print 'loading pickle file took ', (datetime.now() - now)
            f.close()

        print 'parsing caffemodel'
        self._parse_caffe_model()

        

    def _open_caffe_model(self, caffemodel_file):
        """
        Opens binary format .caffemodel files. Returns protobuf object.
        """
        binary_content = open(caffemodel_file, "rb").read()
        protobuf = caffe_pb2.NetParameter()
        protobuf.ParseFromString(binary_content)
        
        return protobuf

    def _blob_to_ndarray(self, blob):
        """
        Converts a caffe protobuf blob into an ndarray
        """
        dimnames = ["num", "channels", "height", "width"]
        data = np.array(blob.data)
        shape = tuple([getattr(blob, dimname) for dimname in dimnames])
        return data.reshape(shape)
    
    def _get_property(self, obj, property_path):
        if isinstance(property_path, tuple):
            if len(property_path) == 1:
                return getattr(obj, property_path[0])
            else:
                return self._get_property(getattr(obj, property_path[0]),
                                     property_path[1:])
        else:
            return getattr(obj, property_path)

    def _parse_caffe_model(self):
        if self.parsedmodel is not None:
            return
        try:
            _layer_types = caffe_pb2.LayerParameter.LayerType.items()
        except AttributeError:
            _layer_types = caffe_pb2.V1LayerParameter.LayerType.items()

        # create a dictionary that indexes both ways, number->name, name->number
        layer_types = dict(_layer_types)
        for v, k in _layer_types:
            layer_types[k] = v

        layers_raw = self.caffemodel.layers
        parsed = []

        # VGG CNN models doesn't have a data layer defined, thus it is added manually
        first_layer_descriptor = dict(type='DATA',
                                      name='data',
                                      top_blobs=('data',),
                                      bottom_blobs=tuple())
        parsed.append(first_layer_descriptor)

        for layer in layers_raw:
            # standard properties
            ltype = layer_types[layer.type]
            layer_descriptor = dict(type=ltype,
                                    name=layer.name,
                                    top_blobs=tuple(layer.top),
                                    bottom_blobs=tuple(layer.bottom))
            parsed.append(layer_descriptor)
            # specific properties
            specifics = self.LAYER_PROPERTIES[ltype]
            if specifics is None:
                continue
            for param in specifics:
                if param == 'blobs':
                    layer_descriptor['blobs'] = map(self._blob_to_ndarray,
                                                    layer.blobs)
                else:
                    param_name = '__'.join(param)
                    param_value = self._get_property(layer, param)
                    layer_descriptor[param_name] = param_value
        self.parsedmodel = parsed
        self.caffemodel = None

    def preprocess_image(self, input_image):
        """
        This will preprocess the input image, swap channels into BGR ordering and subtract the mean pixel value

        Parameters
        ==========
        input_image: numpy.ndarray
            a 3-channel input image with RGB channel ordering and of shape height * width * 3
        Returns
        =======
        image: numpy.ndarray
            a 3-channel image with BGR channel ordering and zero-centered by mean pixel value
        """
        image = input_image.astype(np.float32)
        image = image[:,:,[2,1,0]]
        if self.use_mean_pixel:
            image -= self.mean_pixel
        else:
            image -= self.mean_image
        return image

    def __layerwiseTransform__(self, input_data, float_dtype='float32', verbose=0):
        transformations = {}
        input_data = self.preprocess_image(input_data)
        X = check_tensor(input_data, dtype=np.float32, n_dim=4)
        last_expression = None
        current_expression = None
        # bc01 ordering 
        trans_order = (0, 3, 1, 2)
        X = X.transpose(trans_order)


        layers = OrderedDict()
        inputs = OrderedDict()
        blobs = OrderedDict()

        for i, layer in enumerate(self.parsedmodel):
            layer_type = layer['type']
            layer_name = layer['name']
            top_blobs = layer['top_blobs']
            bottom_blobs = layer['bottom_blobs']
            layer_blobs = layer.get('blobs', None)

            if layer_name == 'fc6':
                break
            if verbose > 0:
                print("%d\t%s\t%s" % (i, layer_type, layer_name))

            if layer_type == 'DATA':
                # DATA layers contain input data in top_blobs, create input
                # variables, float for 'data' and int for 'label'
                for data_blob_name in top_blobs:
                    if data_blob_name == 'label':
                        blobs['label'] = T.ivector()
                        inputs['label'] = blobs['label']
                    else:
                        blobs[data_blob_name] = T.tensor4(dtype=float_dtype)
                        last_expression = blobs[data_blob_name]
                        inputs[data_blob_name] = blobs[data_blob_name]
            elif layer_type == 'CONVOLUTION':
                # CONVOLUTION layers take input from bottom_blob, convolve with
                # layer_blobs[0], and add bias layer_blobs[1]
                stride = layer['convolution_param__stride']
                stride_h = max(layer['convolution_param__stride_h'], stride)
                stride_w = max(layer['convolution_param__stride_w'], stride)
                if stride_h > 1 or stride_w > 1:
                    subsample = (stride_h, stride_w)
                else:
                    subsample = None
                pad = layer['convolution_param__pad']
                pad_h = max(layer['convolution_param__pad_h'], pad)
                pad_w = max(layer['convolution_param__pad_w'], pad)
                conv_filter = layer_blobs[0].astype(float_dtype)[..., ::-1, ::-1]
                conv_bias = layer_blobs[1].astype(float_dtype).ravel()

                convolution_input = blobs[bottom_blobs[0]]
                convolution = Convolution(conv_filter, biases=conv_bias,
                                          activation=None, subsample=subsample,
                                          input_dtype=float_dtype)
                # If padding is specified, need to pad. In practice, I think
                # caffe prevents padding that would make the filter see only
                # zeros, so technically this can also be obtained by sensibly
                # cropping a border_mode=full convolution. However, subsampling
                # may then be off by 1 and would have to be done separately :/
                if pad_h > 0 or pad_w > 0:
                    zp = ZeroPad((pad_h, pad_w))
                    zp._build_expression(convolution_input)
                    expression = zp.expression_
                    layers[layer_name] = (zp, convolution)
                else:
                    layers[layer_name] = convolution
                    expression = convolution_input
                convolution._build_expression(expression)
                expression = convolution.expression_
                # if subsample is not None:
                #     expression = expression[:, :, ::subsample[0],
                #                                     ::subsample[1]]

                blobs[top_blobs[0]] = expression
                current_expression = expression
            elif layer_type == "RELU":
                # RELU layers take input from bottom_blobs, set everything
                # negative to zero and write the result to top_blobs
                relu_input = blobs[bottom_blobs[0]]
                relu = Relu()
                relu._build_expression(relu_input)
                layers[layer_name] = relu
                blobs[top_blobs[0]] = relu.expression_
                current_expression = relu.expression_
            elif layer_type == "POOLING":
                # POOLING layers take input from bottom_blobs, perform max
                # pooling according to stride and kernel size information
                # and write the result to top_blobs
                pooling_input = blobs[bottom_blobs[0]]
                kernel_size = layer['pooling_param__kernel_size']
                kernel_h = max(layer['pooling_param__kernel_h'], kernel_size)
                kernel_w = max(layer['pooling_param__kernel_w'], kernel_size)
                stride = layer['pooling_param__stride']
                stride_h = max(layer['pooling_param__stride_h'], stride)
                stride_w = max(layer['pooling_param__stride_w'], stride)
                pad = layer['pooling_param__pad']
                pad_h = max(layer['pooling_param__pad_h'], pad)
                pad_w = max(layer['pooling_param__pad_w'], pad)
                pool_types = {0: 'max', 1: 'avg'}
                pool_type = pool_types[layer['pooling_param__pool']]
                # print "POOL TYPE is %s" % pool_type
                # pooling = FancyMaxPool((kernel_h, kernel_w),
                #                        (stride_h, stride_w),
                #                        ignore_border=False)
                pooling = CaffePool((kernel_h, kernel_w),
                                    (stride_h, stride_w),
                                    (pad_h, pad_w),
                                    pool_type=pool_type)
                pooling._build_expression(pooling_input)
                layers[layer_name] = pooling
                blobs[top_blobs[0]] = pooling.expression_
                current_expression = pooling.expression_
            elif layer_type == "DROPOUT":
                # DROPOUT may figure in some networks, but it is only relevant
                # at the learning stage, not at the prediction stage.
                pass
            elif layer_type == "SOFTMAX_LOSS" or layer_type == 'SOFTMAX':
                softmax_input = blobs[bottom_blobs[0]]
                # have to write our own softmax expression, because of shape
                # issues
                si = softmax_input.reshape((softmax_input.shape[0],
                                            softmax_input.shape[1], -1))
                shp = (si.shape[0], 1, si.shape[2])
                exp = T.exp(si - si.max(axis=1).reshape(shp))
                softmax_expression = (exp / exp.sum(axis=1).reshape(shp)
                                      ).reshape(softmax_input.shape)
                layers[layer_name] = "SOFTMAX"
                blobs[top_blobs[0]] = softmax_expression
                current_expression = softmax_expression
            elif layer_type == "SPLIT":
                split_input = blobs[bottom_blobs[0]]
                for top_blob in top_blobs:
                    blobs[top_blob] = split_input
                # Should probably make a class to be able to add to layers
                layers[layer_name] = "SPLIT"
            elif layer_type == "LRN":
                # Local normalization layer
                lrn_input = blobs[bottom_blobs[0]]
                lrn_factor = layer['lrn_param__alpha']
                lrn_exponent = layer['lrn_param__beta']
                axis = {0:'channels'}[layer['lrn_param__norm_region']]
                nsize = layer['lrn_param__local_size']
                lrn = LRN(nsize, lrn_factor, lrn_exponent, axis=axis)
                lrn._build_expression(lrn_input)
                layers[layer_name] = lrn
                blobs[top_blobs[0]] = lrn.expression_
                current_expression = lrn.expression_
            elif layer_type == "CONCAT":
                input_expressions = [blobs[bottom_blob] for bottom_blob
                                     in bottom_blobs]
                axis = layer['concat_param__concat_dim']
                output_expression = T.concatenate(input_expressions, axis=axis)
                blobs[top_blobs[0]] = output_expression
                layers[layer_name] = "CONCAT"
                current_expression = output_expression
            elif layer_type == "INNER_PRODUCT":
                weights = layer_blobs[0].astype(float_dtype)
                biases = layer_blobs[1].astype(float_dtype).squeeze()
                fully_connected_input = blobs[bottom_blobs[0]]
                if layer_name == 'fc6':
                    fully_connected_input = fully_connected_input.reshape((fully_connected_input.shape[0],
                                                                                18432, 1, 1))
                fc_layer = Convolution(weights.transpose((2, 3, 0, 1)), biases,
                                        activation=None)
                fc_layer._build_expression(fully_connected_input)
                layers[layer_name] = fc_layer
                blobs[top_blobs[0]] = fc_layer.expression_
                current_expression = fc_layer.expression_
            else:
                raise ValueError('layer type %s is not known to sklearn-theano'
                                 % layer_type)
            
            if layer_type == 'DATA':
                continue
            else:
                if isinstance(X, list):
                    X = X[0]
                to_compile = [current_expression]
                transform_function = theano.function([last_expression], to_compile)
                X = transform_function(X)
                transformations[top_blobs[0]] = X[0]
                last_expression = current_expression
        return transformations

    def transform(self, image, name=None):
        """
        This will transform the input image, get channel activations and return a hyperimage as feature for the image
        
        Parameters
        ==========
        image: numpy.ndarray
            the input image, a 3 channel image with RGB ordering, of shape `height * width * 3`
        name: str
            name of the image, so as to save the transformed image features as a pickle file
            default is None and in the default case pickle file won't be saved
        Returns
        =======
        hyperimage: numpy.ndarray
            features from all the convolutional layers in the network, concatenated to form a hyperimage of 
            size `height x width x 4224` for VGG 16 net.
        """
        transed = None
        raw_transed = None
        if name is not None:
            if name[-4:] != '.pkl':
                name = name + '.pkl'
            if os.path.exists(self.model_trans + name):
                f = file(self.model_trans + name, 'rb')
                transed = cPickle.load(f)
                f.close()
        if transed is None:
            raw_transed = self.__layerwiseTransform__(image)
        transed = {}
        for l in raw_transed.keys():
            if l.startswith('conv'):
                transed[l] = raw_transed[l][0,...].transpose((1,2,0))
        if name is not None:
            f = file(self.model_trans + name, 'wb')
            cPickle.dump(transed, f, protocol=2)
            f.close()

        stack_tup = tuple()
        for l in sorted(transed.keys()):
            sh = transed[l].shape
#            transed[l] = zoom(transed[l], (self.image_height / sh[0], self.image_width / sh[1], 1), order=1)
            transed[l] = cv2.resize(transed[l], (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            stack_tup = stack_tup + (transed[l],)
        hyperimage = np.dstack(stack_tup)
        transed = None
        return hyperimage


def main():
    vgg = VGG16Extractor()

if __name__ == '__main__':
    main()
