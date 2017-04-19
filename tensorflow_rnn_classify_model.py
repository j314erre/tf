
"""RNNClassify for Text Classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops



import logging
logger = logging.getLogger("tensorflow_RNNClassify")


class TFRNNClassifyModel(object):
    """RNNClassify for Text Classification


    """

    def __init__(self, source_vocab_size, sequence_length, embedding_size, num_classes, num_layers,
                             dropout_keep_prob, batch_size, learning_rate, use_lstm=False,
                             l2_reg_lambda=0.0):
        """Create the model.

        Args:
            source_vocab_size: max size of the source vocabulary.
            sequence_length: max number of symbols/words in a sequence/sentence
            embedding_size: number of units in the embedding layer.
            num_classes: number of classes or labels in the target vocabulary
            filters: list of filter window sizes (e.g. [3, 4, 5])
            num_filters: number of filters per window size
            dropout_keep_prob: Dropout keep probability
            batch_size: number of training examples per mini-batch
            learning_rate: learning rate to start with.
            l2_reg_lambda: L2 regularizaion lambda
        """
        self.source_vocab_size = source_vocab_size
        self.sequence_length = sequence_length
        #self.filters = filters
        self.num_classes = num_classes
        #self.num_filters = num_filters
        self.batch_size = batch_size
        
         
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        # learning rate can be set on the fly
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        
        # keep track of where we are
        self.global_step = tf.Variable(0, trainable=False)
        

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(
                tf.random_uniform([source_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # Tensor("embedding/embedding_lookup:0", shape=(?, sequence_length, embedding_size), dtype=float32
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            #self.embedded_concat = tf.reduce_mean(self.embedded_chars, 1)
            #  Tensor("embedding/ExpandDims:0", shape=(?, sequence_length, embedding_size, 1), dtype=float32, device=/device:CPU:0)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        encoder_inputs = tf.unstack(self.embedded_chars, axis=1)
            
        # Create the internal multi-layer cell for our RNN.
        
#         # Create a convolution + maxpool layer for each filter size
#         pooled_outputs = []
#         for i, filter_size in enumerate(filters):
#             with tf.name_scope("conv-maxpool-%s" % filter_size):
#                 # Convolution Layer
#                 filter_shape = [filter_size, embedding_size, 1, num_filters]
#                 W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#                 b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#                 conv = tf.nn.conv2d(
#                     self.embedded_chars_expanded,
#                     W,
#                     strides=[1, 1, 1, 1],
#                     padding="VALID",
#                     name="conv")
#                 # Apply nonlinearity
#                 h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#                 # Maxpooling over the outputs
#                 pooled = tf.nn.max_pool(
#                     h,
#                     ksize=[1, sequence_length - filter_size + 1, 1, 1],
#                     strides=[1, 1, 1, 1],
#                     padding='VALID',
#                     name="pool")
#                 pooled_outputs.append(pooled)
# 
# 
# 
#         # Combine all the pooled features
#         num_filters_total = num_filters * len(filters)
#         self.h_pool = tf.concat(3, pooled_outputs)
#         self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
#         
#         # Add dropout
#         with tf.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # default is GRU
        single_cell = tf.contrib.rnn.GRUCell(embedding_size)
        
        # otherwise use LS
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
            
        # start with single cell    
        encoder_cell = single_cell
        
        # create multi-layers
        if num_layers > 1:
            encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

        #encoder_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
        current_batch_size = array_ops.shape(encoder_inputs[0])[0]

        # encoder_outputs: list len=[encoder bucket sequence length] of 2D Tensors with TensorShape: (?=batch_size, embedding_size)
        encoder_state = None
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        with variable_scope.variable_scope("RNN_encoder") as varscope1:
            if varscope1.caching_device is None:
                varscope1.set_caching_device(lambda op: op.device)
        
                # encoder_outputs: list len=[encoder bucket sequence length] of 2D Tensors with TensorShape: (?=batch_size, embedding_size)
                encoder_outputs = []
                # encoder_state: TensorShape(?=batch_size, num_layers x embedding_size)
                encoder_state = encoder_cell.zero_state(current_batch_size, dtypes.float32)            
            
                for time, input_t in enumerate(encoder_inputs):
                    if time > 0: 
                        variable_scope.get_variable_scope().reuse_variables()
                    
                    
                    # output: 2D Tensor with TensorShape: (?=batch_size, embedding_size)
                    # encoder_state: TensorShape(?=batch_size, num_layers x embedding_size)
                    (output, encoder_state) = encoder_cell(input_t, encoder_state)
            
                    encoder_outputs.append(output)
        


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            #W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            W = tf.Variable(tf.constant(0.0, shape=[embedding_size, num_classes]), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(encoder_outputs[-1], W, b, name="logits")
            self.output_layer = tf.nn.softmax(self.logits, name="output_layer")
            self.predictions = tf.argmax(self.output_layer, 1, name="predictions")
            
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            #self.losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
            #self.losses = tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.input_y)
            self.losses = tf.nn.weighted_cross_entropy_with_logits(self.logits, self.input_y, 1.0)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss
            
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)   

        params = tf.trainable_variables()
        
        self.total_parameters = 0
        for p in params:
            # shape is an array of tf.Dimension
            shape = p.get_shape()
            pcount = 1
            for dim in shape:
                pcount *= dim.value
            self.total_parameters += pcount

        self.saver = tf.train.Saver(tf.all_variables())


    def get_batch(self, data):
        """Get a random batch of data from a data array.


        Args:
            data: a list of tuples...each tuple contains 
                [0] a list of of size self.sequence_length symbol/word ids for the input sequence/sentence of length
                [1] a list of size self.num_classes with output values


        Returns:
            A pair of lists (inputs, outputs) containing input words and output labels for the batch
            inputs: list of np.array of length self.sequence_length with word ids
            outputs: list of np.array of length self.num_classes with output values
        """
        encoder_inputs, decoder_inputs = [], []

            
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            
            assert len(encoder_input)==self.sequence_length
            assert len(decoder_input)==self.num_classes
            
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # package data in numpy arrays
        for batch_idx in xrange(self.batch_size):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                                        for length_idx in xrange(self.sequence_length)], dtype=np.int32))
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                                        for length_idx in xrange(self.num_classes)], dtype=np.int32))

        return batch_encoder_inputs, batch_decoder_inputs

    def get_x_y(self, data):
        """Get a x,y data from a data array.

        Args:
            data: a list of tuples...each tuple contains 
                [0] a list of of size self.sequence_length symbol/word ids for the input sequence/sentence of length
                [1] a list of size self.num_classes with output values

        Returns:
            A pair of lists (inputs, outputs) containing input words and output labels for the batch
            inputs: list of np.array of length self.sequence_length with word ids
            outputs: list of np.array of length self.num_classes with output values
        """
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for encoder_input, decoder_input in data:
            
            assert len(encoder_input)==self.sequence_length
            assert len(decoder_input)==self.num_classes

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []


        for batch_idx in xrange(len(data)):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                                        for length_idx in xrange(self.sequence_length)], dtype=np.int32))
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                                        for length_idx in xrange(self.num_classes)], dtype=np.int32))

        return batch_encoder_inputs, batch_decoder_inputs
