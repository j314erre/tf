
"""CNN for Text Classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf



import logging
logger = logging.getLogger("tensorflow_cnn")


class TFCNNModel(object):
    """CNN for Text Classification

    This class implements a  Convolutional Neural Networks for Sentence Classification as described in
        http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

    """

    def __init__(self, source_vocab_size, sequence_length, embedding_size, num_classes, filters, num_filters,
                             dropout_keep_prob, batch_size, learning_rate,
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
        self.filters = filters
        self.num_classes = num_classes
        self.num_filters = num_filters
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
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create the internal multi-layer cell for our RNN.
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)



        # Combine all the pooled features
        num_filters_total = num_filters * len(filters)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            #W = tf.Variable(tf.constant(0.0, shape=[num_filters_total, num_classes]), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.output_layer = tf.nn.softmax(self.logits, name="output_layer")
            self.predictions = tf.argmax(self.output_layer, 1, name="predictions")
            
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
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
