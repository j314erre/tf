
"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf


import logging
logger = logging.getLogger("tensorflow_seq2seq")


class TFSeq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
        http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
        http://arxiv.org/pdf/1412.2007v2.pdf
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                             num_layers, max_gradient_norm, batch_size, learning_rate,
                             learning_rate_decay_factor, use_lstm=False,
                             num_samples=512, forward_only=False, device=None, summary=False):
        """Create the model.

        Args:
            source_vocab_size: size of the source vocabulary.
            target_vocab_size: size of the target vocabulary.
            buckets: a list of pairs (I, O), where I specifies maximum input length
                that will be processed in that bucket, and O specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than O will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.summary = summary
        
        # learning rate can be set on the fly
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        
        # keep track of where we are
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [size, self.target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                                                        self.target_vocab_size)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        
        # default is GRU
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        
        # otherwise use LS
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            
        # start with single cell    
        cell = single_cell
        
        # create multi-layers
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            
            
            # seq2seq.embedding_attention_seq2seq()  
            #    Args:
            #     encoder_inputs: a list of 1D int32 Tensors of shape [batch_size].
            #     decoder_inputs: a list of 1D int32 Tensors of shape [batch_size].
            #     cell: rnn_cell.MultiRNNCell / rnn_cell.GRUCell / rnn_cell.BasicLSTMCell
            #     num_encoder_symbols=source_vocab_size
            #     num_decoder_symbols=target_vocab_size
            #     output_projection: None or a pair (W, B) of output projection weights and
            #       biases; W has shape [cell.output_size x num_decoder_symbols] and B has
            #       shape [num_decoder_symbols]; if provided and feed_previous=True, each
            #       fed previous output will first be multiplied by W and added B.
            #     feed_previous=do_decode if True, only the first
            #       of decoder_inputs will be used (the "GO" symbol), and all other decoder
            #       inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            #       If False, decoder_inputs are used as given (the standard decoder case).

            # 
            #   Returns:
            #     outputs: A list of the same length as decoder_inputs of 2D Tensors with
            #       shape [batch_size x num_decoder_symbols] containing the generated outputs.
            #     states: The state of each decoder cell in each time-step. This is a list
            #       with length len(decoder_inputs) -- one item for each time-step.
            #       Each item is a 2D Tensor of shape [batch_size x cell.state_size].
            
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                                                             encoder_inputs=encoder_inputs, 
                                                             decoder_inputs=decoder_inputs, 
                                                             cell=cell, 
                                                             num_encoder_symbols=source_vocab_size,
                                                             num_decoder_symbols=target_vocab_size,
                                                             embedding_size=size, 
                                                             output_projection=output_projection,
                                                             feed_previous=do_decode)

        # Feeds for inputs...
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        
        # loop through highest number of inputs in the last bucket 
        for i in xrange(buckets[-1][0]):    # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            
        # loop through highest number of decoder inputs PLUS one
        # Since our targets are decoder inputs shifted by one, we need one more.
        # # Decoder inputs get an extra "GO" symbol
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one...not including the "GO" symbol
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, True),
                    softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [tf.nn.xw_plus_b(output, output_projection[0], output_projection[1]) for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, False),
                    softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        self.total_parameters = 0
        for p in params:
            # shape is an array of tf.Dimension
            shape = p.get_shape()
            pcount = 1
            for dim in shape:
                pcount *= dim.value
            self.total_parameters += pcount

        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))

        # Summary
        for bucket_id, loss in enumerate(self.losses):
            tf.scalar_summary("loss_%d" % bucket_id, loss)
            
        if self.summary:
            self.summaries = tf.merge_all_summaries()
            
        # Saver
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
                     bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of enconder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
#         if not forward_only:
#             output_feed = [self.updates[bucket_id],    # Update Op that does SGD.
#                                          self.gradient_norms[bucket_id],    # Gradient norm.
#                                          self.losses[bucket_id]]    # Loss for this batch.
#         else:
#             output_feed = [self.losses[bucket_id]]    # Loss for this batch.
#             for l in xrange(decoder_size):    # Output logits.
#                 output_feed.append(self.outputs[bucket_id][l])
# 
#         outputs = session.run(output_feed, input_feed)
#         if not forward_only:
#             return outputs[1], outputs[2], None    # Gradient norm, loss, no outputs.
#         else:
#             return None, outputs[0], outputs[1:]    # No gradient norm, loss, outputs.

        # Output feed: depends on whether we do a backward step or not.
        if forward_only:
            output_feed = [
                           self.losses[bucket_id]
                           ]    # Loss for this batch.
            for l in xrange(decoder_size):    # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            outputs = session.run(output_feed, input_feed)
            return None, None, None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
        elif self.summary:
            output_feed = [self.global_step,
                           self.summaries,
                           self.updates[bucket_id],    # Update Op that does SGD.
                           self.gradient_norms[bucket_id],    # Gradient norm.
                           self.losses[bucket_id] # Loss for this batch.
                           ]    
            (global_step, summaries, _, gradient_norms, losses) = session.run(output_feed, input_feed)
            return global_step, summaries, gradient_norms, losses, None    # Gradient norm, loss, no outputs.
        else:
            output_feed = [
                           self.global_step,
                           self.updates[bucket_id],    # Update Op that does SGD.
                           self.gradient_norms[bucket_id],    # Gradient norm.
                           self.losses[bucket_id] # Loss for this batch.
                           ]    
            (global_step, _, gradient_norms, losses) = session.run(output_feed, input_feed)
            return global_step, None, gradient_norms, losses, None    # Gradient norm, loss, no outputs.

