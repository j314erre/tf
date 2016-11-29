
"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


import logging
logger = logging.getLogger("tensorflow_Nseq2seq")


class TFNSeq2SeqModel(object):
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

    def __init__(self, source_vocab_size, target_vocab_size, buckets, max_context_window, size,
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
        self.max_context_window = max_context_window
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
        def Nseq2seq_f(encoder_inputs_matrix, decoder_inputs, do_decode):
            
            
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
            
            return self.embedding_attention_encoder_context_decoder_Nseq2seq(
                                                             encoder_inputs_matrix=encoder_inputs_matrix, 
                                                             decoder_inputs=decoder_inputs, 
                                                             cell=cell, 
                                                             num_encoder_symbols=source_vocab_size,
                                                             num_decoder_symbols=target_vocab_size,
                                                             embedding_size=size, 
                                                             context_size=size,
                                                             output_projection=output_projection,
                                                             feed_previous=do_decode)

        # Feeds for inputs...
        self.encoder_inputs_matrix = []
        self.decoder_inputs = []
        self.target_weights = []
        
        # loop through highest number of inputs in the last bucket 
        for n in range(max_context_window):
            encoder_inputs = []
            for i in xrange(buckets[-1][0]):    # Last bucket is the biggest one.
                encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}_{1}".format(i, n)))
            self.encoder_inputs_matrix.append(encoder_inputs)
            
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
            self.outputs, self.losses = self.model_with_buckets_Nseq2seq(
                                                                         encoder_inputs_matrix=self.encoder_inputs_matrix, 
                                                                         decoder_inputs=self.decoder_inputs, 
                                                                         targets=targets,
                                                                         weights=self.target_weights, 
                                                                         buckets=buckets,
                                                                         seq2seq=lambda x, y: Nseq2seq_f(x, y, True),
                                                                         softmax_loss_function=softmax_loss_function
                                                                         )
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [tf.nn.xw_plus_b(output, output_projection[0], output_projection[1]) for output in self.outputs[b]]
        else:
            self.outputs, self.losses = self.model_with_buckets_Nseq2seq(
                                                                         encoder_inputs_matrix=self.encoder_inputs_matrix, 
                                                                         decoder_inputs=self.decoder_inputs, 
                                                                         targets=targets,
                                                                         weights=self.target_weights, 
                                                                         buckets=buckets,
                                                                         seq2seq=lambda x, y: Nseq2seq_f(x, y, False),
                                                                         softmax_loss_function=softmax_loss_function
                                                                         )

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
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

    def step(self, session, encoder_inputs_matrix, decoder_inputs, target_weights,
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
            A tuple consisting of:
                global_step: current global step (training only)
                summary: tensorboard summary (training only, if summary=True)
                gradient norm: gradient norm (training only)
                losses: 
                outputs: forward_only
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of enconder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs_matrix[0]) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(encoder_inputs_matrix[0]), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for n in range(len(encoder_inputs_matrix)):
            for l in xrange(encoder_size):
                input_feed[self.encoder_inputs_matrix[n][l].name] = encoder_inputs_matrix[n][l]

        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

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
        



    def embedding_attention_encoder_context_decoder_Nseq2seq(self, 
                                     encoder_inputs_matrix, 
                                     decoder_inputs, 
                                     cell,
                                    num_encoder_symbols, 
                                    num_decoder_symbols,
                                    embedding_size,
                                    context_size,
                                    num_heads=1, 
                                    output_projection=None,
                                    feed_previous=False, 
                                    dtype=dtypes.float32,
                                    scope=None, 
                                    initial_state_attention=False
                                    ):
        """Embedding sequence-to-sequence model with attention.
    
      This model first embeds encoder_inputs by a newly created embedding (of shape
      [num_encoder_symbols x input_size]). Then it runs an RNN to encode
      embedded encoder_inputs into a state vector. It keeps the outputs of this
      RNN at every step to use for attention later. Next, it embeds decoder_inputs
      by another newly created embedding (of shape [num_decoder_symbols x
      input_size]). Then it runs attention decoder, initialized with the last
      encoder state, on embedded decoder_inputs and attending to encoder outputs.
    
      Args:
        encoder_inputs: A list of len[encoder bucket sequence size] 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of len[decoder bucket sequence size] 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
          biases; W has shape [output_size x num_decoder_symbols] and B has
          shape [num_decoder_symbols]; if provided and feed_previous=True, each
          fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
          of decoder_inputs will be used (the "GO" symbol), and all other decoder
          inputs will be taken from previous outputs (as in embedding_rnn_decoder).
          If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
          "embedding_attention_Nseq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states.
    
      Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x num_decoder_symbols] containing the generated
            outputs.
          state: The state of each decoder cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
        """
        with variable_scope.variable_scope(scope or "embedding_attention_Nseq2seq"):
            
            batch_size = array_ops.shape(encoder_inputs_matrix[0][0])[0]

            # ENCODER
            
            # add embedding to the RNN cell
            encoder_cell = rnn_cell.EmbeddingWrapper(
                                                     cell=cell, 
                                                     embedding_classes=num_encoder_symbols,
                                                     embedding_size=embedding_size
                                                     )
            
            # create the RNN network from the cell and encoder_inputs
            #     encoder_inputs: list len=[encoder bucket sequence length] of 1D Tensors with TensorShape: (?=batch_size)
#             encoder_outputs, encoder_state = rnn.rnn(
#                                                      cell=encoder_cell, 
#                                                      inputs=encoder_inputs, 
#                                                      dtype=dtype
#                                                      )
        
        
            # encoder_outputs: list len=[encoder bucket sequence length] of 2D Tensors with TensorShape: (?=batch_size, embedding_size)
            encoder_outputs_matrix = []
            encoder_state = None
            # Create a new scope in which the caching device is either
            # determined by the parent scope, or is set to place the cached
            # Variable using the same placement as for the rest of the RNN.
            with variable_scope.variable_scope(scope or "RNN_encoder") as varscope1:
                if varscope1.caching_device is None:
                    varscope1.set_caching_device(lambda op: op.device)
            
                # run encoder N times for each input sequence
                for N, encoder_inputs in enumerate(encoder_inputs_matrix):  
                    # encoder_outputs: list len=[encoder bucket sequence length] of 2D Tensors with TensorShape: (?=batch_size, embedding_size)
                    encoder_outputs = []
                    # encoder_state: TensorShape(?=batch_size, num_layers x embedding_size)
                    encoder_state = encoder_cell.zero_state(batch_size, dtype)            
                
                    for time, input_t in enumerate(encoder_inputs):
                        if time > 0: 
                            variable_scope.get_variable_scope().reuse_variables()
                        
                        
                        # output: 2D Tensor with TensorShape: (?=batch_size, embedding_size)
                        # encoder_state: TensorShape(?=batch_size, num_layers x embedding_size)
                        (output, encoder_state) = encoder_cell(input_t, encoder_state)
                
                        encoder_outputs.append(output)
            
                    encoder_outputs_matrix.append(encoder_outputs);
            
            
            # context inputs are the outputs from the last timestep of each of the N steps
            context_inputs = []
            for encoder_output in encoder_outputs_matrix:
                context_inputs.append(encoder_output[-1])
                
            context_outputs = []
            with variable_scope.variable_scope(scope or "RNN_context") as varscope2:
                if varscope2.caching_device is None:
                    varscope2.set_caching_device(lambda op: op.device)
            
                # context_state: TensorShape(?=batch_size, num_layers x embedding_size)
                context_state = cell.zero_state(batch_size, dtype)            
            
                # context_inputs:  list len=[N] of 2D TensorShape(?=batch_size, embedding_size)
                for context_time, context_input in enumerate(context_inputs):
                    if context_time > 0: 
                        variable_scope.get_variable_scope().reuse_variables()
                    
                    
                    # output: 2D Tensor with TensorShape: (?=batch_size, embedding_size)
                    # context_state: TensorShape(?=batch_size, num_layers x embedding_size)
                    (output, context_state) = cell(context_input, context_state)
            
                    context_outputs.append(output)
        
 
            # First calculate a concatenation of encoder outputs to put attention on.
            # encoder_outputs_matrix[-1] are the len=[encoder bucket sequence length] outputs from all timesteps for just the Nth sequence
            # context_outputs is len=[N] 
            # encoder_context_outputs: list len=[N+encoder_bucket_size] of 2D Tensor with TensorShape: (?=batch_size, embedding_size)
            encoder_context_outputs = encoder_outputs_matrix[-1] + context_outputs
            
            # top_states: list len=[input_sequence_size] of 3D Tensors with TensorShape: (?=batch_size, 1, embedding_size)
            top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                          for e in encoder_context_outputs]
            
            # attention_states: 3D TensorShape(?=batch_size, N + input_sequence_length, embedding_size)
            attention_states = array_ops.concat(1, top_states)
        
            # concat encoder + context state 
            #decoder_initial_state = tf.concat(1, [encoder_state, context_state])
            with variable_scope.variable_scope("encoder_context_combo"):
                decoder_initial_state = rnn_cell.linear([encoder_state, context_state], cell.state_size, True)
            
            # DECODER
            
            # if not using output_projection (softmax sampling) output is number of decoder symbols 
            # ...but this is rarely the case...
            output_size = None
            if output_projection is None:
                cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
                output_size = num_decoder_symbols
        
            # this is usually the case...
            if isinstance(feed_previous, bool):
                #return tf.nn.seq2seq.embedding_attention_decoder(
                
                # decoder_inputs: list len=[output_sequence_size] of 1D TensorShape(?=batch_size)
                return self.embedding_attention_decoder_Nseq2seq(
                                                                 decoder_inputs=decoder_inputs, 
                                                                 initial_state=decoder_initial_state, 
                                                                 attention_states=attention_states, 
                                                                 cell=cell,
                                                                 num_symbols=num_decoder_symbols, 
                                                                 embedding_size=embedding_size, 
                                                                 num_heads=num_heads,
                                                                 output_size=output_size, 
                                                                 output_projection=output_projection,
                                                                 feed_previous=feed_previous,
                                                                 initial_state_attention=initial_state_attention
                                                                 )
        
            # If feed_previous is a Tensor, we construct 2 graphs and use cond.
            def decoder(feed_previous_bool):
                reuse = None if feed_previous_bool else True
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
                    outputs, state = self.embedding_attention_decoder_Nseq2seq(
                        decoder_inputs, encoder_state, attention_states, cell,
                        num_decoder_symbols, embedding_size, num_heads=num_heads,
                        output_size=output_size, output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False,
                        initial_state_attention=initial_state_attention)
                return outputs + [state]
        
            outputs_and_state = control_flow_ops.cond(feed_previous,
                                                      lambda: decoder(True),
                                                      lambda: decoder(False))
            return outputs_and_state[:-1], outputs_and_state[-1]

    def attention_decoder_Nseq2seq(self, decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False):
        """RNN decoder with attention for the sequence-to-sequence model.
        
          In this context "attention" means that, during decoding, the RNN can look up
          information in the additional tensor attention_states, and it does this by
          focusing on a few entries from the tensor. This model has proven to yield
          especially good results in a number of sequence-to-sequence tasks. This
          implementation is based on http://arxiv.org/abs/1412.7449 (see below for
          details). It is recommended for complex sequence-to-sequence tasks.
        
          Args:
            decoder_inputs: A list of len[decoder bucket sequence length] 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor [batch_size x cell.state_size].
            attention_states: 3D Tensor [batch_size x attn_length x attn_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            output_size: Size of the output vectors; if None, we use cell.output_size.
            num_heads: Number of attention heads that read from attention_states.
            loop_function: If not None, this function will be applied to i-th output
              in order to generate i+1-th input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            dtype: The dtype to use for the RNN initial state (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "attention_decoder".
            initial_state_attention: If False (default), initial attentions are zero.
              If True, initialize the attentions from the initial state and attention
              states -- useful when we wish to resume decoding from a previously
              stored decoder state and attention states.
        
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                  cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                  new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                  output = linear(cell_output, new_attn).
              state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
        
          Raises:
            ValueError: when num_heads is not positive, there are no inputs, or shapes
              of attention_states are not set.
        """
        if not decoder_inputs:
            raise ValueError("Must provide at least 1 input to attention decoder.")
        if num_heads < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                             % attention_states.get_shape())
        if output_size is None:
            output_size = cell.output_size
        
        with variable_scope.variable_scope(scope or "attention_decoder_Nseq2seq"):
            
            # decoder_inputs: list len=[output_sequence_size] of 1D TensorShape(?=batch_size)
            batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            
            # attention_states: TensorShape(?=batch_size, input_sequence_length, embedding_size)      
            
            # length of encoder input sequence      
            attn_length = attention_states.get_shape()[1].value
            
            # encoder embedding / hidden state size
            attn_size = attention_states.get_shape()[2].value
        
            # compute W1 * [encoder hidden states]
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.            
            # attention_states: 3D TensorShape(?=batch_size, input_sequence_length, embedding_size)      
            # hidden: 4D TensorShape(?=batch_size, input_sequence_length, 1, embedding_size)
            hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
            
            # collect W1*h matrix for each "head"
            hidden_features = []
            
            # collect V matrix for each "head"
            v = []
            
            # Size of query vectors for attention = size of embedding / hidden state size
            attention_vec_size = attn_size  
            
            # default is just 1 head
            for a in xrange(num_heads):
                k = variable_scope.get_variable("AttnW_%d" % a,
                                              [1, 1, attn_size, attention_vec_size])
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(variable_scope.get_variable("AttnV_%d" % a,
                                                   [attention_vec_size]))
        
        
            def attention(query):
                """Put attention masks on hidden using hidden_features and query.
                    Args:
                        query: a DECODER hidden state 2D TensorShape(?=batch_size, num_layers X decoder_size)
                
                    Returns:
                        attention-weighted vector d_t = sum_i(a_t * hidden)
                        2D TensorShape: (?=batch_size, attention_vec_size)
                        
                """
                ds = []  # Results of attention heads will be stored here.
                for a in xrange(num_heads):
                    with variable_scope.variable_scope("Attention_%d" % a):

                        # compute W2 * [decoder hidden states]
                        # linear map:   sum_i(query[i] * W2[i]), where W2[i]s are newly created matrices. 
                        # y: 2D TensorShape: (?=batch_size, attention_vec_size)
                        y = rnn_cell.linear(query, attention_vec_size, bias=True)
                        
                        # A 4D TensoShape: (?=batch_size, 1, 1, attention_vec_size)
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        
                        # Attention mask is a softmax of v^T * tanh(...).
                        
                        # u_t = vT * tanh(W1*h + W2*d)
                        # u_t: 2D TensorShape: (?=batch_size, attn_length(encoder_time_steps))
                        u_t = math_ops.reduce_sum(
                                                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                        
                        # a = softmax(u_t)
                        # a_t: 2D TensorShape: (?=batch_size, attn_length(encoder_time_steps))
                        a_t = nn_ops.softmax(u_t)
                        
                        # Now calculate the attention-weighted vector d_prime = sum_t(a_t * h_t)
                        # 2D TensorShape: (?=batch_size, attention_vec_size(enbedding_size))
                        d_prime = math_ops.reduce_sum(
                                                array_ops.reshape(a_t, [-1, attn_length, 1, 1]) * hidden,
                                                [1, 2])
                        ds.append(array_ops.reshape(d_prime, [-1, attn_size]))
                return ds
        
        
            # initialize the decoder
            outputs = []
            state = initial_state
            prev = None
            batch_attn_size = array_ops.pack([batch_size, attn_size])
            
            # initialize attention vector to zeros: 2D TensorShape: (?=batch_size, attention_vec_size)
            attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                     for _ in xrange(num_heads)]
            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])
                
                
            if initial_state_attention:
                attns = attention(initial_state)
                
            # loop over decoder_inputs
            # decoder_inputs: len[decoder bucket sequence length] of 2D TensorShape: (?=batch_size, embedding_size)    
            for i, inp in enumerate(decoder_inputs):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                    # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    with variable_scope.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                # Merge input and previous attentions into one vector of the right size.
                x = rnn_cell.linear([inp] + attns, cell.input_size, True)
                # Run the RNN.
                cell_output, state = cell(x, state)
                # Run the attention mechanism.
                if i == 0 and initial_state_attention:
                    with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
                        attns = attention(state)
                else:
                    # attns: len[num_heads] of 2D TensorShape: (?=batch_size, attention_vec_size(enbedding_size))
                    # aka a list of [ d_prime = sum_t(a_t * h_t) ]
                    attns = attention(state)

                with variable_scope.variable_scope("AttnOutputProjection"):
                    output = rnn_cell.linear([cell_output] + attns, output_size, True)
                if loop_function is not None:
                    prev = output
                outputs.append(output)
        
        return outputs, state
        
        
    def embedding_attention_decoder_Nseq2seq(self, decoder_inputs, initial_state, attention_states,
                                        cell, num_symbols, embedding_size, num_heads=1,
                                        output_size=None, output_projection=None,
                                        feed_previous=False,
                                        update_embedding_for_previous=True,
                                        dtype=dtypes.float32, scope=None,
                                        initial_state_attention=False):
        """RNN decoder with embedding and attention and a pure-decoding option.
        
          Args:
            decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
            initial_state: 2D Tensor [batch_size x cell.state_size].
            attention_states: 3D Tensor [batch_size x attn_length x attn_size].
            cell: rnn_cell.RNNCell defining the cell function.
            num_symbols: Integer, how many decoder symbols come into the embedding.
            embedding_size: Integer, the length of the embedding vector for each symbol.
            num_heads: Number of attention heads that read from attention_states.
            output_size: Size of the output vectors; if None, use output_size.
            output_projection: None or a pair (W, B) of output projection weights and
              biases; W has shape [output_size x num_symbols] and B has shape
              [num_symbols]; if provided and feed_previous=True, each fed previous
              output will first be multiplied by W and added B.
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
              In effect, this implements a greedy decoder. It can also be used
              during training to emulate http://arxiv.org/abs/1506.03099.
              If False, decoder_inputs are used as given (the standard decoder case).
            update_embedding_for_previous: Boolean; if False and feed_previous=True,
              only the embedding for the first symbol of decoder_inputs (the "GO"
              symbol) will be updated by back propagation. Embeddings for the symbols
              generated from the decoder itself remain unchanged. This parameter has
              no effect if feed_previous=False.
            dtype: The dtype to use for the RNN initial states (default: tf.float32).
            scope: VariableScope for the created subgraph; defaults to
              "embedding_attention_decoder".
            initial_state_attention: If False (default), initial attentions are zero.
              If True, initialize the attentions from the initial state and attention
              states -- useful when we wish to resume decoding from a previously
              stored decoder state and attention states.
        
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
        
          Raises:
            ValueError: When output_projection has the wrong shape.
        """
        if output_size is None:
            output_size = cell.output_size
        if output_projection is not None:
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])
        
        with variable_scope.variable_scope(scope or "embedding_attention_decoder_Nseq2seq"):
            
            # create decoder embedding
            with ops.device("/cpu:0"):
                decoder_embedding = variable_scope.get_variable("decoder_embedding",
                                                      [num_symbols, embedding_size])
                
            # Get a loop_function that extracts the previous symbol and embeds it    
            loop_function = tf.nn.seq2seq._extract_argmax_and_embed(
                decoder_embedding, output_projection,
                update_embedding_for_previous) if feed_previous else None
                
            # decoder_inputs: list len=[output_sequence_size] of 1D TensorShape(?=batch_size)    
            # convert decoder symbol inputs into decoder embedding inputs     
            # embedding_decoder_inputs:  list len=[output_sequence_size] of 2D TensorShape(?=batch_size, embedding_size)          
            embedding_decoder_inputs = [embedding_ops.embedding_lookup(decoder_embedding, i) for i in decoder_inputs]
            
            
            return self.attention_decoder_Nseq2seq(
                                                   decoder_inputs=embedding_decoder_inputs, 
                                                   initial_state=initial_state, 
                                                   attention_states=attention_states, 
                                                   cell=cell, 
                                                   output_size=output_size,
                                                   num_heads=num_heads, 
                                                   loop_function=loop_function,
                                                   initial_state_attention=initial_state_attention
                                                   )



    def model_with_buckets_Nseq2seq(self, encoder_inputs_matrix, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
        """Create a sequence-to-sequence model with support for bucketing.
    
      The seq2seq argument is a function that defines a sequence-to-sequence model,
      e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))
    
      Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
          agree with encoder_inputs and decoder_inputs, and returns a pair
          consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
          to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
          tensor of losses for each sequence in the batch. If unset, it will be
          a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".
    
      Returns:
        A tuple of the form (outputs, losses), where:
          outputs: The outputs for each bucket. Its j'th element consists of a list
            of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
          losses: List of scalar Tensors, representing losses for each bucket, or,
            if per_example_loss is set, a list of 1D batch-sized float Tensors.
    
      Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
              than the largest (last) bucket.
        """
        
        if len(encoder_inputs_matrix[0]) < buckets[-1][0]:
            raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                             "st bucket (%d)." % (len(encoder_inputs_matrix[0]), buckets[-1][0]))
        if len(targets) < buckets[-1][1]:
            raise ValueError("Length of targets (%d) must be at least that of last"
                             "bucket (%d)." % (len(targets), buckets[-1][1]))
        if len(weights) < buckets[-1][1]:
            raise ValueError("Length of weights (%d) must be at least that of last"
                             "bucket (%d)." % (len(weights), buckets[-1][1]))
    
        all_inputs = encoder_inputs_matrix + decoder_inputs + targets + weights
        losses = []
        outputs = []
        with ops.op_scope(all_inputs, name, "model_with_buckets"):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                     reuse=True if j > 0 else None):
                    
                    
                    encoder_inputs_sliced = []
                    for n in range(len(encoder_inputs_matrix)):
                        encoder_inputs_sliced.append(encoder_inputs_matrix[n][:bucket[0]])
                    bucket_outputs, _ = seq2seq(encoder_inputs_sliced,
                                                decoder_inputs[:bucket[1]])
                    outputs.append(bucket_outputs)
                    if per_example_loss:
                        losses.append(tf.nn.seq2seq.sequence_loss_by_example(
                              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                              softmax_loss_function=softmax_loss_function))
                    else:
                        losses.append(tf.nn.seq2seq.sequence_loss(
                              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                              softmax_loss_function=softmax_loss_function))
        
        return outputs, losses
