import os
import collections
import argparse
import logging
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
logger = logging.getLogger("tensorflow_rnn_word")

class TFRNNWordModel():
    # The hyperparameters used in the model:
    # - init_scale - the initial scale of the weights
    # - learning_rate - the initial value of the learning rate
    # - max_grad_norm - the maximum permissible norm of the gradient
    # - num_layers - the number of LSTM layers
    # - num_steps - the number of unrolled steps of LSTM
    # - hidden_size - the number of LSTM units
    # - max_epoch - the number of epochs trained with the initial learning rate
    # - max_max_epoch - the total number of epochs for training
    # - keep_prob - the probability of keeping weights in the dropout layer
    # - lr_decay - the decay of the learning rate for each epoch after "max_epoch"
    # - batch_size - the batch size
    def __init__(self, 
              batch_size=20,
              num_steps=20,
              hidden_size=200,
              vocab_size=10000,
              keep_prob=1.0,
              num_layers=2,
              max_grad_norm = 5,
              init_scale=0.1,
              epochs=13,
    ):
        # log hyper parameters
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        hyperparameters = "model hyperparameters: "
        for i in args:
            hyperparameters += "{name}={value}, ".format(name=i, value=values[i])
        logger.info(hyperparameters)

        self.batch_size = batch_size
        self.epochs = epochs
        self.num_steps = num_steps
            
            
        # create space for inputs
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # create space for answers/targets
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])    
        
        # create basic LSTM cell
        lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        
        # set dropout
        if keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
        # create layers of LSTM cells    
        self.rnn_layers = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    
        # initialize the state to zeros
        self.initial_state = self.rnn_layers.zero_state(batch_size, tf.float32)
        
        # do embeddings in cpu
        with tf.device("/cpu:0"):
            
            # Gets an existing variable or create a new one named "embedding" with shape [vocab_size, hidden_size]
            embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            #logger.debug("embedding tensor: {tensor}".format(tensor=embedding))
    
            # input layer shape [batch_size, num_steps, hidden_size]
            input_layer = tf.nn.embedding_lookup(embedding, self.input_data)
            logger.debug("input_layer tensor: %s" % input_layer)
    
        # apply dropout to hidden layer
        if keep_prob < 1:
            input_layer = tf.nn.dropout(input_layer, keep_prob)
        
        # book keeping for each time step    
        outputs = []
        states = []
        state = self.initial_state
        
        # all variables created in here will have scope "RNN"
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                # will re-use variables from one step to next
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                
                # slice input_layer by time_step into shape [batch_size, hidden_size]
                (cell_output, state) = self.rnn_layers(input_layer[:, time_step, :], state)
                #logger.debug("cell_output tensor: {tensor}".format(tensor=cell_output))
    
                # cell_output shape is [batch_size, hidden_size]
                outputs.append(cell_output)
                states.append(state)
    
        # outputs is array of tensors with shape [batch_size, hidden_size]
        # concat along dim1 into shape [num_steps, batch_size*hidden_size]
        catoutputs = tf.concat(1, outputs)
        logger.debug("catoutputs: {tensor}".format(tensor=catoutputs))
        
        # output reshape to [num_steps*batchsize, hidden_size]
        output = tf.reshape(catoutputs, [-1, hidden_size])
        logger.debug("output: {tensor}".format(tensor=output))
        
        # apply matmul(output, weights) + biases
        # logits shape [num_steps*batchsize, vocab_size]
        self.logits = tf.nn.xw_plus_b(output,
                                 tf.get_variable("softmax_w", [hidden_size, vocab_size]),
                                 tf.get_variable("softmax_b", [vocab_size]))
        logger.debug("logits: {tensor}".format(tensor=self.logits))
    
        self.output_layer = tf.nn.softmax(self.logits)
        
        logger.debug("output_layer: {tensor}".format(tensor=self.logits))
        
        # define loss function
        #   Weighted cross-entropy loss for a sequence of logits (per example).
        #   Args:
        #     logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
        #     targets: list of 1D batch-sized int32 Tensors of the same length as logits.
        #     weights: list of 1D batch-sized float-Tensors of the same length as logits.
        #     num_decoder_symbols: integer, number of decoder symbols (output classes).
        #     average_across_timesteps: If set, divide the returned cost by the total
        #       label weight.
        #     softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
        #       to be used instead of the standard softmax (the default if this is None).
        #     name: optional name for this operation, default: "sequence_loss_by_example".
        #   Returns:
        #     1D batch-sized float Tensor: the log-perplexity for each sequence.
        #   Raises:
        #     ValueError: if len(logits) is different from len(targets) or len(weights).
        #
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.ones([batch_size * num_steps])],
                                                vocab_size)
        
        # loss has shape [batch_size*num_steps]
        logger.debug("loss: {tensor}".format(tensor=loss))
    
        
        # compute average cost for batch
        self.cost_function = tf.reduce_sum(loss) / batch_size
        
        # name final state (the last row)
        self.final_state = states[-1]
    
        self.learning_rate_variable = tf.Variable(0.0, trainable=False)
        
        # Returns all variables created with trainable=True.
        trainable_vars = tf.trainable_variables()
        #logger.debug("trainable_vars: %{tensor}d".format(tensor=tf.size(trainable_vars)))
        
        # tf.gradients computes partial derivatives cost w.r.t. trainable_vars
        gradients = tf.gradients(self.cost_function, trainable_vars)
        
         
        # tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
        #       Given a tuple or list of tensors t_list, and a clipping ratio clip_norm, 
        #       this operation returns a list of clipped tensors list_clipped and the global norm 
        #       (global_norm) of all tensors in t_list. Optionally, if you've already computed the global 
        #       norm for t_list, you can specify the global norm with use_norm.
        list_clipped_gradients, global_norm  = tf.clip_by_global_norm(gradients, max_grad_norm)
        
        # Construct a new gradient descent optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_variable)
        
        # iterates tuples of gradients and training variables
        self.train_op = optimizer.apply_gradients(zip(list_clipped_gradients, trainable_vars))


    
    def train(self, session, training_data,               
              learning_rate=0.1,
              learning_rate_decay=0.0,
              save_path=None):
    
        logger.info("training_data: %s" % np.shape(training_data))
        
    
        
        ### PERFORM TRAINING
        
    
        tf.initialize_all_variables().run()
        
        # create a saver
        saver = tf.train.Saver(tf.all_variables())
    
        for i in range(self.epochs):
            
            # compute a learning rate decay
            session.run(tf.assign(self.learning_rate_variable, learning_rate))
            
            logger.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(self.learning_rate_variable)))

            
            """Runs the model on the given data."""
            epoch_size = ((len(training_data) // self.batch_size) - 1) // self.num_steps
            costs = 0.0
            iters = 0
            state = self.initial_state.eval()
            for step, (x, y) in enumerate(self.data_iterator(training_data, self.batch_size, self.num_steps)):
                
                # x and y should have shape [batch_size, num_steps]
                cost, state, _ = session.run([self.cost_function, self.final_state, self.train_op],
                                         {self.input_data: x,
                                          self.targets: y,
                                          self.initial_state: state})
                costs += cost
                iters += self.num_steps
        
                logger.debug("epoch %d step %d x: %s perplexity: %f" %(i, step, np.shape(x), np.exp(costs / iters)))

#                 if step>1 and step % 10 == 0:
#                     text, score = self.sample(session, num=20, init='the')
#                     logger.debug("checkpoint: %0.3f %s" % (score, text))
            
            logger.info("Saving...%s" % save_path)        
            saver.save(session, save_path)
                    
            train_perplexity = np.exp(costs / iters)
            logger.info("Finish epoch: %d Train Perplexity: %.3f" % (i, train_perplexity))
    
    def sample(self, sess, num=25, init=None):
        
        # return state tensor with batch size 1 set to zeros, eval
        state = self.rnn_layers.zero_state(1, tf.float32).eval()
        
        # run model forward through the intial characters
        prime = init.split()
        for char in prime[:-1]:
            
            # create a 1,1 tensor/scalar set to zero
            x = np.zeros((1, 1))
            
            # set to the vocab index
            x[0, 0] = vocab[char]
            
            
            # fetch: final_state
            # input_data = x, initial_state = state
            [state] = sess.run([self.final_state], {self.input_data: x, self.initial_state:state})
    
        def weighted_pick(weights):
            
            # an array of cummulative sum of weights
            t = np.cumsum(weights)
            
            # scalar sum of tensor
            s = np.sum(weights)
            
            # randomly selects a value from the probability distribution
            return(int(np.searchsorted(t, np.random.rand(1)*s)))
    
        # output starts with initialization
        ret = init
        
        # get last character in init
        char = prime[-1]
        
        # sample next num chars in the sequence after init
        score = 0.0
        
        for n in xrange(num):
            
            # init input to zeros
            x = np.zeros((1, 1))
            
            # lookup character index
            x[0, 0] = self.lookup_id[char]
     
            # probs = tf.nn.softmax(self.logits)
            # fetch: probs, final_state
            # input_data = x, initial_state = state
            [probs, state] = sess.run([self.output_layer, self.final_state], {self.input_data: x, self.initial_state:state})
    
            p = probs[0]
            logger.info("output=%s" % np.shape(p))
            # sample = int(np.random.choice(len(p), p=p))
            
            # select a random value from the probability distribution
            sample = weighted_pick(p)
            score += p[sample]
            # look up the key with the index
            logger.debug("sample[%d]=%d" % (n, sample))
            pred = self.vocabulary[sample]
            logger.debug("pred=%s" % pred)
            
            # add the car to the output
            ret += pred 
            ret += " "
            
            # set the next input character
            char = pred
        return ret, score
        
    def data_iterator(self, raw_data, batch_size, num_steps):
    
        # a huge array of integers load into a numby array
        raw_data = np.array(raw_data, dtype=np.int32)
    
        # total number of data items
        raw_data_len = len(raw_data)
        
        # the number of words in each batch
        batch_data_len = raw_data_len // batch_size
        
        # create a matrix of zeros with shape [batch_size, batch_data_len]
        data = np.zeros([batch_size, batch_data_len], dtype=np.int32)
        
        # chop data unto sequential blocks of size batch_data_len
        for i in range(batch_size):
            data[i] = raw_data[batch_data_len * i:batch_data_len * (i + 1)]
    
        # number of sequences of length num_steps per batch
        epoch_size = (batch_data_len - 1) // num_steps
    
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
        
        logger.info("Epoch steps: %d" % (epoch_size))
        
        # loop over number of sequences each num_steps long, create a generator 
        for i in range(epoch_size):
            # shape [batch_size, num_steps], y is one step ahead of x
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)
    
    def read_data(self, file_path=None):
        
        # read file into a HUGE array of words
        with open(file_path, "r") as f:
            all_words = f.read().replace("\n", "<eos>").split()    
    
        # create a counter class to tally word frequencies
        counter = collections.Counter(all_words)
        
        # sorts based on the word frequency, lambda is a function to turn data into keys, reverse sort by value
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    
        # get a list of words sorted by inverse frequency
        self.vocabulary, _ = list(zip(*count_pairs))
        
        # create a dictionary of words as keys and values as the word index
        self.lookup_id = dict(zip(self.vocabulary, range(len(self.vocabulary))))
    
        # turn all_words into a huge array of word indexes
        self.data = [self.lookup_id[word] for word in all_words]
        
        # the number of words in the dictionary
        self.vocabulary_size = len(self.lookup_id)
        logger.info("vocabulary_size: %d", self.vocabulary_size)
        
        return self.data, self.vocabulary_size

                  
    
    

