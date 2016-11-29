from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard packages
import math
import os
import random
import sys
import time
import argparse
import logging
import re

# special packages
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

# tensorflow packages
import tensorflow.python.platform
import tensorflow as tf
from tensorflow_seq2seq_model import TFSeq2SeqModel
from tensorflow.python.platform import gfile
from tensorflow.models.image.mnist.convolutional import BATCH_SIZE

# logger
logger = logging.getLogger("tensorflow_seq2seq")

# We use a number of buckets and pad to the closest one for efficiency.
# See tensorflow_seq2seq_model.Seq2SeqModel for details of how they work.
# len(_buckets) is the number of buckets
# for each bucket:
# tuples = (padded_encoder_length, padded_decoder_length)
_BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]
# Special vocabulary symbols - we always put them at the start.

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
_SOURCE_EXTRACT_GROUP = re.compile("^([^\\t]+)")
_TARGET_EXTRACT_GROUP = re.compile("\\t([^\\t]+)")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                                            tokenizer=None, normalize_digits=True, extract_regex=None):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to create vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        logger.info("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    logger.info("    processing %s line %d vocab size so far %d" % (data_path, counter, len(vocab)))
                    
                if extract_regex:
                    m = re.search(extract_regex, line)
                    if m:
                        line = m.group(1)
                    else:
                        logger.warn("Skipping empty data in at %s line %d" % (data_path, counter))
                        continue
                        
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                #logger.debug("line %d tokens=%s" % (counter, tokens))
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")
            logger.info("Read %d lines from %s, found %d vocab items" % (counter, data_path, len(vocab_list)))
    else:
        logger.info("Re-use existing vocabulary %s " % vocabulary_path)

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
        vocabulary_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    logger.info("Initialize vocabulary %s" % vocabulary_path)
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                                                    tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
        sentence: a string, the sentence to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
        a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                                            tokenizer=None, normalize_digits=True, extract_regex=None):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        logger.info("Creating file %s" % target_path)
        logger.info("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        logger.info("    tokenizing %s line %d" % (data_path, counter))
                    if extract_regex:
                        m = re.search(extract_regex, line)
                        if m:
                            line = m.group(1)
                        else:
                            logger.warn("Skipping empty tokens in at %s line %d" % (data_path, counter))
                            continue
                        
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                                                        normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
    else:
        logger.info("Re-using existing token file %s" % target_path)

def prepare_data(input_file, data_dir, source_vocabulary_size, target_vocabulary_size, dev_train_split=0.1, max_data_size=None):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
        input_file: data file with tab-delimited input/output pairs
        data_dir: directory in which the data sets will be stored.
        source_vocabulary_size: size of the source vocabulary to create and use.
        target_vocabulary_size: size of the target vocabulary to create and use.

    Returns:
        A tuple of 4 elements:
            (*) path to the token-ids for source training data-set,
            (*) path to the token-ids for target training data-set,
            (*) path to the Source vocabulary file,
            (*) path to the Target vocabulary file.
    """


    # Create vocabularies of the appropriate sizes.
    target_vocab_path = os.path.join(data_dir, "target%d.vocab" % target_vocabulary_size)
    source_vocab_path = os.path.join(data_dir, "source%d.vocab" % source_vocabulary_size)
    create_vocabulary(target_vocab_path, input_file, target_vocabulary_size, extract_regex=_TARGET_EXTRACT_GROUP)
    create_vocabulary(source_vocab_path, input_file, source_vocabulary_size, extract_regex=_SOURCE_EXTRACT_GROUP)

    # Create token ids for the training data.
    target_data_ids_path = os.path.join(data_dir, "target%d.ids" % target_vocabulary_size)
    source_data_ids_path = os.path.join(data_dir, "source%d.ids" % source_vocabulary_size)
    data_to_token_ids(input_file, target_data_ids_path, target_vocab_path, extract_regex=_TARGET_EXTRACT_GROUP)
    data_to_token_ids(input_file, source_data_ids_path, source_vocab_path, extract_regex=_SOURCE_EXTRACT_GROUP)


    # split dev and training
    target_train_ids_path = target_data_ids_path + ".train"
    target_dev_ids_path = target_data_ids_path + ".dev"
    source_train_ids_path = source_data_ids_path + ".train"
    source_dev_ids_path = source_data_ids_path + ".dev"

    
    if not gfile.Exists(target_train_ids_path):
        logger.info("Create train/dev split=%s" % dev_train_split)
        with gfile.GFile(source_data_ids_path, mode="r") as source_file:
            with gfile.GFile(target_data_ids_path, mode="r") as target_file:
                with gfile.GFile(target_train_ids_path, mode="w") as target_train_ids_file:
                    with gfile.GFile(target_dev_ids_path, mode="w") as target_dev_ids_file:
                        with gfile.GFile(source_train_ids_path, mode="w") as source_train_ids_file:
                            with gfile.GFile(source_dev_ids_path, mode="w") as source_dev_ids_file:
                                
                                counter = 0
                                source, target = source_file.readline(), target_file.readline()
                                while source and target and (not max_data_size or counter < max_data_size):
                                    counter += 1
                                    
                                    if counter % 100000 == 0:
                                        logger.info("....splitting %s line %d" % (source_data_ids_path, counter))
                                        sys.stdout.flush()
                                                                        
                                    # randomly select for dev vs training set
                                    if random.random() < dev_train_split:
                                        target_dev_ids_file.write(target)
                                        source_dev_ids_file.write(source)
                                    else:
                                        target_train_ids_file.write(target)
                                        source_train_ids_file.write(source)
                                        
                                    source, target = source_file.readline(), target_file.readline()

    else:
        logger.info("Re-using existing training file %s" % target_train_ids_path)
        if gfile.Exists(target_dev_ids_path):
            logger.info("Re-using existing dev file %s" % target_dev_ids_path)


    return (source_train_ids_path, target_train_ids_path, source_dev_ids_path, target_dev_ids_path,
                    source_vocab_path, target_vocab_path)

def read_data(source_path, target_path, buckets, max_size=None):
    """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    buckets: list of buckets
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
      """
    data_set = [[] for _ in buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    logger.info("  reading %s line %d" % (source_path, counter))
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (encoder_size, decoder_size) in enumerate(buckets):
                    # pick the first bucket the data fits into
                    # NOTE: if there is not bucket big enough data is not selected
                    if len(source_ids) < encoder_size and len(target_ids) < decoder_size:
                        
                        encoder_inputs, decoder_inputs = pad_sequence(source_ids, encoder_size, target_ids, decoder_size)
                        
                        
                        data_set[bucket_id].append([encoder_inputs, decoder_inputs])
                        break
                    
                source, target = source_file.readline(), target_file.readline()
    return data_set

def pad_sequence(source_sequence, encoder_size, target_sequence, decoder_size):
    """ Pad source and target sequence data up to their respective fixed sizes
    
        Source sequence is padded with PAD symbol up to the source_size
        Source sequence is reversed
        
        Target sequence is prepended with GO symbol
        Target sequence is padded with PAD symbol up to the target_size
        
    Args:
        source_sequence: list with source sequence ids
        encoder_size: size to pad up to, must be greater than len(source_sequence)
        target_sequence: list with target sequence ids
        decoder_size: size to pad up to, must be greater than len(target_sequence)
        
    Returns:
        encoder_inputs[encoder_size]
        decoder_inputs[decoder_size]
    """
    assert(len(source_sequence)<encoder_size)
    
    # Encoder inputs are padded and then reversed.
    encoder_pad = [PAD_ID] * (encoder_size - len(source_sequence))
    encoder_inputs = list(reversed(source_sequence + encoder_pad))

    # Decoder inputs get an extra "GO" symbol, and are padded then.
    assert(len(target_sequence)<decoder_size)
    decoder_pad_size = decoder_size - len(target_sequence) - 1
    decoder_inputs = [GO_ID] + target_sequence + [PAD_ID] * decoder_pad_size
    
    return encoder_inputs, decoder_inputs
    
def get_batch(data, encoder_size, decoder_size, batch_size, randomize=True):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
        data: a list of tuples ([encoder_input], [decoder_input]) training pairs we use to create a batch.
        encoder_size: list size of input data (1st element of tuple)
        decoder_size: list size of output data (2nd element of each tuple)
        batch_size: size of batch, if you want all the data in the batch use len(data)
        randomize: randomly select data (default=True)

    Returns:
        The triple (encoder_inputs, decoder_inputs, target_weights) for
        encoder_inputs[encoder_size][batch_size]: encoder inputs with right shape for model 
        decoder_inputs[decoder_size][batch_size]: decoder inputs with right shape for model
        target_weights[decoder_size][batch_size]: masks out PAD decoder inputs
    """
    # lists will contain matched pairs of input/output data for each training example in the batch
    encoder_inputs, decoder_inputs = [], []
        
    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for idx in xrange(batch_size):
        
        # select a tuple from the data
        if randomize is True:
            encoder_input, decoder_input = random.choice(data)
        else:
            encoder_input, decoder_input = data[idx]
            
        # make sure data matches the model
        assert len(encoder_input)==encoder_size
        assert len(decoder_input)==decoder_size
        
        # add to batch
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # go from encoder_inputs[batch_size][encoder_size] ==> batch_encoder_inputs[encoder_size][batch_size]
    # Batch encoder inputs list is size of padded input sequence length
    for input_idx in xrange(encoder_size):
        # each element is size of the batch
        batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][input_idx]
                                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # go from decoder_inputs[batch_size][decoder_size] ==> batch_decoder_inputs[decoder_size][batch_size]
    # Batch decoder inputs list is size of padded output sequence length
    for input_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][input_idx]
                                    for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if input_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][input_idx + 1]
            if input_idx == decoder_size - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
        
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def create_model(session, model_dir, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """Create translation model and initialize or load parameters in session."""
    model = TFSeq2SeqModel(source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm,
               num_samples, forward_only)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created new model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train(input_file, model_dir, max_train_data_size, dev_train_split, steps_per_checkpoint, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False, checkpoints=0, max_global_step=0):
    """Train a translation model using source, target data."""
    # Prepare WMT data.
    logger.info("Preparing data in %s" % model_dir)
    source_train_data, target_train_data, source_dev_data, target_dev_data, _, _ = prepare_data(input_file, 
                                                                                                model_dir, 
                                                                                                source_vocab_size, 
                                                                                                target_vocab_size,
                                                                                                dev_train_split=dev_train_split,
                                                                                                max_data_size=max_train_data_size
                                                                                                )

    with tf.Session() as sess:
        # Create model.
        logger.info("Creating %d layers of %d units." % (num_layers, size))
        model = create_model(sess, model_dir, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm,
               num_samples, forward_only)
    
        # Read data into buckets and compute their sizes.
        logger.info("Reading development and training data (limit: %d)."
               % max_train_data_size)
        dev_set = read_data(source_dev_data, target_dev_data, buckets)
        train_set = read_data(source_train_data, target_train_data, buckets)        
        
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
        logger.info("Training bucket sizes: %s" % train_bucket_sizes)
        
        train_total_size = sum(train_bucket_sizes)
        logger.info("Training set size: %s" % train_total_size)
        
        dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(buckets))]
        logger.info("Dev bucket sizes: %s" % dev_bucket_sizes)
        
        dev_total_size = sum(dev_bucket_sizes)
        logger.info("Dev set size: %s" % dev_total_size)
    
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / float(train_total_size)
                               for i in xrange(len(train_bucket_sizes))]
    
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        bucket_tally = [0] * len(buckets)
        counter = 0
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                               if train_buckets_scale[i] > random_number_01])
        
            # Get a batch of batch_size training examples
            # input sequence size depends on the bucket id
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights  = get_batch(data=train_set[bucket_id], 
                                                                          encoder_size=buckets[bucket_id][0], 
                                                                          decoder_size=buckets[bucket_id][1],
                                                                          batch_size=batch_size, 
                                                                          randomize=True)     
            
            bucket_tally[bucket_id] +=  np.size(encoder_inputs[0])      
            
            # make a step
            model.batch_size = batch_size
            
            _, _, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
            logger.info("train_step: %d bucket: %d step_loss: %f step_time: %.2f tally: %s" 
                         % (current_step, bucket_id, step_loss, step_time, bucket_tally))
        
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                checkpoint_path = os.path.join(model_dir, "model.checkpoint")
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                my_global_step = model.global_step.eval()
                logger.info("Checkpoint: global_step: %d learning_rate: %f step_time: %.2f loss: %f, perplexity: %f tally: %s" 
                            % (my_global_step, model.learning_rate.eval(),
                                 step_time, loss, perplexity, bucket_tally))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                logger.info("Saving global_step: %s to %s" % (model.global_step, checkpoint_path))
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                if dev_set:
                    logger.info("Starting evaluation on dev set")
                    dev_losses = []
                    #dev_pplxes = []
                    dev_batches = []
                    dev_time = time.time()
                    for bucket_id in xrange(len(buckets)):
                        dev_set_batch = dev_set[bucket_id]
                        if dev_set_batch:
                            model.batch_size = len(dev_set_batch)
                                                        
                            # We decode the whole dev batch
                            encoder_inputs, decoder_inputs, target_weights = get_batch(data=dev_set_batch, 
                                                                                       encoder_size=buckets[bucket_id][0], 
                                                                                       decoder_size=buckets[bucket_id][1],
                                                                                       batch_size=model.batch_size,
                                                                                       randomize=False)                        
                            
                            _, _, _, eval_loss, _ = model.step(session=sess, 
                                                         encoder_inputs=encoder_inputs, 
                                                         decoder_inputs=decoder_inputs,
                                                         target_weights=target_weights, 
                                                         bucket_id=bucket_id, 
                                                         forward_only=True)
                            #eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                            dev_losses.append(eval_loss)
                            #dev_pplxes.append(eval_ppx)
                            dev_batches.append(model.batch_size)
                    
                    dev_loss = 0
                    dev_total = 0
                    for i in range(len(dev_losses)):
                        dev_loss += dev_losses[i] * dev_batches[i]
                        dev_total += dev_batches[i]
                    if dev_total>0:
                        dev_loss = dev_loss / dev_total    
                    dev_perplexity = math.exp(dev_loss)    
                    dev_time = time.time() - dev_time   
                    logger.info("Evaluate: global_step: %d dev_time: %.2f dev_batches: %s dev_loss: %s loss: %s perplexity: %s" % (my_global_step, dev_time, dev_batches, dev_losses, dev_loss, dev_perplexity))
                counter += 1
                if checkpoints>0 and counter>=checkpoints:
                    logger.info("Checkpoint limit reached")
                    break
                if max_global_step>0 and my_global_step>=max_global_step:
                    logger.info("Max global step limit reached")
                    break


def decode(sentences, model_dir, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm,
               num_samples):
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, model_dir, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm,
               num_samples, forward_only=True)
        model.batch_size = 1  # We decode one sentence at a time.
    
        # Load vocabularies.
        source_vocab_path = os.path.join(model_dir,
                                     "source%d.vocab" % source_vocab_size)
        target_vocab_path = os.path.join(model_dir,
                                     "target%d.vocab" % target_vocab_size)
        source_vocab, _ = initialize_vocabulary(source_vocab_path)
        _, rev_target_vocab = initialize_vocabulary(target_vocab_path)
    
        # Decode from standard input.
        #sys.stdout.write("> ")
        #sys.stdout.flush()
        #sentence = sys.stdin.readline()
        for sentence in sentences:
            # Get token-ids for the input sentence.
            token_ids = sentence_to_token_ids(sentence, source_vocab)
            
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(buckets))
                           if buckets[b][0] > len(token_ids)])
            encoder_size, decoder_size = buckets[bucket_id]
            padded_token_ids, padded_decoder_ids = pad_sequence(source_sequence=token_ids, 
                                                                encoder_size=encoder_size, 
                                                                target_sequence=[], 
                                                                decoder_size=decoder_size)
            data_set = [(padded_token_ids, padded_decoder_ids)]

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = get_batch(
                                                                       data=data_set,
                                                                       encoder_size=encoder_size, 
                                                                       decoder_size=decoder_size,
                                                                       batch_size=1, 
                                                                       randomize=False)
            # Get output logits for the sentence.
            _, _, _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            logger.debug("decode: bucket_id=%d outputs=%s" % (bucket_id, outputs))
            # If there is an EOS symbol in outputs, cut them at that point.
            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]
                # Print out French sentence corresponding to outputs.
                
            logger.info("DECODE input=[%s] output=[%s]" % (sentence, " ".join([rev_target_vocab[output] for output in outputs])))


def self_test():
    # 2 small buckets
    buckets = [(3, 3), (6, 6)]
    
    # Fake data set for both the (3, 3) and (6, 6) bucket.
    fake_data = [
                 ([1, 1], [2, 2]), 
                 ([3, 3], [4]), 
                 ([5], [6]),
                 ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), 
                 ([3, 3, 3], [5, 6])
                ]
    
    data_set = [[] for _ in buckets]
    for source_ids, target_ids in fake_data:
        for bucket_id, (encoder_size, decoder_size) in enumerate(buckets):
            # pick the first bucket the data fits into
            # NOTE: if there is not bucket big enough data is not selected
            if len(source_ids) < encoder_size and len(target_ids) < decoder_size:
                
                encoder_inputs, decoder_inputs = pad_sequence(source_ids, encoder_size, target_ids, decoder_size)
                
                
                data_set[bucket_id].append([encoder_inputs, decoder_inputs])
                break
                
    """Test the translation model."""
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        logger.info("Self-test for neural translation model.")
        # Create model with vocabularies of 10
        
        # batch size 32
        batch_size = 32
        
        # 2 layers of size 32
        model = TFSeq2SeqModel(source_vocab_size=10, 
                               target_vocab_size=10, 
                               buckets=buckets, 
                               size=32, 
                               num_layers=2,
                               max_gradient_norm=5.0, 
                               batch_size=batch_size, 
                               learning_rate=0.3, 
                               learning_rate_decay_factor=0.99, 
                               num_samples=8)
        sess.run(tf.initialize_all_variables())
    
        
        loss = 0.0
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            
            encoder_inputs, decoder_inputs, target_weights = get_batch(data=data_set[bucket_id], 
                                                                       encoder_size=buckets[bucket_id][0], 
                                                                       decoder_size=buckets[bucket_id][1],
                                                                       batch_size=batch_size, 
                                                                       randomize=True)
            _, _, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                     bucket_id, False)
            loss += step_loss
        perplexity = math.exp(loss)
        my_global_step = model.global_step.eval()
        logger.info("Self-test: global_step=%d learning_rate=%f perplexity=%f" 
                            % (my_global_step, model.learning_rate.eval(), perplexity))


def main():
    parser = argparse.ArgumentParser(description='RNN Encoder-Decoder')
    parser.add_argument("--input", help="Input file for training, testing, or decoding")
    parser.add_argument("--model_dir", help="Model directory", default=None)
    parser.add_argument("--learning_rate", help="Learning rate.", type=float, default=0.5)
    parser.add_argument("--learning_rate_decay_factor", help="Learning rate decays by this much.", type=float, default=0.99)
    parser.add_argument("--max_gradient_norm", help="Clip gradients to this norm.", type=float, default=5.0)
    parser.add_argument("--batch_size", help="Batch size to use during training.", type=int, default=64)
    parser.add_argument("--size", help="Size of each model layer.", type=int, default=1024)
    parser.add_argument("--num_layers", help="Number of layers in the model.", type=int, default=3)
    parser.add_argument("--source_vocab_size", help="Source vocabulary size.", type=int, default=40000)
    parser.add_argument("--target_vocab_size", help="Target vocabulary size.", type=int, default=40000)
    parser.add_argument("--dev_train_split", help="Fraction of examples for dev/validation set", type=float, default=0.1)
    parser.add_argument("--max_train_data_size", help="Limit on the size of training data (0: no limit).", type=int, default=0)
    parser.add_argument("--steps_per_checkpoint", help="How many training steps to do per checkpoint.", type=int, default=200)
    parser.add_argument("--checkpoints", help="How many checkpoints to run", type=int, default=0)
    parser.add_argument("--max_global_step", help="Max number of global steps", type=int, default=0)
    parser.add_argument("--decode", help="Run decoding on input file if this is set to True.", default=False, action='store_true')
    parser.add_argument("--debug", help="Debug mode", default=False, action='store_true')
    parser.add_argument("--self_test", help="Run a self-test if this is set to True.", default=False, action='store_true')

    
    args = parser.parse_args()
    
        
    
    logger.setLevel(logging.INFO)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # create file handler which logs even debug messages
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info("Command line: %s" % " ".join(sys.argv))
    logger.info(args)

    if args.self_test:
        logger.info("START SELF-TEST")
        self_test()
        
    elif args.decode:
        if not os.path.exists(args.model_dir):
            raise ValueError("Model directory %s does not exist.", args.model_dir)
        
        if not os.path.isdir(args.model_dir):
            raise ValueError("Model directory %s is not a directory.", args.model_dir)
        
        if not os.path.exists(args.input) or not os.path.isfile(args.input):
            raise ValueError("Input file %s does not exist.", args.input)
        logger.info("START DECODE")
        sentences = []
        with open(args.input, "r") as f:
            for line in f:
                sentence = line.strip()
                if len(sentence)>0:
                    sentences.append(sentence)
        decode(sentences=sentences, 
               model_dir=args.model_dir, 
               source_vocab_size=args.source_vocab_size, 
               target_vocab_size=args.target_vocab_size, 
               buckets=_BUCKETS, 
               size=args.size,
               num_layers=args.num_layers, 
               max_gradient_norm=args.max_gradient_norm, 
               batch_size=args.batch_size, 
               learning_rate=args.learning_rate,
               learning_rate_decay_factor=args.learning_rate_decay_factor, 
               use_lstm=False,
               num_samples=512)
                
    else:
        if not os.path.exists(args.model_dir):
            raise ValueError("Model directory %s does not exist.", args.model_dir)
        
        if not os.path.isdir(args.model_dir):
            raise ValueError("Model directory %s is not a directory.", args.model_dir)
        
        if not os.path.exists(args.input) or not os.path.isfile(args.input):
            raise ValueError("Input file %s does not exist.", args.input)
        logger.info("START TRAINING")
        train(input_file=args.input, 
                model_dir=args.model_dir, 
                max_train_data_size=args.max_train_data_size,
                dev_train_split=args.dev_train_split,
                steps_per_checkpoint=args.steps_per_checkpoint,
                source_vocab_size=args.source_vocab_size, 
                target_vocab_size=args.target_vocab_size, 
                buckets=_BUCKETS, 
                size=args.size,
                num_layers=args.num_layers, 
                max_gradient_norm=args.max_gradient_norm, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate,
                learning_rate_decay_factor=args.learning_rate_decay_factor, 
                use_lstm=False,
                num_samples=512,
                checkpoints=args.checkpoints,
                max_global_step=args.max_global_step)

        
        
    logger.info("DONE")    

if __name__ == "__main__":
    main()
