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
from tensorflow_cnn_model import TFCNNModel
from tensorflow.python.platform import gfile

# logger
logger = logging.getLogger("tensorflow_cnn")


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


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, init_vocab=None,
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
            vocab_list = init_vocab + sorted(vocab, key=vocab.get, reverse=True)
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

def prepare_data(input_file, data_dir, source_vocabulary_size, target_vocabulary_size, dev_train_split=0.1, max_data_size=None, normalize_digits=True):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
        input_file: data file with tab-delimited input/output pairs
        data_dir: directory in which the data sets will be stored.
        source_vocabulary_size: size of the source vocabulary to create and use.
        target_vocabulary_size: size of the target vocabulary to create and use.
        dev_train_split: fraction of examples to use for dev set.
        max_data_size: max number of records to read [or None for read all available].

    Returns:
        A tuple of 6 elements:
            (*) path to the token-ids for source training data-set,
            (*) path to the token-ids for target training data-set,
            (*) path to the token-ids for source dev data-set,
            (*) path to the token-ids for target dev data-set,
            (*) path to the Source vocabulary file,
            (*) path to the Target vocabulary file.
    """


    # Create vocabularies of the appropriate sizes.
    target_vocab_path = os.path.join(data_dir, "target%d.vocab" % target_vocabulary_size)
    source_vocab_path = os.path.join(data_dir, "source%d.vocab" % source_vocabulary_size)
    create_vocabulary(target_vocab_path, input_file, target_vocabulary_size, [], extract_regex=_TARGET_EXTRACT_GROUP, normalize_digits=normalize_digits)
    create_vocabulary(source_vocab_path, input_file, source_vocabulary_size, _START_VOCAB, extract_regex=_SOURCE_EXTRACT_GROUP, normalize_digits=normalize_digits)

    # Create token ids for the training data.
    target_data_ids_path = os.path.join(data_dir, "target%d.ids" % target_vocabulary_size)
    source_data_ids_path = os.path.join(data_dir, "source%d.ids" % source_vocabulary_size)
    data_to_token_ids(input_file, target_data_ids_path, target_vocab_path, extract_regex=_TARGET_EXTRACT_GROUP, normalize_digits=normalize_digits)
    data_to_token_ids(input_file, source_data_ids_path, source_vocab_path, extract_regex=_SOURCE_EXTRACT_GROUP, normalize_digits=normalize_digits)

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
                                        logger.info("  splitting data line %d" % counter)
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
   
    # Create token ids for the development data.
#     fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
#     en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
#     data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path)
#     data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path)

    return (source_train_ids_path, target_train_ids_path, source_dev_ids_path, target_dev_ids_path,
                    source_vocab_path, target_vocab_path)

def read_data(source_path, target_path, max_sequence_length, num_classes, max_filter_size=1, max_size=None):
    """Read data from source and target files.
    
        Apply padding to source sequences, truncate sequences to max sequence length.

  Args:
    source_path: path to the files with token-ids for the input sentences.
    target_path: path to the file with token-ids for the target classes;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_sequence_length: length for all data sequences, shorter sequences will be padded to this length
    num_classes: number of classes in target
    max_filter_length: size of largest filter for padding purposes, use 1 if no filter padding is needed
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set:  data_set[n] is a list of
      [[source_ids], [target_ids]] pairs read from the provided data files 
      len([source_ids]) = max_sequence_length
      len([target_ids]) = num_classes
      """
    data_set = []
    logger.info("Loading data files: %s %s" % (source_path, target_path))
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    logger.info("  reading data line %d" % counter)
                    sys.stdout.flush()
                    
                
                # read source ids    
                source_ids = [int(x) for x in source.split()]
                
                # pad source ids to a fixed length
                source_ids_padded = pad_sequence(source_ids, max_sequence_length, max_filter_size)
                
                target_ids = [int(x) for x in target.split()]
                
                target_vector = [0.0]*num_classes
                for target_id in target_ids:
                    target_vector[target_id] = 1.0

                data_set.append([source_ids_padded, target_vector])

                source, target = source_file.readline(), target_file.readline()
                
    return data_set

def pad_sequence(sequence, max_sequence_length, max_filter_size=1):
    """ Pad a sequence for CNN.
    
    Padding at the beginning is added to accommodate max_filter_size
    Padding at the end to bring sequence up to max_sequence_length including space for max_filter_size
    Sequence is truncated if it can't fit into max_sequence_length
    PAD_ID is the padding symbol
    
    Args:
        sequence: the list to be padded
        max_sequence_length: the size of the list returned with padding
        max_filter_size: filter size to figure out how much padding goes at the beginning
    
    Returns:
        padded_sequence: sequence of length max_sequence_length
    """
    
    # padding to accommodate filter at beginning and end
    pad_filter = max_filter_size -1
    pad_beginning = pad_filter
    pad_ending = pad_filter
    
    # extra space left over that needs to be padded
    pad_extra = max_sequence_length - len(sequence) - 2*pad_filter
    
    # add extra padding to the end if needed
    if pad_extra>0:
        pad_ending = pad_ending + pad_extra
        
    # see if we need to truncate the sequence
    truncate = None
    if pad_extra<0:
        truncate = pad_extra
        
    # create a padded fixed length sequence
    padded_sequence = [PAD_ID]*pad_beginning + sequence[:truncate] + [PAD_ID]*pad_ending

    assert len(padded_sequence)==max_sequence_length
    
    return padded_sequence


def initialize_embedding_word2vec(input_size, embedding_size, vocab, word2vec_path):
    """ Initialize embedding tensor using word2vec data
    
    Uses values found in word2vec file, otherwise random uniform
    
    Args:
        input_size: the number of inputs in the first layer
        embedding_size: the size of the embedding (must match the data size in word2vec file)
        vocab: a dictionary mapping words to inputs
        word2vec_path: the file path of the word2vec data file (i.e. GoogleNews-vectors-negative300.bin)
        
        
    Returns:
        W: numpy array of shape (input_size, embedding_size)
    """
    
    # initial matrix with random uniform
    W = np.random.uniform(-0.25,0.25,(input_size, embedding_size))
    
    #assert(len(W)==vocab_size)
    
    # load any vectors from the word2vec
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                idx = vocab[word]
                W[idx] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)    
        
        
    return W

def create_model(session, model_dir, source_vocab_size, max_sequence_length, num_classes, embedding_size, filters, num_filters,
               dropout_keep_prob, batch_size, learning_rate, word2vec_path=None):
    """Create tensorflow CNN model and initialize or load parameters in session."""
    model = TFCNNModel(source_vocab_size, max_sequence_length, embedding_size, num_classes, filters, num_filters,
                             dropout_keep_prob, batch_size, learning_rate,
                             l2_reg_lambda=0.0)    
    
    

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created new model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        if word2vec_path:
            source_vocab_path = os.path.join(model_dir, "source%d.vocab" % source_vocab_size)
            source_vocab, _ = initialize_vocabulary(source_vocab_path)
            logger.info("Initialize embedding from word2vec file: " + word2vec_path)
            init_embedding = initialize_embedding_word2vec(input_size=source_vocab_size, 
                                                           embedding_size=embedding_size,
                                                           vocab=source_vocab,
                                                           word2vec_path=word2vec_path)
            session.run(model.embedding.assign(init_embedding))
    return model


def train(input_file, model_dir, max_data_size, dev_train_split, steps_per_checkpoint, source_vocab_size,  max_sequence_length, 
          num_classes, embedding_size, filters, num_filters,
               dropout_keep_prob, batch_size, learning_rate,
               checkpoints=0, max_global_step=0, word2vec_path=None, normalize_digits=True):
    """Train a CNN classification model using source, target data."""
    # Prepare WMT data.
    logger.info("Preparing data in %s" % model_dir)
    source_train_data, target_train_data, source_dev_data, target_dev_data, _, _ = prepare_data(input_file, 
                                                                                                data_dir=model_dir, 
                                                                                                source_vocabulary_size=source_vocab_size, 
                                                                                                target_vocabulary_size=num_classes, 
                                                                                                dev_train_split=dev_train_split,
                                                                                                max_data_size=max_data_size,
                                                                                                normalize_digits=normalize_digits
                                                                                                )

    with tf.Session() as sess:
        # Create model.
        logger.info("Creating %d filters of sizes %s" % (num_filters, filters))
        model = create_model(sess, model_dir, source_vocab_size,  max_sequence_length, num_classes, embedding_size, filters, num_filters,
               dropout_keep_prob, batch_size, learning_rate, word2vec_path)
    
        logger.info("Total Parameters: %s" % model.total_parameters)
        # Read data into buckets and compute their sizes.
        logger.info("Reading development and training data (limit: %d)."
               % max_data_size)
        #dev_set = read_data(en_dev, fr_dev)
        train_set = read_data(source_train_data, target_train_data, max_sequence_length, num_classes, max_filter_size=max(filters))    
        dev_set = read_data(source_dev_data, target_dev_data, max_sequence_length, num_classes, max_filter_size=max(filters))    

        
        logger.info("Training set size: %s" % len(train_set))
        logger.info("Dev set size: %s" % len(dev_set))
        
        #dev_x, dev_y = model.get_x_y(dev_set)     
             
        # This is the training loop.
        step_time, loss, accuracy = 0.0, 0.0, 0.0
        current_step = 0
        previous_losses = []
        counter = 0
        tally = 0
        while True:
            # make a step            
            model.batch_size = batch_size
        
            # Get a batch of batch_size training examples
            # input sequence size depends on the bucket id
            start_time = time.time()
            encoder_inputs, decoder_inputs = model.get_batch(train_set)     
            
            
            
            feed_dict = {
                         model.input_x: encoder_inputs,
                         model.input_y: decoder_inputs,
                         model.dropout_keep_prob: dropout_keep_prob
                         }
            logger.debug("train: input_x=%s", encoder_inputs)
            logger.debug("train: input_y=%s", decoder_inputs)
            logits, softmax, _, step , step_loss, step_accuracy = sess.run(
                                                    [model.logits, model.output_layer, model.train_op, model.global_step, model.loss, model.accuracy],
                                                          feed_dict)
            logger.debug("train: logits=%s", logits)
            logger.debug("train: softmax=%s", softmax)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            accuracy += step_accuracy / steps_per_checkpoint
            current_step += 1
            tally = step*len(encoder_inputs)
            logger.info("train: step: %d  step-time: %.2f tally: %d step_loss: %f step_acc: %f" 
                         % (current_step, step_time, tally, step_loss, step_accuracy))
        
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                checkpoint_path = os.path.join(model_dir, "model.checkpoint")
                # Print statistics for the previous epoch.
                #perplexity = math.exp(loss) if loss < 300 else float('inf')
                my_global_step = model.global_step.eval()
                
                logger.info("Checkpoint: global_step: %d learning_rate: %f step_time: %.2f tally: %d loss: %f accuracy: %f " 
                            % (my_global_step, model.learning_rate.eval(),
                                 step_time, my_global_step*model.batch_size, loss, accuracy))
                # Decrease learning rate if no improvement was seen over last 3 times.
#                 if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
#                     sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss, accuracy = 0.0, 0.0, 0.0
                
                # Run evals on development set and print loss and accuracy
                if dev_set:
                    logger.info("Starting evaluation on dev set")
                    dev_results = []
                    #dev_pplxes = []
                    dev_batches = []
                    dev_time = time.time()


                    dev_set_batch_offset = 0
                    dev_set_batch_count = 1

                    while dev_set_batch_offset < len(dev_set):
                        dev_set_batch_time = time.time()
                        model.batch_size = batch_size
                        if dev_set_batch_offset+batch_size > len(dev_set):
                            model.batch_size = len(dev_set) - dev_set_batch_offset 
                                                
                        dev_set_batch = dev_set[dev_set_batch_offset:dev_set_batch_offset+model.batch_size]      
        
                        dev_x, dev_y = model.get_x_y(dev_set_batch)     

                        dev_feed_dict = {
                          model.input_x: dev_x,
                          model.input_y: dev_y,
                          model.dropout_keep_prob: 1.0
                        }
                        _, dev_step_loss, dev_step_accuracy = sess.run(
                                                                [model.global_step, model.loss, model.accuracy],
                                                                dev_feed_dict)

                        dev_set_batch_time = time.time() - dev_set_batch_time

                        logger.info("dev_step: %d batch_size: %d loss: %f accuracy: %f step_time: %.2f " 
                                    % (dev_set_batch_count, model.batch_size, dev_step_loss, dev_step_accuracy, dev_set_batch_time))
                        dev_results.append((dev_step_loss, dev_step_accuracy, model.batch_size))
                        dev_batches.append(model.batch_size)
                        dev_set_batch_offset += model.batch_size
                        dev_set_batch_count += 1

                    dev_loss = 0
                    dev_accuracy = 0
                    dev_total = 0
                    for (dev_step_loss, dev_step_accuracy, dev_batch_size) in dev_results:
                        dev_loss += dev_step_loss * dev_batch_size
                        dev_accuracy += dev_step_accuracy * dev_batch_size
                        dev_total += dev_batch_size
                    if dev_total>0:
                        dev_loss /= dev_total    
                        dev_accuracy /= dev_total    
                    dev_time = time.time() - dev_time   
                    

                    logger.info("Evaluate: global_step: %d dev_time: %.2f loss: %f accuracy: %f" % (my_global_step, dev_time, dev_loss, dev_accuracy)) 
                                
                counter += 1
                if checkpoints>0 and counter>=checkpoints:
                    logger.info("Checkpoint limit reached")
                    break
                if max_global_step>0 and my_global_step>=max_global_step:
                    logger.info("Max global step limit reached")
                    break


def classify(sentences, model_dir, source_vocab_size, max_sequence_length, num_classes, embedding_size,
               filters, num_filters, dropout_keep_prob, batch_size, learning_rate):
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, model_dir, source_vocab_size, max_sequence_length, num_classes, embedding_size, filters, num_filters,
               dropout_keep_prob, batch_size, learning_rate)
        model.batch_size = 1  # We decode one sentence at a time.
        logger.info("Total Parameters: %s" % model.total_parameters)
    
        # Load vocabularies.
        source_vocab_path = os.path.join(model_dir,
                                     "source%d.vocab" % source_vocab_size)
        target_vocab_path = os.path.join(model_dir,
                                     "target%d.vocab" % num_classes)
        source_vocab, _ = initialize_vocabulary(source_vocab_path)
        _, rev_target_vocab = initialize_vocabulary(target_vocab_path)
    
        # Decode from standard input.
        #sys.stdout.write("> ")
        #sys.stdout.flush()
        #sentence = sys.stdin.readline()
        for sentence_batch in [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]:
            # Get token-ids for the input sentence.
            data_batch = []
            for sentence in sentence_batch:                
                token_ids = sentence_to_token_ids(sentence, source_vocab)
                padded_sequence = pad_sequence(token_ids, max_sequence_length, max(filters))
                data_batch.append((padded_sequence, [0]*num_classes))

            # Get a batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs = model.get_x_y(data_batch)
                        
            feed_dict = {
                         model.input_x: encoder_inputs,
                         model.input_y: decoder_inputs,
                         model.dropout_keep_prob: 1.0
                         }
            
            logits, output_layer = sess.run(
                                        [model.logits, model.output_layer],
                                        feed_dict
                                        )
            
            
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            #logger.debug("classify:logits=%s" % (logits))
            #logger.debug("classify: output_layer=%s" % (output_layer))
            for s, sentence in enumerate(sentence_batch):
                outputs = []
                for i in range(len(output_layer[s])):
                    outputs.append((rev_target_vocab[i], output_layer[s][i]))
                outputs.sort(key=lambda tup: tup[1], reverse=True)    
                logger.info("CLASSIFY input=[%s] output=[%s]" % (sentence, outputs))


def self_test():
    """Test the classification model."""
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        logger.info("Self-test for neural classification model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = TFCNNModel(source_vocab_size=10,
                           sequence_length=8, 
                           embedding_size=3, 
                           num_classes=2, 
                           filters=[1, 2], 
                           num_filters=2, 
                           dropout_keep_prob=1.0, 
                           batch_size=2, 
                           learning_rate=0.3                           )
        
        logger.info("Total Parameters: %s" % model.total_parameters)
        
        sess.run(tf.initialize_all_variables())
    
        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = [
                    ([1, 1, 1, 1, 1, 1, 1, 1], [0.0, 1.0]),
                    ([0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0.0]),
                    ([1, 0, 1, 0, 1, 0, 1, 0], [0.0, 1.0]),
                    ([0, 1, 0, 1, 0, 1, 0, 1], [1.0, 0.0]),
                    ]

        
        loss = 0.0
        for _ in xrange(5):  # Train the fake model for 5 steps.
            encoder_inputs, decoder_inputs = model.get_batch(data_set)     

            feed_dict = {
                         model.input_x: encoder_inputs,
                         model.input_y: decoder_inputs,
                         model.dropout_keep_prob: 1.0
                         }
            
            _, step ,step_loss, accuracy = sess.run(
                                                    [model.train_op, model.global_step, model.loss, model.accuracy],
                                                          feed_dict)
    
            loss += step_loss
        perplexity = math.exp(loss)
        my_global_step = model.global_step.eval()
        logger.info("Self-test: global_step=%d learning_rate=%f perplexity=%f accuracy=%f" 
                            % (my_global_step, model.learning_rate.eval(), perplexity, accuracy))


def main():
    parser = argparse.ArgumentParser(description='CNN Classifier')
    parser.add_argument("--input", help="Input file for training, testing, or decoding")
    parser.add_argument("--model_dir", help="Model directory", default=None)
    parser.add_argument("--word2vec", help="word2vec .bin file", default=None)
    parser.add_argument("--learning_rate", help="Learning rate.", type=float, default=0.001)
    parser.add_argument("--l2_reg_lambda", help="L2 regularization lambda", type=float, default=0.0)
    parser.add_argument("--dropout_keep_prob", help="Dropout keep probability", type=float, default=0.5)
    parser.add_argument("--batch_size", help="Batch size to use during training.", type=int, default=50)
    parser.add_argument("--embedding_size", help="Size of embedding", type=int, default=300)
    parser.add_argument("--filters", help="Comma-separated filter sizes", default='3,4,5')
    parser.add_argument("--num_filters", help="Number of filters per filter size", type=int, default=100)
    parser.add_argument("--source_vocab_size", help="Source vocabulary size.", type=int, default=40000)
    parser.add_argument("--num_classes", help="Number of class labels", type=int, default=None)
    parser.add_argument("--max_data_size", help="Limit on the size of training data (0: no limit).", type=int, default=0)
    parser.add_argument("--sequence_length", help="Max word length", type=int, default=60)
    parser.add_argument("--steps_per_checkpoint", help="How many training steps to do per checkpoint.", type=int, default=200)
    parser.add_argument("--checkpoints", help="How many checkpoints to run", type=int, default=0)
    parser.add_argument("--max_global_step", help="Max number of global steps", type=int, default=0)
    parser.add_argument("--dev_train_split", help="Fraction of examples for dev/validation set", type=float, default=0.1)
    parser.add_argument("--digits", help="Keep (don't normalize) digits.", default=False, action='store_true')
    parser.add_argument("--classify", help="Run classify on input file if this is set to True.", default=False, action='store_true')
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
    logger.info("Tensorflow version: %s" % tensorflow.__version__)

    filters = [int(f) for f in args.filters.split(",")]
    
    if args.self_test:
        logger.info("START SELF-TEST")
        self_test()
    elif args.classify:
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
                sentence = line.strip().split('\t')[0]
                if len(sentence)>0:
                    sentences.append(sentence)

            classify(sentences, 
                     model_dir=args.model_dir, 
                     source_vocab_size=args.source_vocab_size, 
                     max_sequence_length=args.sequence_length, 
                     num_classes=args.num_classes, 
                     embedding_size=args.embedding_size,
                     filters=filters, 
                     num_filters=args.num_filters, 
                     dropout_keep_prob=args.dropout_keep_prob, 
                     batch_size=args.batch_size, 
                     learning_rate=args.learning_rate)                
    else:
        if not os.path.exists(args.model_dir):
            raise ValueError("Model directory %s does not exist.", args.model_dir)
        
        if not os.path.isdir(args.model_dir):
            raise ValueError("Model directory %s is not a directory.", args.model_dir)
        
        if not os.path.exists(args.input) or not os.path.isfile(args.input):
            raise ValueError("Input file %s does not exist.", args.input)
        logger.info("START TRAINING")
        normalize_digits = not args.digits
        train(input_file=args.input, 
              model_dir=args.model_dir, 
              max_data_size=args.max_data_size, 
              dev_train_split=args.dev_train_split,
              steps_per_checkpoint=args.steps_per_checkpoint, 
              source_vocab_size=args.source_vocab_size,  
              max_sequence_length=args.sequence_length, 
              num_classes=args.num_classes, 
              embedding_size=args.embedding_size, 
              filters=filters, 
              num_filters=args.num_filters,
              dropout_keep_prob=args.dropout_keep_prob, 
              batch_size=args.batch_size, 
              learning_rate=args.learning_rate,
              checkpoints=args.checkpoints, 
              max_global_step=args.max_global_step,
              word2vec_path=args.word2vec)


        
        
    logger.info("DONE")    

if __name__ == "__main__":
    main()
