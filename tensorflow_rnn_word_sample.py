import os
import collections
import argparse
import logging
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow_rnn_word import TFRNNWordModel

logger = logging.getLogger("tensorflow_rnn_word")
                  
def main():
    parser = argparse.ArgumentParser(description='RNN Word Model')
    parser.add_argument("--input", help="Input file")
    parser.add_argument("--save", help="Save model file")
    parser.add_argument("--batch_size", help="Mini batch size", type=int, default=32)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1)

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(args)
    
    model = TFRNNWordModel(vocab_size=6049, batch_size=1, num_steps=1)
    
    data, _ = model.read_data(file_path=args.input)
    
    save_path, filename = os.path.split(args.save)
    
    with  tf.Session() as session:        
    
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_path)
        print ckpt
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            text, score = model.sample(session, num=20, init='the')
            logger.debug("checkpoint: %0.3f %s" % (score, text))    
    
if __name__ == "__main__":
    main()
   

