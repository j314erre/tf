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
    parser.add_argument("--save", help="Model save file")
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
    
    model = TFRNNWordModel(vocab_size=6049, batch_size=args.batch_size, epochs=args.epochs)
    
    data, _ = model.read_data(file_path=args.input)

    with  tf.Session() as session:        
        model.train(session, training_data=data, save_path=args.save)
    
    
if __name__ == "__main__":
    main()
   

