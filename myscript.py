import argparse
import logging

# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
logger = logging.getLogger("tensorflow_rnn_word")


def main():
    parser = argparse.ArgumentParser(description='LSTM Character Model')
    parser.add_argument("--input", help="Input file")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(args)


if __name__ == "__main__":
    main()

