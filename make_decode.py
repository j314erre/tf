import argparse
import logging

_DELIMITER = '\t'
logger = logging.getLogger("make_decode")


def main():
    parser = argparse.ArgumentParser(description='Make decoder file')
    parser.add_argument("--vocab", help="Vocab file")
    parser.add_argument("--ids", help="Ids file")
    parser.add_argument("--output", help="Output file")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(args)
    
    with open(args.vocab, "r") as vocab_file:
        with open(args.ids, "r") as ids_file:
            with open(args.output, "w") as output_file:
                
                
                logger.info("Loading vocabulary file: %s" % args.vocab)
                vocab_words = []
                for vocab_line in vocab_file:
                    word = vocab_line.strip()
                    vocab_words.append(word)
                    
                logger.info("Vocab size: %s" % len(vocab_words))
    
                logger.info("Reading ids file: %s" % args.ids)
                counter = 0
                for ids_line in ids_file:
                    counter += 1
                    if counter % 10000 == 0:
                        logger.info("    reading %s line %d" % (args.ids, counter))
                    output_utterances = []
                    utterances = ids_line.strip().split(_DELIMITER)
                    for utterance in utterances:
                        utterance_ids = [int(x) for x in utterance.strip().split()]
                        utterance_words = []
                        for utterance_id in utterance_ids:
                            utterance_word = vocab_words[utterance_id]
                            utterance_words.append(utterance_word)
                            
                        output_utterances.append(" ".join(utterance_words))
                        
                    output_file.write(_DELIMITER.join(output_utterances) + "\n")

                output_file.close()
    logger.info("DONE")                


if __name__ == "__main__":
    main()

