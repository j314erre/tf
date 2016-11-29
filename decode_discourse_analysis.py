import sys
import re
import argparse
import logging
from __builtin__ import reversed

_DELIMITER = '|'
_OUTPUT_EXTRACT_GROUP = re.compile("output=\[(.*?)\]\s*$")
_DEXIS_EXTRACT_GROUP = re.compile(r'\b(here|there|then|now|later|next|this|that)\b')
_PRONOUN_EXTRACT_GROUP = re.compile(r'\b(he|him|his|she|her|hers|they|them|their|theirs)\b')
_CUE_PHRASE_EXTRACT_GROUP = re.compile(r'^(so|after all|in addition|furthermore|therefore|thus|also|but|however|otherwise|although|if|then)\b')

logger = logging.getLogger("discourse_analysis")


def print_histogram(hist):
    sort_hist = sorted(hist.items(), key=lambda value: value[1], reverse=True)
    for (key, value) in sort_hist:
        logger.info("%s: %s" % (key, value))

def extract(regex, text):
    result = regex.search(text)
    
    if result:
        return result.group(1)
    
    return None
    
def main():
    parser = argparse.ArgumentParser(description='Discourse Analysis')
    parser.add_argument("--limit", help="Number of examples to analyze", type=int, default=0)


    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(args)
    
    counter = 0
    dexis = 0.0
    pronoun = 0.0
    cue_phrase = 0.0
    dexis_dict = {}
    pronoun_dict = {}
    cue_phrase_dict = {}
    for line in sys.stdin.read().splitlines():
        output = extract(_OUTPUT_EXTRACT_GROUP, line.strip())
        if output:

            dexis_marker = extract(_DEXIS_EXTRACT_GROUP, output)
            if dexis_marker:
                #logger.info(line)
                dexis += 1.0
                if not dexis_marker in dexis_dict:
                    dexis_dict[dexis_marker] = 1
                else:
                    dexis_dict[dexis_marker] += 1        
                            
            pronoun_marker = extract(_PRONOUN_EXTRACT_GROUP ,output)
            if pronoun_marker:
                #logger.info(text)
                pronoun += 1.0
                if not pronoun_marker in pronoun_dict:
                    pronoun_dict[pronoun_marker] = 1
                else:
                    pronoun_dict[pronoun_marker] += 1        
                
            cue_phrase_marker = extract(_CUE_PHRASE_EXTRACT_GROUP ,output)                
            if cue_phrase_marker:
                #logger.info(text)
                cue_phrase += 1.0
                if not cue_phrase_marker in cue_phrase_dict:
                    cue_phrase_dict[cue_phrase_marker] = 1
                else:
                    cue_phrase_dict[cue_phrase_marker] += 1        
            
            
            
            counter += 1        
            if args.limit and counter >= args.limit:
                break
        
    logger.info("COUNTER=%s" % (counter))
    logger.info("DEXIS=%s" % (dexis/counter))
    print_histogram(dexis_dict)
    logger.info("PRONOUN=%s" % (pronoun/counter))
    print_histogram(pronoun_dict)
    logger.info("CUE_PHRASE=%s" % (cue_phrase/counter))
    print_histogram(cue_phrase_dict)
    
    logger.info("DONE")                


if __name__ == "__main__":
    main()

