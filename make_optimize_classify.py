#!/usr/bin/python
import argparse
import sys
import re
from tensorflow_classify import basic_tokenizer
from tensorflow_optimize import _DELIMITER
from tensorflow_classify import _EOS

def main():
    parser = argparse.ArgumentParser(description='Make classify training file from optimize training file')
    parser.add_argument("--test", help="Omit class label if this is set to True.", default=False, action='store_true')


    args = parser.parse_args()

    for line in sys.stdin:
        fields = line.strip().split(_DELIMITER)
    
        if len(fields)==3:
            input = []
            awords = basic_tokenizer(fields[0])
            for aword in awords:
                input.append("_%s" % aword)
            input.append(_EOS)
                
            bwords = basic_tokenizer(fields[1])
            for bword in bwords:
                input.append(bword)
            
            if args.test:
                print "%s" % (" ".join(input))                
            else:
                score = float(fields[2])
                label = "NO"
                if score>0.5:
                    label = "YES"
                print "%s\t%s" % (" ".join(input), label)    

if __name__ == "__main__":
    main()
