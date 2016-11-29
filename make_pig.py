#!/usr/bin/python

import sys
import re

VOWEL = re.compile("[AaEeIiOoUu]")
NON_WORD = re.compile("[^a-zA-Z]")

for line in sys.stdin.read().splitlines():
    source = line.lower()

    words = source.split()
    target = ''
    for word in words:
        new_word = word
        if not re.search(NON_WORD, word):            
            head = ''
            tail = ''
            foundvowel=False
            for c in word:
                if re.match(VOWEL, c):
                    foundvowel=True
                if foundvowel:
                    tail += c
                else:
                    head += c
            new_word = tail
            new_word += head
            if len(head)==0:
                new_word += "way"
            else:
                new_word += "ay"
            
        target += new_word
        target += " "   
        
    print "%s\t%s" % (source, target)

