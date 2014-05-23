SHELL = /bin/bash

BRILL_TAGGER = $(shell first_existing /home/mmachace/RULE_BASED_TAGGER_V1.14/Bin_and_Data/tagger)

.PHONY: all clean

all: test.tagged


test.tagged: lexicon test.corpus test.bigrams lexical.rules contextual.rules
	$(BRILL_TAGGER) \
		lexicon \
		test.corpus \
		test.bigrams \
		lexical.rules \
		contextual.rules \
		> $@
	

clean:
	rm -rf extracted.segments source.parsed 
