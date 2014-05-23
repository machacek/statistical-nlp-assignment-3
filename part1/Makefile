SHELL = /bin/bash

BRILL_TAGGER = $(shell first_existing /home/mmachace/RULE_BASED_TAGGER_V1.14/Bin_and_Data/tagger)

.PHONY: all clean

all: cz.1.S.spl cz.1.T.spl cz.1.H.spl


test.tagged: lexicon test.corpus test.bigrams lexical.rules contextual.rules
	$(BRILL_TAGGER) \
		lexicon \
		test.corpus \
		test.bigrams \
		lexical.rules \
		contextual.rules \
		> $@



%.1.S.ptg: text%2.ptg
	cat $< \
		| tail -n 40000 \
		> $@

%.1.T.ptg: text%2.ptg
	cat $< \
		| head -n -60000 \
		> $@

%.1.H.ptg: text%2.ptg
	cat $< \
		| tail -n 60000 \
		| head -n 20000 \
		> $@

%.spl: %.ptg
	cat $< \
		| tr "\n" " " \
		| sed "s/ *###\/### */\n/g" \
		| grep -v "^$$" \
		> $@


clean:
	rm -rf cz.*.ptg en.*.ptg *.spl
