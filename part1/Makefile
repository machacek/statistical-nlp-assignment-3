SHELL = /bin/bash

BRILL_DIR = $(shell first_existing /home/mmachace/RULE_BASED_TAGGER_V1.14)
BRILL_TAGGER = $(BRILL_DIR)/Bin_and_Data/tagger

.PHONY: all clean

all: cz.1.T.lexicon cz.1.S.untagged


#test.tagged: lexicon test.corpus test.bigrams lexical.rules contextual.rules
#	$(BRILL_TAGGER) \
		lexicon \
		test.corpus \
		test.bigrams \
		lexical.rules \
		contextual.rules \
		> $@

%.word-tag-list: %.for-unknown.spl
	cat $^ | ./create-word-tag-list >$@

%.big-word-list: %.T.untagged %.S.untagged %.H.untagged
	cat $^ | ./create-word-list >$@

%.for-unknown.spl: %.T.spl
	cat $< | awk 'NR % 2 == 0' > $@

%.for-contextual.spl: %.T.spl
	cat $< | awk 'NR % 2 == 1' > $@

%.untagged: %.spl
	cat $< | ./remove-tags > $@

%.lexicon: %.ptg
	cat $< | ./create-lexicon > $@

%.1.S.ptg: text%2.ptg
	cat $< | tail -n 40000 > $@

%.1.T.ptg: text%2.ptg
	cat $< | head -n -60000 > $@

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
	rm -rf cz.*.ptg en.*.ptg *.spl *.lexicon *.untagged *.big-word-list *.word-tag-list *.big-word-list
