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

#
# Trenovani pravidel pro neznama slova
#
%.unknown-rules: %.big-word-list %.word-tag-list %.bigrams
	perl $(BRILL_DIR)/Learner_Code/unknown-lexical-learn.prl \
		$^ 300 $@

#
# Vytvoreni pomocnych souboru pro trenovani pravidel pro neznama slova
# (k seznamu slov a bigramu pouzivame vsechna neanotovana data)
#
%.word-tag-list: %.for-unknown.spl
	cat $^ | ./create-word-tag-list >$@

%.bigrams: %.T.untagged %.S.untagged %.H.untagged
	cat $^ \
		| perl $(BRILL_DIR)/Utilities/bigram-generate.prl \
		| cut "-d " -f1,2 \
		> $@

%.big-word-list: %.T.untagged %.S.untagged %.H.untagged
	cat $^ | ./create-word-list >$@

#
# Rozdeleni trenovaci mnoziny (T) na dve trenovaci mnoziny
# (Pozdeji prejmenovat)
#
%.for-unknown.spl: %.T.spl
	cat $< | awk 'NR % 2 == 0' > $@

%.for-contextual.spl: %.T.spl
	cat $< | awk 'NR % 2 == 1' > $@

#
# Genericke pravidlo pro odstraneni tagu z formatu jedna veta na radek
#
%.untagged: %.spl
	cat $< | ./remove-tags > $@

#
# Genericke pravidlo pro vytvoreni lexiconu
#
%.lexicon: %.spl
	cat $< | ./create-lexicon > $@

#
# Zde jsou definovana rozdeleni na mnoziny S T H pro jednotlive behy
#
%.1.S.ptg: text%2.ptg
	cat $< | tail -n 40000 > $@

%.1.T.ptg: text%2.ptg
	cat $< | head -n -60000 > $@

%.1.H.ptg: text%2.ptg
	cat $< \
		| tail -n 60000 \
		| head -n 20000 \
		> $@

#
# Genericke pravidlo pro prevod do formatu jedna veta na radek
#
%.spl: %.ptg
	cat $< \
		| tr "\n" " " \
		| sed "s/ *###\/### */\n/g" \
		| grep -v "^$$" \
		> $@

#
# Smazani nepotrebnych souboru
#
clean:
	rm -rf \
		cz.*.ptg \
		en.*.ptg \
		*.spl \
		*.lexicon \
		*.untagged \
		*.big-word-list \
		*.word-tag-list \
		*.big-word-list \
		*.bigrams \
		*.unknown-rules
