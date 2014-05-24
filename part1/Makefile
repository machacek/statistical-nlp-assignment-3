SHELL = /bin/bash

BRILL_DIR = $(shell first_existing /home/mmachace/RULE_BASED_TAGGER_V1.14)
BRILL_TAGGER = $(BRILL_DIR)/Bin_and_Data/tagger

.PHONY: all clean

.SECONDARY:

all: cz.fold1.S.tagged

#
# Evaluace
#
# %.S.accuracy: %.S.tagged %.S.spl

#
# Tagovani testovaci sady
#
%.S.tagged: %.T.lexicon %.S.untagged %.bigram-list %.unknown-word-rules %.context-rules
	$(BRILL_TAGGER) $^ > $@

#
# Trenovani kontextualnich pravidel
#
%.context-rules: %.T.part2.spl %.T.part2.dummy-tagged %.T.part1.lexicon
	$(BRILL_DIR)/Bin_and_Data/contextual-rule-learn \
		$*.T.part2.spl \
		$*.T.part2.dummy-tagged \
		$@ \
		$*.T.part1.lexicon

%.T.part2.dummy-tagged: %.T.part1.lexicon %.T.part2.untagged %.bigram-list %.unknown-word-rules %.word-list
	dir=$$PWD; cd $(BRILL_DIR)/Bin_and_Data; \
	tagger \
		$$dir/$*.T.part1.lexicon \
		$$dir/$*.T.part2.untagged \
		$$dir/$*.bigram-list \
		$$dir/$*.unknown-word-rules \
		/dev/null \
		-w $$dir/$*.word-list -S \
		> $$dir/$@ ; \
	cd $$dir


#
# Trenovani pravidel pro neznama slova
#
%.unknown-word-rules: %.word-list %.T.part1.word-tag-list %.bigram-list
	perl $(BRILL_DIR)/Learner_Code/unknown-lexical-learn.prl \
		$^ 100 $@

#
# Vytvoreni pomocnych souboru pro trenovani pravidel pro neznama slova
# (k seznamu slov a bigramu pouzivame vsechna neanotovana data)
#
%.bigram-list: %.T.untagged %.S.untagged %.H.untagged
	cat $^ \
		| perl $(BRILL_DIR)/Utilities/bigram-generate.prl \
		| cut "-d " -f1,2 \
		> $@

%.word-list: %.T.untagged %.S.untagged %.H.untagged
	cat $^ | ./create-word-list >$@

#
# Rozdeleni trenovaci mnoziny (T) na dve trenovaci mnoziny
#
%.T.part1.spl: %.T.spl
	cat $< | awk 'NR % 2 == 0' > $@

%.T.part2.spl: %.T.spl
	cat $< | awk 'NR % 2 == 1' > $@

#
# Genericke pravidlo pro vytvoreni seznamu nejcastejsich dvojic word/tag
#
%.word-tag-list: %.spl
	cat $^ | ./create-word-tag-list > $@

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
# Zde jsou definovana rozdeleni na mnoziny S T H pro jednotlive foldy
#
%.fold1.S.ptg: text%2.ptg
	cat $< | tail -n 40000 > $@

%.fold1.T.ptg: text%2.ptg
	cat $< | head -n -60000 | head 10000 > $@

%.fold1.H.ptg: text%2.ptg
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
		*.word-list \
		*.bigram-list \
		*.unknown-word-rules \
		*.context-rules
