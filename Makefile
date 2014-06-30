SHELL = /bin/bash

BRILL_DIR = $(shell readlink -f RULE_BASED_TAGGER_V1.14)
BRILL_TAGGER = $(BRILL_DIR)/Bin_and_Data/tagger

TRAIN_SIZE = 20000000000

.PHONY: all clean RULE_BASED_TAGGER_V1.14

.SECONDARY:

all: en.baseline.results cz.baseline.results en.hmm-supervised.results cz.hmm-supervised.results en.hmm-unsupervised.results cz.hmm-unsupervised.results en.brill.results cz.brill.results

#########################################################
# Baseline experiment                                   #
#########################################################

%.baseline.results: %.fold1.S.baseline.accuracy %.fold2.S.baseline.accuracy %.fold3.S.baseline.accuracy %.fold4.S.baseline.accuracy %.fold5.S.baseline.accuracy
	./results-summary $^ > $@

%.S.baseline.accuracy: %.S.baseline.tagged %.S.ptg
	./measure-accuracy $^ > $@

%.S.baseline.tagged: %.S.ptg %.baseline.model
	cat $< | ./remove-tags | ./decode $*.baseline.model > $@

%.baseline.model: %.T.ptg
	./train-model-baseline $< $@

#########################################################
# Supervised HMM experiment                             #
#########################################################

%.hmm-supervised.results: %.fold1.S.hmm-supervised.accuracy %.fold2.S.hmm-supervised.accuracy %.fold3.S.hmm-supervised.accuracy %.fold4.S.hmm-supervised.accuracy %.fold5.S.hmm-supervised.accuracy
	./results-summary $^ > $@

%.S.hmm-supervised.accuracy: %.S.hmm-supervised.tagged %.S.ptg
	./measure-accuracy $^ > $@

%.S.hmm-supervised.tagged: %.S.ptg %.hmm-supervised.model
	cat $< | ./remove-tags | ./decode $*.hmm-supervised.model > $@

%.hmm-supervised.model: %.T.ptg %.H.ptg
	./train-model-hmm $< \
		--heldout $*.H.ptg \
		--lang $$(echo $* | cut -d. -f1) \
		$@

#########################################################
# Unsupervised HMM experiment                           #
#########################################################

%.hmm-unsupervised.results: %.fold1.S.hmm-unsupervised.accuracy %.fold2.S.hmm-unsupervised.accuracy %.fold3.S.hmm-unsupervised.accuracy %.fold4.S.hmm-unsupervised.accuracy %.fold5.S.hmm-unsupervised.accuracy
	./results-summary $^ > $@

%.S.hmm-unsupervised.accuracy: %.S.hmm-unsupervised.tagged %.S.ptg
	./measure-accuracy $^ > $@

%.S.hmm-unsupervised.tagged: %.S.ptg %.hmm-unsupervised.model
	cat $< | ./remove-tags | ./decode $*.hmm-unsupervised.model > $@

%.hmm-unsupervised.model: %.T.labeled.ptg %.T.unlabeled.ptg %.H.ptg
	./train-model-hmm $< \
		--unlabeled $*.T.unlabeled.ptg \
		--heldout $*.H.ptg \
		--lang $$(echo $* | cut -d. -f1) \
		$@ \
		2> $@.log

%.T.labeled.ptg: %.T.ptg
	cat $< | head -n10000 > $@

%.T.unlabeled.ptg: %.T.ptg
	cat $< | tail -n+10001 | ./remove-tags > $@

#########################################################
# Brill's tagger experiment                             #
#########################################################

#
# Evaluace
#
%.brill.results: %.fold1.S.brill.accuracy %.fold2.S.brill.accuracy %.fold3.S.brill.accuracy %.fold4.S.brill.accuracy %.fold5.S.brill.accuracy
	./results-summary $^ > $@

%.brill.accuracy: %.brill.tagged %.spl
	./measure-accuracy $^ > $@

#
# Tagovani testovaci sady
#
%.S.brill.tagged: %.T.lexicon %.S.untagged %.bigram-list %.unknown-word-rules %.context-rules
	export PATH=$(BRILL_DIR)/Bin_and_Data:$$PATH && \
	$(BRILL_TAGGER) $^ > $@

#
# Trenovani kontextualnich pravidel
#
%.context-rules: %.T.part2.spl %.T.part2.dummy-tagged %.T.part1.lexicon RULE_BASED_TAGGER_V1.14
	$(BRILL_DIR)/Bin_and_Data/contextual-rule-learn \
		$*.T.part2.spl \
		$*.T.part2.dummy-tagged \
		$@ \
		$*.T.part1.lexicon ; \
	echo "Exit code: $$? (contextual-rule-learn)" | tee $*.exit-code

%.T.part2.dummy-tagged: %.T.part1.lexicon %.T.part2.untagged %.bigram-list %.unknown-word-rules %.word-list RULE_BASED_TAGGER_V1.14
	export PATH=$(BRILL_DIR)/Bin_and_Data:$$PATH && \
	tagger \
		$*.T.part1.lexicon \
		$*.T.part2.untagged \
		$*.bigram-list \
		$*.unknown-word-rules \
		/dev/null \
		-w $*.word-list -S \
		> $@

#
# Trenovani pravidel pro neznama slova
#
%.unknown-word-rules: %.word-list %.T.part1.word-tag-list %.bigram-list
	perl $(BRILL_DIR)/Learner_Code/unknown-lexical-learn.prl \
			$^ 300 $@

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
# Kompilace Brill's tagger
#
RULE_BASED_TAGGER_V1.14:
	$(MAKE) -C $@

#########################################################
# Ostatni pomocne cile                                  #
#########################################################

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

# Fold 1
%.fold1.S.ptg: text%2.ptg
	cat $< | tail -n 40000 > $@

%.fold1.T.ptg: text%2.ptg
	cat $< \
		| head -n -60000 \
		| head -n$(TRAIN_SIZE) \
		> $@

%.fold1.H.ptg: text%2.ptg
	cat $< \
		| tail -n 60000 \
		| head -n 20000 \
		> $@

# Fold 2
%.fold2.S.ptg: text%2.ptg
	cat $< | head -n 40000 > $@

%.fold2.T.ptg: text%2.ptg
	cat $< \
		| tail -n +60001 \
		| head -n$(TRAIN_SIZE)  \
		> $@

%.fold2.H.ptg: text%2.ptg
	cat $< \
		| head -n 60000 \
		| tail -n 20000 \
		> $@

# Fold 3
%.fold3.S.ptg: text%2.ptg
	cat $< \
		| tail -n +60001 \
		| head -n 40000 > $@

%.fold3.T.ptg: text%2.ptg
	cat $< | head -n 60000 > $@; \
	cat $< | tail -n +120001 >> $@; \
	cat $@ | head -n$(TRAIN_SIZE) | sponge $@

%.fold3.H.ptg: text%2.ptg
	cat $< \
		| tail -n +60001 \
		| head -n 60000 \
		| tail -n 20000 \
		> $@

# Fold 4
%.fold4.S.ptg: text%2.ptg
	cat $< \
		| tail -n +120001 \
		| head -n 40000 > $@

%.fold4.T.ptg: text%2.ptg
	cat $< | head -n 120000 > $@; \
	cat $< | tail -n +180001 >> $@; \
	cat $@ | head -n$(TRAIN_SIZE) | sponge $@

%.fold4.H.ptg: text%2.ptg
	cat $< \
		| tail -n +120001 \
		| head -n 60000 \
		| tail -n 20000 \
		> $@

# Fold 5
%.fold5.S.ptg: text%2.ptg
	cat $< \
		| tail -n +180001 \
		| head -n 40000 > $@

%.fold5.T.ptg: text%2.ptg
	cat $< | head -n 180000 > $@; \
	cat $< | tail -n +240001 >> $@; \
	cat $@ | head -n$(TRAIN_SIZE) | sponge $@

%.fold5.H.ptg: text%2.ptg
	cat $< \
		| tail -n +180001 \
		| head -n 60000 \
		| tail -n 20000 \
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
		*.context-rules \
		*.dummy-tagged \
		*.tagged \
		*.accuracy \
		*.exit-code \
		*.results \
		*.model \
		*.log \
		run_make.sh.*
