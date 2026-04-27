DATA     := ../rice_data
PLINK_IN := $(DATA)/RiceDiversity_44K_Genotypes_PLINK

CHAIN    := $(DATA)/MSU6_to_IRGSP-1.0.chain
FASTA    := $(DATA)/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa
STEM     := sativas413

PED      := $(PLINK_IN)/$(STEM).ped
MAP      := $(PLINK_IN)/$(STEM).map

VCF_MSU6 := $(PLINK_IN)/$(STEM).vcf
VCF_MSU7 := $(DATA)/$(STEM)_msu7.vcf
VCF_FILT := $(DATA)/$(STEM)_msu7_biallelic.vcf
VCF_OUT  := $(DATA)/$(STEM)_msu7_final.vcf
PGEN     := $(DATA)/$(STEM)_msu7_final.pgen

CROSSMAP  := conda run -n svar CrossMap
PLINK2    := conda run -n svar plink2
BCFTOOLS  := conda run -n svar bcftools

.PHONY: all clean

all: $(PGEN) $(VCF_OUT)

$(VCF_MSU6): $(PED) $(MAP)
	$(PLINK2) --ped $(PED) --map $(MAP) --export vcf --out $(PLINK_IN)/$(STEM)

$(VCF_MSU7): $(VCF_MSU6) $(CHAIN) $(FASTA)
	$(CROSSMAP) vcf $(CHAIN) $< $(FASTA) $@

$(VCF_FILT): $(VCF_MSU7)
	$(BCFTOOLS) view --min-alleles 2 $< -o $@.tmp
	$(BCFTOOLS) sort $@.tmp -o $@
	rm $@.tmp

$(PGEN) $(VCF_OUT) &: $(VCF_FILT) $(FASTA)
	$(PLINK2) --vcf $< --fa $(FASTA) --ref-from-fa force \
	          --make-pgen --export vcf --out $(DATA)/$(STEM)_msu7_final

clean:
	rm -f $(VCF_MSU6) $(VCF_MSU7) $(VCF_FILT) $(VCF_OUT) $(DATA)/$(STEM)_msu7.unmap
	rm -f $(DATA)/$(STEM)_msu7_final.pgen $(DATA)/$(STEM)_msu7_final.psam $(DATA)/$(STEM)_msu7_final.pvar

# ── Equivalence test: bcftools consensus vs dataset.py on-the-fly application ─

TABIX        := conda run -n svar tabix
TEST_SAMPLES := 081215-A05_1 081215-A06_3 081215-A07_4 081215-A08_5 090414-A09_6
TEST_DIR     := test_consensus
VCF_BGZ      := $(VCF_OUT).gz
TEST_FASTAS  := $(patsubst %, $(TEST_DIR)/%.fa, $(TEST_SAMPLES))

.PHONY: test-consensus clean-test
test-consensus: $(TEST_FASTAS)

# bgzip + tabix the final VCF once (bcftools consensus needs an index)
$(VCF_BGZ): $(VCF_OUT)
	$(BCFTOOLS) view $< -O z -o $@
	$(TABIX) -p vcf $@

$(TEST_DIR):
	mkdir -p $@

# One consensus genome per test sample; $* expands to the sample name stem
$(TEST_DIR)/%.fa: $(VCF_BGZ) $(FASTA) | $(TEST_DIR)
	$(BCFTOOLS) consensus -f $(FASTA) -s $* -H 1 $< > $@

clean-test:
	rm -rf $(TEST_DIR) $(VCF_BGZ) $(VCF_BGZ).tbi
