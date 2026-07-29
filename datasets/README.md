# datasets/

Standardized, reproducible recipes for every dataset this project trains on.
One subdirectory per dataset, each a self-contained Makefile that **downloads →
converts → sanity-checks** the data. Only the recipes and provenance notes are
committed; the (large, often license-restricted) data is rebuilt with `make`.

```
datasets/
├── common.mk        # shared: data location + conda tool wrappers
├── .gitignore       # tracks ONLY Makefile / *.mk / *.md — never the data
├── rice/            # Oryza sativa, 44K RiceDiversity panel (complete)
│   ├── Makefile
│   └── SOURCES.md
├── soy/             # Glycine max, USDA SoySNP50K panel + SoyDNGP phenotypes (complete)
│   ├── Makefile
│   └── SOURCES.md
├── wheat/           # Triticum aestivum, CIMMYT SeeDs Iranian landraces, DArTSeq (complete)
│   ├── Makefile
│   └── SOURCES.md
└── arabidopsis/     # Arabidopsis thaliana, 1001 Genomes panel (TAIR10)
    ├── Makefile
    └── SOURCES.md
```

## The standard process

Every dataset follows the same four stages, expressed as Make targets:

1. **download** — fetch the reference genome, the SNP genotypes (+ any remap
   chain), and the trait/phenotype table; decompress and index.
2. **build** — convert genotypes to VCF, remap coordinates to the reference
   assembly if needed, filter to biallelic + QC, and force the VCF REF allele to
   match the reference genome (`plink2 --ref-from-fa force`).
3. **check** — sanity-check that SNP REF alleles (and flanking sequence, where a
   flanking table exists) agree with the reference genome.
4. **clean / distclean** — drop intermediates, or everything (it's re-buildable).

## Where the data goes

Built data lands under cluster scratch, **off the home quota**:
`$(DATA_ROOT)/<dataset>`, where `DATA_ROOT` defaults to
`$SVAR_SCRATCH/datasets` (`/90daydata/small_grains/andrew.dickson/datasets`).
Scratch is 90-day space — that's fine, because `make` rebuilds anything purged.

Override the location per-invocation:

```bash
make DATA_ROOT=/some/other/root        # whole tree elsewhere
make SVAR_SCRATCH=/different/scratch   # different scratch base
```

## Bootstrapping on a new server

```bash
git clone <repo> && cd svar
conda env create -f environment.yml && conda activate svar   # plink2, CrossMap, bcftools, samtools, tabix
cd datasets/rice
make            # download + build everything (into $DATA_ROOT/rice)
make check      # verify REF alleles against the reference genome
```

That's the whole point of committing only Makefiles + notes: a fresh checkout
plus `make` reproduces the dataset anywhere, with no data sync required.

## Adding a dataset

Copy an existing dataset dir, then:

1. Set `DATASET := <name>` at the top of the Makefile and `include ../common.mk`.
2. Fill in the download URLs (record them in `SOURCES.md` with provenance:
   assembly/coordinate version, release, license, retrieval date).
3. Wire the `build` graph to your genotype format and coordinate system.
4. Point `check` at a sanity test for REF-allele/genome agreement.

> **Note:** the existing rice data in `~/rice_data` predates this layout and is
> left in place (`crop_embed/data/coords.py` still points there). The rice
> Makefile here faithfully reproduces that pipeline into the new scratch
> location; it does not move or depend on `~/rice_data`.
