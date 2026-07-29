#!/usr/bin/env python
"""
wheat_validate_project.py — stage 3 of the wheat DArTSeq build.

Given the tag->genome alignment (aln.bam) and tag metadata, validate the mapping
and project each SNP to a genome coordinate. Keeps only uniquely-mapped
(MAPQ>=min-mapq), high-identity (flanking identity>=min-ident), biallelic-concordant
markers (genome base is one of the marker's two DArT alleles). Writes
marker_coords.tsv: fid, AlleleID, CloneID, chrom, pos1, strand, ref_fwd, alt_fwd,
cs_allele, mapq, identity.
"""
import argparse, pysam

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheat-dir", required=True)
    ap.add_argument("--genome", required=True, help="reference FASTA (faidx'd)")
    ap.add_argument("--min-mapq", type=int, default=30)
    ap.add_argument("--min-ident", type=float, default=0.90)
    a = ap.parse_args()

    meta = {}
    with open(f"{a.wheat_dir}/tags_meta.tsv") as fh:
        next(fh)
        for line in fh:
            fid, allele_id, clone_id, snp_pos, snp, tag_len = line.rstrip("\n").split("\t")
            base = snp.split(":", 1)[1] if ":" in snp else snp
            ref, alt = base.split(">") if ">" in base else (base, base)
            meta[fid] = (allele_id, clone_id, int(snp_pos), ref, alt)

    fasta = pysam.FastaFile(a.genome)
    bam = pysam.AlignmentFile(f"{a.wheat_dir}/aln.bam", "rb")
    n_uni = n_placed = n_biallelic = n_ref = n_alt = n_good = n_written = 0
    ident_sum = 0.0
    out = open(f"{a.wheat_dir}/marker_coords.tsv", "w")
    out.write("fid\tAlleleID\tCloneID\tchrom\tpos1\tstrand\tref_fwd\talt_fwd\tcs_allele\tmapq\tidentity\n")
    for r in bam.fetch(until_eof=True):
        if r.is_secondary or r.is_supplementary or r.is_unmapped or r.mapping_quality < a.min_mapq:
            continue
        n_uni += 1
        allele_id, clone_id, sp, ref, alt = meta[r.query_name]
        qseq = r.query_sequence; L = len(qseq)
        refspan = fasta.fetch(r.reference_name, r.reference_start, r.reference_end).upper()
        rs = r.reference_start
        qi = sp if not r.is_reverse else (L - 1 - sp)
        matches = aligned = 0; refpos_snp = None
        for qp, rp in r.get_aligned_pairs():
            if qp is not None and rp is not None:
                aligned += 1
                if qseq[qp].upper() == refspan[rp - rs]:
                    matches += 1
            if qp == qi:
                refpos_snp = rp
        identity = matches / aligned if aligned else 0.0
        ident_sum += identity
        if identity >= a.min_ident:
            n_good += 1
        if refpos_snp is None:
            continue
        n_placed += 1
        gbase = refspan[refpos_snp - rs]
        ref_fwd = ref if not r.is_reverse else ref.translate(COMP)
        alt_fwd = alt if not r.is_reverse else alt.translate(COMP)
        if gbase == ref_fwd: n_ref += 1
        elif gbase == alt_fwd: n_alt += 1
        biallelic = gbase in (ref_fwd, alt_fwd)
        if biallelic: n_biallelic += 1
        if biallelic and identity >= a.min_ident:
            n_written += 1
            strand = "-" if r.is_reverse else "+"
            out.write(f"{r.query_name}\t{allele_id}\t{clone_id}\t{r.reference_name}\t"
                      f"{refpos_snp+1}\t{strand}\t{ref_fwd}\t{alt_fwd}\t{gbase}\t"
                      f"{r.mapping_quality}\t{identity:.3f}\n")
    out.close()
    print(f"[validate] uniquely mapped (MAPQ>={a.min_mapq}): {n_uni}")
    print(f"[validate] mean flanking identity: {ident_sum/n_uni:.3f}; identity>={a.min_ident}: {n_good}")
    print(f"[validate] SNP placeable: {n_placed}; biallelic-concordant: {n_biallelic} "
          f"(ref {n_ref}/alt {n_alt})")
    print(f"[validate] wrote {n_written} markers -> {a.wheat_dir}/marker_coords.tsv")

if __name__ == "__main__":
    main()
