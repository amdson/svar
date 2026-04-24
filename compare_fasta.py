"""
Compare Oryza_sativa.IRGSP-1.0.dna_rm.toplevel.fa vs GCA_rice.fasta.

Reports:
  - Sequence counts and IDs unique to each file
  - Per-chromosome length comparison (matched by chromosome number)
  - N-base (hard-masked) and soft-masked counts per sequence
  - GC content per sequence
"""

import re
from Bio import SeqIO

IRGSP = "/home/adickson/rice_data/Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa"
GCA   = "/home/adickson/rice_data/GCA_rice.fasta"

# AP014957 → chr1, AP014958 → chr2, ...
GCA_ACCESSION_TO_CHR = {f"AP01495{d}": str(i) for i, d in enumerate("789", start=1)}
GCA_ACCESSION_TO_CHR.update({
    "AP014960": "4", "AP014961": "5", "AP014962": "6",
    "AP014963": "7", "AP014964": "8", "AP014965": "9",
    "AP014966": "10", "AP014967": "11", "AP014968": "12",
})


def parse_gca_chr(seq_id: str) -> str | None:
    """Extract accession from 'ENA|AP014957|AP014957.1' and map to chr number."""
    m = re.search(r"(AP\d+)", seq_id)
    if m:
        acc = m.group(1).split(".")[0]
        return GCA_ACCESSION_TO_CHR.get(acc)
    return None


def seq_stats(seq) -> dict:
    s = str(seq).upper()
    length = len(s)
    n_count = s.count("N")
    gc = (s.count("G") + s.count("C"))
    soft = sum(1 for c in str(seq) if c.islower())
    return {
        "length": length,
        "N_count": n_count,
        "soft_masked": soft,
        "gc_pct": round(100 * gc / (length - n_count), 2) if length > n_count else 0.0,
    }


print("=" * 70)
print("Loading IRGSP ...")
irgsp_seqs = {r.id: r for r in SeqIO.parse(IRGSP, "fasta")}
print(f"  {len(irgsp_seqs)} sequences")

print("Loading GCA ...")
gca_seqs = {r.id: r for r in SeqIO.parse(GCA, "fasta")}
print(f"  {len(gca_seqs)} sequences")

# --- sequence set differences ---
print("\n" + "=" * 70)
print("SEQUENCES ONLY IN IRGSP (not in GCA):")
irgsp_only = sorted(set(irgsp_seqs) - set(gca_seqs))
for sid in irgsp_only:
    st = seq_stats(irgsp_seqs[sid].seq)
    print(f"  {sid:30s}  len={st['length']:>12,}  N={st['N_count']:>10,}  "
          f"soft={st['soft_masked']:>10,}  GC={st['gc_pct']}%")

print(f"\nSEQUENCES ONLY IN GCA (not in IRGSP):")
gca_only = sorted(set(gca_seqs) - set(irgsp_seqs))
for sid in gca_only:
    st = seq_stats(gca_seqs[sid].seq)
    print(f"  {sid:60s}  len={st['length']:>12,}")

# --- per-chromosome comparison ---
print("\n" + "=" * 70)
print(f"{'CHR':<5} {'IRGSP_ID':<8} {'GCA_ID':<30} {'IRGSP_len':>12} {'GCA_len':>12} "
      f"{'ΔLEN':>8} {'IRGSP_N':>10} {'GCA_N':>10} {'IRGSP_soft':>12} {'GCA_soft':>12} "
      f"{'IRGSP_GC%':>10} {'GCA_GC%':>10}")
print("-" * 140)

chr_nums = [str(i) for i in range(1, 13)]
for chrom in chr_nums:
    irgsp_rec = irgsp_seqs.get(chrom)
    gca_rec   = next((r for r in gca_seqs.values() if parse_gca_chr(r.id) == chrom), None)

    if irgsp_rec is None or gca_rec is None:
        print(f"  chr{chrom}: missing in {'IRGSP' if irgsp_rec is None else 'GCA'}")
        continue

    i_st = seq_stats(irgsp_rec.seq)
    g_st = seq_stats(gca_rec.seq)
    delta = i_st["length"] - g_st["length"]
    flag = " <-- MISMATCH" if delta != 0 else ""

    print(f"{chrom:<5} {'chr'+chrom:<8} {gca_rec.id:<30} "
          f"{i_st['length']:>12,} {g_st['length']:>12,} {delta:>+8,} "
          f"{i_st['N_count']:>10,} {g_st['N_count']:>10,} "
          f"{i_st['soft_masked']:>12,} {g_st['soft_masked']:>12,} "
          f"{i_st['gc_pct']:>10} {g_st['gc_pct']:>10}{flag}")

print("=" * 70)
print("Done.")
