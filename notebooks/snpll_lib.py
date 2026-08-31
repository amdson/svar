"""snpll_lib.py — prep + scoring for the SNP-level log-likelihood benchmark.

Single source of truth imported by BOTH conda envs:
  * svar     env : prep (pysam/pyfaidx), AgroNT, Carbon
  * plantcad env : PlantCaduceus (needs mamba_ssm)
All heavy imports are lazy (inside the branch that needs them) so importing this
module never pulls mamba into the svar env or transformers-carbon into plantcad.

Method
------
For each SNP we want the model's distribution over {A,C,G,T} at the SNP position,
in reference context (the 5 non-SNP bases of the SNP's 6-mer = reference):
  * k-mer models (agront masked / carbon causal): read the logits for the SNP's
    6-mer token (agront: mask it; carbon: read the preceding position, causal),
    restrict the vocab to the 4 six-mers that equal the reference 6-mer except at
    the SNP offset, and renormalize -> 4-way nucleotide distribution.
  * plantcad (single-base masked): mask the SNP base, read the native 4-way dist.
Empirical target = alt-carrier frequency across the dataset's samples (1 if a
sample carries the alt allele, else 0/missing) — the codebase gt_alts convention.
"""
import numpy as np, torch, torch.nn.functional as F

WIN, HALF = 510, 255          # 510 = 85 clean 6-mers; SNP centered at offset 255
ACGT = set("ACGT")
BASES = "ACGT"


# ============================================================== PREP ==========
def prep_species(species, fasta_path, vcf_path, out, n=5000, seed=42):
    """Reservoir-sample n eligible SNPs, compute alt-carrier freq, extract the
    reference window. Saves a plain-typed dict for the scoring stage."""
    import pysam
    from pyfaidx import Fasta
    rng = np.random.default_rng(seed)
    fa = Fasta(fasta_path)
    cname = {int(k): k for k in fa.keys() if k.isdigit()}
    clen = {c: len(fa[k]) for c, k in cname.items()}
    vf = pysam.VariantFile(vcf_path)
    n_samples = len(vf.header.samples)
    print(f"[{species}] samples={n_samples} chroms={sorted(cname)}", flush=True)

    K = int(n * 1.15)
    res, seen = [], 0
    for rec in vf:
        try:
            chrom = int(rec.chrom)
        except (ValueError, TypeError):
            continue
        if chrom not in cname or not rec.alts or len(rec.alts) != 1:
            continue
        ref, alt = rec.ref, rec.alts[0]
        if ref is None or alt is None or len(ref) != 1 or len(alt) != 1:
            continue
        ref, alt = ref.upper(), alt.upper()
        if ref not in ACGT or alt not in ACGT:
            continue
        pos0 = rec.pos - 1
        if pos0 - HALF < 0 or pos0 + HALF > clen[chrom]:
            continue
        seen += 1
        if len(res) < K:
            keep = len(res)
        else:
            j = int(rng.integers(0, seen)); keep = j if j < K else -1
        if keep < 0:
            continue
        carriers = sum(1 for s in rec.samples.values()
                       if (g := s.get("GT")) is not None and any(x not in (0, None) for x in g))
        entry = {"chrom": chrom, "pos0": pos0, "ref": ref, "alt": alt,
                 "alt_freq": carriers / n_samples}
        if len(res) < K:
            res.append(entry)
        else:
            res[keep] = entry
        if seen % 200000 == 0:
            print(f"  scanned {seen}, reservoir {len(res)}", flush=True)

    ref_seqs, ref_c, alt_c, af, chroms, poss = [], [], [], [], [], []
    mism = 0
    for e in res:
        if len(ref_seqs) >= n:
            break
        a = e["pos0"] - HALF
        seq = str(fa[cname[e["chrom"]]][a:a + WIN]).upper()
        if len(seq) != WIN or "N" in seq:
            continue
        if seq[HALF] != e["ref"]:
            mism += 1; continue
        ref_seqs.append(seq); ref_c.append(e["ref"]); alt_c.append(e["alt"])
        af.append(e["alt_freq"]); chroms.append(e["chrom"]); poss.append(e["pos0"])
    print(f"[{species}] final windows={len(ref_seqs)} ref-mismatch-skipped={mism}", flush=True)
    torch.save({"species": species, "win": WIN, "offset": HALF,
                "ref_seq": ref_seqs, "ref_char": ref_c, "alt_char": alt_c,
                "alt_freq": np.array(af, np.float32), "chrom": chroms, "pos0": poss}, out)
    return out


# ============================================================== SCORING =======
def score_model(model, prep_path, out, device="cuda:0", carbon_size="500M", batch_size=32):
    """Score one model over one prep file; saves p_ref/p_alt/alt_freq."""
    dev = torch.device(device)
    bidx = {b: i for i, b in enumerate(BASES)}
    d = torch.load(prep_path, map_location="cpu", weights_only=False)
    seqs, off = d["ref_seq"], d["offset"]
    ref_c, alt_c, alt_freq = d["ref_char"], d["alt_char"], d["alt_freq"]
    N = len(seqs)
    tk = 1 + off // 6                      # SNP 6-mer token index (cls/<dna> prefix=1)
    k6s = (off // 6) * 6
    o_in = off - k6s
    print(f"[{model}] N={N} tk={tk} o_in={o_in} (carbon_size={carbon_size})", flush=True)

    if model == "agront":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        name = "InstaDeepAI/agro-nucleotide-transformer-1b"
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForMaskedLM.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()
        vocab = tok.get_vocab(); kmer_id = lambda s: vocab[s]; mask_id = tok.mask_token_id
    elif model == "carbon":
        from CARBON_modules import load_carbon_lm
        m, tok = load_carbon_lm(repo_id=f"HuggingFaceBio/Carbon-{carbon_size}", device=dev); m.eval()
        kmer_id = lambda s: tok.convert_tokens_to_ids(s)
    else:
        from PlantCAD_modules import load_plantcad_mlm
        m, tok = load_plantcad_mlm(device=dev); m.eval()
        v = tok.get_vocab(); base_ids = torch.tensor([v[b.lower()] for b in BASES]); mask_id = tok.mask_token_id

    @torch.no_grad()
    def batch(bseqs, brefc, baltc):
        if model == "plantcad":
            ids = tok(bseqs, return_tensors="pt", padding=False)["input_ids"].to(dev)
            ids[:, off] = mask_id
            lg = m(input_ids=ids).logits.float()[:, off, :]
            four = lg[:, base_ids.to(dev)]
        else:
            if model == "carbon":
                enc = tok(["<dna>" + s for s in bseqs], return_tensors="pt", add_special_tokens=False)
                ids = enc["input_ids"].to(dev)
                lg = m(input_ids=ids).logits.float()[:, tk - 1, :]     # causal: predict tk from tk-1
            else:
                enc = tok(bseqs, return_tensors="pt"); ids = enc["input_ids"].clone()
                ids[:, tk] = mask_id; ids = ids.to(dev)
                lg = m(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()[:, tk, :]
            cand = torch.empty(len(bseqs), 4, dtype=torch.long)
            for r, s in enumerate(bseqs):
                ref6 = s[k6s:k6s + 6]
                for j, b in enumerate(BASES):
                    cand[r, j] = kmer_id(ref6[:o_in] + b + ref6[o_in + 1:])
            four = torch.gather(lg, 1, cand.to(dev))
        p = F.softmax(four, dim=1).cpu().numpy()
        pr = np.array([p[i, bidx[brefc[i]]] for i in range(len(bseqs))])
        pa = np.array([p[i, bidx[baltc[i]]] for i in range(len(bseqs))])
        return pr, pa

    p_ref = np.empty(N, np.float32); p_alt = np.empty(N, np.float32)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        pr, pa = batch(seqs[s:e], ref_c[s:e], alt_c[s:e]); p_ref[s:e] = pr; p_alt[s:e] = pa
    torch.save({"model": model, "species": d["species"], "carbon_size": carbon_size,
                "p_ref": p_ref, "p_alt": p_alt, "alt_freq": alt_freq,
                "ref_char": ref_c, "alt_char": alt_c}, out)
    r = float(np.corrcoef(p_alt, alt_freq)[0, 1])
    print(f"[{model}/{d['species']}] mean p_ref={p_ref.mean():.3f} p_alt={p_alt.mean():.3f} "
          f"| corr(p_alt,alt_freq)={r:.3f}", flush=True)
    return out
