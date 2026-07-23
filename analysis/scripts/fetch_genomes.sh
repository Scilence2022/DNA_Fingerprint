#!/usr/bin/env bash
# Download a curated RefSeq genome panel spanning a known divergence gradient:
# intra-species (E. coli strains) -> genus -> family -> order -> phylum.
# Resolves the versioned assembly directory from the FTP listing so we do not
# have to hard-code assembly names.
set -uo pipefail

OUT="${1:?usage: fetch_genomes.sh <outdir>}"
mkdir -p "$OUT"

# accession<TAB>short label
PANEL=$(cat <<'EOF'
GCF_000005845.2	Ecoli_K12_MG1655
GCF_000008865.2	Ecoli_O157H7_Sakai
GCF_000007445.1	Ecoli_CFT073
GCF_000010485.1	Ecoli_O111
GCF_000026345.1	Ecoli_UMN026
GCF_000013305.1	Ecoli_UTI89
GCF_000006925.2	Shigella_flexneri_2a
GCF_000012005.1	Shigella_sonnei
GCF_000006945.2	Salmonella_Typhimurium_LT2
GCF_000009505.1	Salmonella_Paratyphi
GCF_000027085.1	Citrobacter_rodentium
GCF_000025565.1	Enterobacter_cloacae
GCF_000240185.1	Klebsiella_pneumoniae
GCF_000009065.1	Yersinia_pestis_CO92
GCF_000006765.1	Pseudomonas_aeruginosa_PAO1
GCF_000009045.1	Bacillus_subtilis_168
EOF
)

fetch_one() {
    local acc="$1" label="$2"
    local dest="$OUT/${label}.fna"
    [ -s "$dest" ] && { echo "  cached  $label"; return 0; }

    # GCF_000005845.2 -> GCF/000/005/845
    local base="${acc%%.*}"           # GCF_000005845
    local pfx="${base%%_*}"           # GCF
    local digits="${base#*_}"         # 000005845
    local d1="${digits:0:3}" d2="${digits:3:3}" d3="${digits:6:3}"
    local dir="https://ftp.ncbi.nlm.nih.gov/genomes/all/${pfx}/${d1}/${d2}/${d3}"

    # The versioned assembly dir name embeds the assembly name; resolve it.
    local asmdir
    asmdir=$(curl -s --max-time 60 "${dir}/" \
             | grep -o "${acc}_[^\"/]*" | head -1)
    if [ -z "$asmdir" ]; then
        echo "  FAIL    $label ($acc): could not resolve assembly dir" >&2
        return 1
    fi

    local url="${dir}/${asmdir}/${asmdir}_genomic.fna.gz"
    if curl -s --max-time 300 "$url" | gunzip -c > "$dest" 2>/dev/null && [ -s "$dest" ]; then
        local bp
        bp=$(grep -v '^>' "$dest" | tr -d '\n' | wc -c | tr -d ' ')
        echo "  ok      $label  (${bp} bp)"
    else
        echo "  FAIL    $label ($acc): download failed" >&2
        rm -f "$dest"
        return 1
    fi
}

echo "Fetching RefSeq panel into $OUT"
while IFS=$'\t' read -r acc label; do
    [ -z "$acc" ] && continue
    fetch_one "$acc" "$label" &
done <<< "$PANEL"
wait

echo
echo "Downloaded: $(ls -1 "$OUT"/*.fna 2>/dev/null | wc -l | tr -d ' ') genomes"
ls -la "$OUT"/*.fna 2>/dev/null | awk '{print $5, $9}'
