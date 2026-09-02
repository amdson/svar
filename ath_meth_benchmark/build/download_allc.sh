#!/bin/bash
# Resumable bulk download of GSE43857 mC_calls files listed in the manifest.
# Skips files already present and gzip-valid; logs failures to allc_failed.log.
set -u
BASE=/90daydata/small_grains/andrew.dickson/datasets/arabidopsis/methylation
MANIFEST=$BASE/meta/allc_download_manifest.txt
DEST=$BASE/allc
FAIL=$BASE/meta/allc_failed.log
: > "$FAIL"

fetch() {
    url="$1"
    f="$DEST/$(basename "$url")"
    if [ -s "$f" ] && gzip -t "$f" 2>/dev/null; then
        return 0
    fi
    for attempt in 1 2 3; do
        wget -c -q -T 120 -O "$f" "$url" && gzip -t "$f" 2>/dev/null && return 0
        sleep $((attempt * 10))
    done
    echo "$url" >> "$FAIL"
    rm -f "$f"
    return 1
}
export -f fetch
export DEST FAIL

xargs -a "$MANIFEST" -P 6 -n 1 -I{} bash -c 'fetch "$@"' _ {}
n_ok=$(ls "$DEST" | wc -l)
n_fail=$(wc -l < "$FAIL")
echo "DONE: $n_ok files present, $n_fail failed (see $FAIL)"
