#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prepare_safe_names_fast.sh
#
#  • Repairs any "...fixednamefixedname..."   → "...fixedname"
#  • Adds ONE trailing "fixedname" to names that still end with "_<digits>"
#  • Keeps original order in meta_data/test_list_initial.txt
#  • Renames files/dirs in test_data/ accordingly (single depth-first walk)
# ---------------------------------------------------------------------------

set -euo pipefail
FIXTAG="fixedname"          # change the tag if you like

ROOT="/workspace/data/ForAINetV2"
LIST="${ROOT}/meta_data/test_list_initial.txt"
LIST_BAK="${ROOT}/meta_data/test_list_initial_original.txt"
DATA="${ROOT}/test_data"

cp "$LIST" "$LIST_BAK"
echo "✔  Backup created → $(basename "$LIST_BAK")"

# ---------- helper: convert a single basename to its final “safe” form -------
convert_name() {
    local name="$1"
    # 1. collapse repeated fixedname tokens
    local clean; clean=$(echo "$name" | sed -E "s/(${FIXTAG})+/${FIXTAG}/g")
    # 2. ensure it does not end in raw _digits
    if [[ "$clean" =~ _[0-9]+$ ]] && [[ "$clean" != *"${FIXTAG}" ]]; then
        echo "${clean}${FIXTAG}"
    else
        echo "$clean"
    fi
}

# ---------- (A) create a new list, preserving order --------------------------
tmp_list="${LIST}.tmp"
> "$tmp_list"                # start fresh

declare -A rename_map        # original basename → final safe basename

while IFS= read -r scan || [ -n "${scan:-}" ]; do
    safe=$(convert_name "$scan")
    echo "$safe" >> "$tmp_list"
    rename_map["$scan"]="$safe"
done < "$LIST"

mv "$tmp_list" "$LIST"
echo "✔  $(basename "$LIST") updated ( $(wc -l < "$LIST") lines )"

# ---------- (B) single depth-first walk over test_data -----------------------
find "$DATA" -depth | while read -r path; do
    name="$(basename "$path")"
    dir="$(dirname "$path")"

    target="$(convert_name "$name")"

    # If the list–based map knows a different final name, prefer that.
    # (handles offsets, *_bluepoints_*, etc. matching the scan root)
    if [[ -n "${rename_map[$name]+x}" ]]; then
        target="${rename_map[$name]}"
    fi

    if [[ "$name" != "$target" ]]; then
        if [[ -e "$dir/$target" ]]; then
            echo "⚠  Skip (target exists): $dir/$target"
        else
            mv "$path" "$dir/$target"
            echo "✔  mv $name → $target"
        fi
    fi
done

echo "All names are now safe and consistent."
