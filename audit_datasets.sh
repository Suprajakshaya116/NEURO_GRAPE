#!/usr/bin/env bash
# =============================================================================
# NEURO-GRAPE  |  Read-Only Dataset Audit Script
# Run in the existing Linux terminal as user RL.
# Output is saved to /tmp/neuro_grape_audit.txt and also printed to screen.
# NO sudo, NO dataset writes, NO downloads, NO expensive full-dataset scans.
# =============================================================================

set -euo pipefail
OUTFILE="/tmp/neuro_grape_audit.txt"
: > "$OUTFILE"          # clear any previous run
exec > >(tee -a "$OUTFILE") 2>&1

EEG_ROOT="/home/RL/Downloads/BrainFLORA/Dataset1/ds003825-1.1.0"
MEG_ROOT="/home/RL/Downloads/BrainFLORA/Dataset2/ds004212-2.0.0"
FMRI_ROOT="/home/RL/Downloads/BrainFLORA/Dataset3/ds004192-1.0.7"
BRAINFLORA="/home/RL/Downloads/BrainFLORA"

SEP="=================================================================="
SUBSEP="------------------------------------------------------------------"
section()  { echo; echo "$SEP"; echo "  $1"; echo "$SEP"; }
subsect()  { echo; echo "  --- $1 ---"; }

# ──────────────────────────────────────────────────────────────────────────────
section "0. SESSION VERIFICATION"
echo "Date/time  : $(date)"
echo "User       : $(whoami)"
echo "Hostname   : $(hostname)"
echo "PWD        : $(pwd)"
echo ""
echo "Disk space (filesystem containing BrainFLORA):"
df -h "$BRAINFLORA" 2>/dev/null || df -h "$HOME"
echo ""
echo "Disk usage per dataset (surface level only):"
du -sh "$EEG_ROOT"  2>/dev/null || echo "EEG root not found"
du -sh "$MEG_ROOT"  2>/dev/null || echo "MEG root not found"
du -sh "$FMRI_ROOT" 2>/dev/null || echo "fMRI root not found"

# ──────────────────────────────────────────────────────────────────────────────
section "1. DATASET DIRECTORY EXISTENCE"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    if [ -d "$PATH_" ]; then
        echo "[$NAME]  FOUND  ->  $PATH_"
    else
        echo "[$NAME]  *** NOT FOUND ***  ->  $PATH_"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
section "2. DATASET DESCRIPTION FILES"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    subsect "$NAME  dataset_description.json"
    FILE="$PATH_/dataset_description.json"
    if [ -f "$FILE" ]; then cat "$FILE"; else echo "(not found)"; fi
done

# ──────────────────────────────────────────────────────────────────────────────
section "3. TOP-LEVEL DIRECTORY LISTING"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    subsect "$NAME top-level"
    ls -1 "$PATH_" 2>/dev/null || echo "(ls failed)"
done

# ──────────────────────────────────────────────────────────────────────────────
section "4. PARTICIPANT LISTS"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    subsect "$NAME  participants.tsv"
    FILE="$PATH_/participants.tsv"
    if [ -f "$FILE" ]; then
        echo "(lines including header: $(wc -l < "$FILE"))"
        cat "$FILE"
    else
        echo "(not found)"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
section "5. SUBJECT DIRECTORIES"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    subsect "$NAME subject folders"
    SUBJECTS=( $(ls -1 "$PATH_" 2>/dev/null | grep '^sub-' || true) )
    echo "Count : ${#SUBJECTS[@]}"
    echo "List  : ${SUBJECTS[*]:-none}"
done

# ──────────────────────────────────────────────────────────────────────────────
section "6. EVENT FILE DISCOVERY"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    subsect "$NAME  events.tsv – first 6 found"
    find "$PATH_" -maxdepth 5 -name "*events.tsv" 2>/dev/null | head -6
    subsect "$NAME  events.tsv – total count"
    find "$PATH_" -maxdepth 5 -name "*events.tsv" 2>/dev/null | wc -l
done

# ──────────────────────────────────────────────────────────────────────────────
section "7. EVENT FILE COLUMN HEADERS  (sub-01, one file per modality)"
# EEG
EEG_EV=$(find "$EEG_ROOT/sub-01" -name "*events.tsv" 2>/dev/null | head -1 || true)
subsect "EEG events.tsv columns"
if [ -n "${EEG_EV:-}" ]; then
    echo "File: $EEG_EV"
    head -1 "$EEG_EV"
else
    echo "(no events.tsv found under $EEG_ROOT/sub-01)"
fi

# MEG
MEG_EV=$(find "$MEG_ROOT/sub-01" -name "*events.tsv" 2>/dev/null | head -1 || true)
subsect "MEG events.tsv columns"
if [ -n "${MEG_EV:-}" ]; then
    echo "File: $MEG_EV"
    head -1 "$MEG_EV"
else
    echo "(no events.tsv found under $MEG_ROOT/sub-01)"
fi

# fMRI
FMRI_EV=$(find "$FMRI_ROOT/sub-01" -name "*events.tsv" 2>/dev/null | head -1 || true)
subsect "fMRI events.tsv columns"
if [ -n "${FMRI_EV:-}" ]; then
    echo "File: $FMRI_EV"
    head -1 "$FMRI_EV"
else
    echo "(no events.tsv found under $FMRI_ROOT/sub-01)"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "8. EVENT FILE SAMPLE ROWS  (first 6 data rows, sub-01)"
subsect "EEG"
[ -n "${EEG_EV:-}" ] && head -7 "$EEG_EV" || echo "(skipped – no file found)"

subsect "MEG"
[ -n "${MEG_EV:-}" ] && head -7 "$MEG_EV" || echo "(skipped – no file found)"

subsect "fMRI"
[ -n "${FMRI_EV:-}" ] && head -7 "$FMRI_EV" || echo "(skipped – no file found)"

# ──────────────────────────────────────────────────────────────────────────────
section "9. EEG – stimname FIELD ANALYSIS"
ALL_EEG_EV=$(find "$EEG_ROOT" -name "*events.tsv" 2>/dev/null || true)

if [ -n "$ALL_EEG_EV" ]; then
    FIRST_EEG_EV=$(echo "$ALL_EEG_EV" | head -1)
    STIMCOL=$(head -1 "$FIRST_EEG_EV" | tr '\t' '\n' | grep -nx 'stimname' | cut -d: -f1 || true)

    subsect "stimname column index in events.tsv"
    echo "${STIMCOL:-(not found – check column headers above)}"

    if [ -n "${STIMCOL:-}" ]; then
        subsect "Unique stimnames across ALL EEG event files"
        UNIQ_EEG=$(echo "$ALL_EEG_EV" | \
            xargs awk -F'\t' -v c="$STIMCOL" 'NR>1{print $c}' | \
            sort -u | wc -l)
        echo "Unique stimnames: $UNIQ_EEG"

        subsect "Sample stimnames (first 20 globally unique)"
        echo "$ALL_EEG_EV" | \
            xargs awk -F'\t' -v c="$STIMCOL" 'NR>1{print $c}' | \
            sort -u | head -20

        subsect "EEG trial count across all event files (total data rows)"
        echo "$ALL_EEG_EV" | \
            xargs awk 'NR>1{count++} END{print count}' | \
            awk '{sum+=$1} END{print "Total EEG trial rows:", sum}'
    fi
else
    echo "(no EEG events.tsv files found)"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "10. MEG – NUMERIC-ID MAPPING INVESTIGATION"
subsect "All files at MEG dataset root (non-subject)"
find "$MEG_ROOT" -maxdepth 1 -type f 2>/dev/null

subsect "Search: sample_attributes* anywhere in MEG dataset"
find "$MEG_ROOT" -name "sample_attributes*" 2>/dev/null || echo "(none found)"

subsect "Search: all *.csv files in MEG dataset (excluding per-sub recording dirs)"
find "$MEG_ROOT" -maxdepth 4 -name "*.csv" 2>/dev/null | head -20

subsect "Search: all *.tsv files at MEG root (first 20)"
find "$MEG_ROOT" -maxdepth 2 -name "*.tsv" 2>/dev/null | head -20

subsect "MEG sub-01/meg/ folder contents"
ls -1 "$MEG_ROOT/sub-01/meg/" 2>/dev/null | head -30

subsect "MEG events.tsv – all column indices (sub-01)"
if [ -n "${MEG_EV:-}" ]; then
    echo "Numbered columns:"
    head -1 "$MEG_EV" | tr '\t' '\n' | cat -n
fi

subsect "MEG events.tsv – first 10 data rows (sub-01)"
if [ -n "${MEG_EV:-}" ]; then
    sed -n '2,11p' "$MEG_EV"
fi

subsect "MEG – unique 'value' entries (sub-01)"
if [ -n "${MEG_EV:-}" ]; then
    VCOL=$(head -1 "$MEG_EV" | tr '\t' '\n' | grep -nx '^value$' | cut -d: -f1 || true)
    if [ -n "${VCOL:-}" ]; then
        U=$(awk -F'\t' -v c="$VCOL" 'NR>1{print $c}' "$MEG_EV" | sort -u | wc -l)
        echo "Unique 'value' entries: $U"
        echo "Sample (first 20):"
        awk -F'\t' -v c="$VCOL" 'NR>1{print $c}' "$MEG_EV" | sort -u | head -20
    else
        echo "('value' column not found)"
    fi
fi

subsect "MEG – unique 'things_category_nr' (sub-01)"
if [ -n "${MEG_EV:-}" ]; then
    TCOL=$(head -1 "$MEG_EV" | tr '\t' '\n' | grep -nx 'things_category_nr' | cut -d: -f1 || true)
    if [ -n "${TCOL:-}" ]; then
        U=$(awk -F'\t' -v c="$TCOL" 'NR>1{print $c}' "$MEG_EV" | sort -nu | wc -l)
        echo "Unique things_category_nr values: $U"
        echo "Sample (first 20):"
        awk -F'\t' -v c="$TCOL" 'NR>1{print $c}' "$MEG_EV" | sort -nu | head -20
    else
        echo "('things_category_nr' column not found)"
    fi
fi

subsect "sample_attributes_P1.csv – if found, print first 20 lines"
SA=$(find "$MEG_ROOT" -name "sample_attributes_P*.csv" 2>/dev/null | head -1 || true)
if [ -n "$SA" ]; then
    echo "File: $SA"
    head -20 "$SA"
else
    echo "(sample_attributes_P*.csv not found – MEG image mapping via this file is UNRESOLVED)"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "11. fMRI – IMAGE IDENTIFIER ANALYSIS"
ALL_FMRI_EV=$(find "$FMRI_ROOT" -name "*events.tsv" 2>/dev/null || true)

if [ -n "$ALL_FMRI_EV" ]; then
    FIRST_FMRI_EV=$(echo "$ALL_FMRI_EV" | head -1)
    FPCOL=$(head -1 "$FIRST_FMRI_EV" | tr '\t' '\n' | grep -nx 'file_path' | cut -d: -f1 || true)

    subsect "file_path column index"
    echo "${FPCOL:-(not found)}"

    subsect "fMRI events.tsv – numbered columns (sub-01)"
    if [ -n "${FMRI_EV:-}" ]; then
        head -1 "$FMRI_EV" | tr '\t' '\n' | cat -n
    fi

    subsect "fMRI events.tsv – first 10 data rows (sub-01)"
    if [ -n "${FMRI_EV:-}" ]; then
        sed -n '2,11p' "$FMRI_EV"
    fi

    if [ -n "${FPCOL:-}" ]; then
        subsect "Unique file_paths in sub-01"
        U1=$(awk -F'\t' -v c="$FPCOL" 'NR>1{print $c}' "${FMRI_EV:-/dev/null}" | sort -u | wc -l)
        echo "Count: $U1"
        echo "Sample (first 15):"
        awk -F'\t' -v c="$FPCOL" 'NR>1{print $c}' "${FMRI_EV:-/dev/null}" | sort -u | head -15

        subsect "Unique file_path basenames across ALL fMRI subjects"
        UNIQ_FMRI=$(echo "$ALL_FMRI_EV" | \
            xargs awk -F'\t' -v c="$FPCOL" 'NR>1{print $c}' | \
            awk -F'/' '{print $NF}' | sort -u | wc -l)
        echo "Unique basenames across all fMRI subjects: $UNIQ_FMRI"
    fi

    subsect "fMRI – unique things_exemplar_nr (sub-01)"
    if [ -n "${FMRI_EV:-}" ]; then
        ECOL=$(head -1 "$FMRI_EV" | tr '\t' '\n' | grep -nx 'things_exemplar_nr' | cut -d: -f1 || true)
        if [ -n "${ECOL:-}" ]; then
            U=$(awk -F'\t' -v c="$ECOL" 'NR>1{print $c}' "$FMRI_EV" | sort -nu | wc -l)
            echo "Unique things_exemplar_nr: $U"
        else
            ECOL=$(head -1 "$FMRI_EV" | tr '\t' '\n' | grep -nx 'exemplar_nr' | cut -d: -f1 || true)
            [ -n "${ECOL:-}" ] && {
                U=$(awk -F'\t' -v c="$ECOL" 'NR>1{print $c}' "$FMRI_EV" | sort -nu | wc -l)
                echo "Unique exemplar_nr: $U"
            } || echo "(neither things_exemplar_nr nor exemplar_nr found)"
        fi
    fi
else
    echo "(no fMRI events.tsv files found)"
fi

# ──────────────────────────────────────────────────────────────────────────────
section "12. CROSS-MODALITY IMAGE IDENTIFIER OVERLAP"
EEG_TMP=$(mktemp /tmp/ng_eeg_XXXX.txt)
FMRI_TMP=$(mktemp /tmp/ng_fmri_XXXX.txt)

# Build EEG set
if [ -n "${ALL_EEG_EV:-}" ] && [ -n "${STIMCOL:-}" ]; then
    echo "$ALL_EEG_EV" | \
        xargs awk -F'\t' -v c="$STIMCOL" 'NR>1{print $c}' | \
        sort -u > "$EEG_TMP"
    echo "EEG unique stimnames: $(wc -l < "$EEG_TMP")"
else
    echo "EEG stimname set not built (check section 9)"
fi

# Build fMRI set (basenames)
if [ -n "${ALL_FMRI_EV:-}" ] && [ -n "${FPCOL:-}" ]; then
    echo "$ALL_FMRI_EV" | \
        xargs awk -F'\t' -v c="$FPCOL" 'NR>1{print $c}' | \
        awk -F'/' '{print $NF}' | sort -u > "$FMRI_TMP"
    echo "fMRI unique file_path basenames: $(wc -l < "$FMRI_TMP")"
else
    echo "fMRI file_path set not built (check section 11)"
fi

subsect "EEG ∩ fMRI overlap"
if [ -s "$EEG_TMP" ] && [ -s "$FMRI_TMP" ]; then
    OVERLAP=$(comm -12 "$EEG_TMP" "$FMRI_TMP" | wc -l)
    echo "Overlap count (exact filename match): $OVERLAP"
    echo "Sample overlapping names (first 20):"
    comm -12 "$EEG_TMP" "$FMRI_TMP" | head -20
    echo ""
    subsect "Names in EEG but NOT in fMRI (first 15)"
    comm -23 "$EEG_TMP" "$FMRI_TMP" | head -15
    echo "..."
    subsect "Names in fMRI but NOT in EEG (first 15)"
    comm -13 "$EEG_TMP" "$FMRI_TMP" | head -15
else
    echo "(one or both sets empty – cannot compute overlap)"
fi

rm -f "$EEG_TMP" "$FMRI_TMP"

# ──────────────────────────────────────────────────────────────────────────────
section "13. fMRI – DATA REPRESENTATIONS"
subsect "derivatives/ directories"
find "$FMRI_ROOT" -maxdepth 2 -type d -name "derivatives" 2>/dev/null || echo "(none)"

subsect ".npz files – total count and first 10 paths"
NPZCOUNT=$(find "$FMRI_ROOT" -name "*.npz" 2>/dev/null | wc -l)
echo "Total .npz files: $NPZCOUNT"
find "$FMRI_ROOT" -name "*.npz" 2>/dev/null | head -10

subsect ".nii.gz file count per subject"
for SUBJ in sub-01 sub-02 sub-03; do
    C=$(find "$FMRI_ROOT/$SUBJ" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "$SUBJ: $C .nii.gz files"
done

subsect "First 10 .nii.gz filenames in sub-01"
find "$FMRI_ROOT/sub-01" -name "*.nii.gz" 2>/dev/null | head -10

# ──────────────────────────────────────────────────────────────────────────────
section "14. EEG – RECORDING FILE TYPES (sub-01)"
subsect "Extension breakdown in sub-01"
find "$EEG_ROOT/sub-01" -type f 2>/dev/null | \
    awk -F. '{print $NF}' | sort | uniq -c | sort -rn

subsect "sub-01/eeg/ folder contents"
ls -1 "$EEG_ROOT/sub-01/eeg/" 2>/dev/null | head -20 || echo "(not found)"

# ──────────────────────────────────────────────────────────────────────────────
section "15. MEG – RECORDING FILE TYPES (sub-01)"
subsect "sub-01/meg/ folder contents"
ls -1 "$MEG_ROOT/sub-01/meg/" 2>/dev/null | head -30 || echo "(not found)"

subsect ".ds directories in sub-01"
find "$MEG_ROOT/sub-01" -name "*.ds" -type d 2>/dev/null | head -10

# ──────────────────────────────────────────────────────────────────────────────
section "16. THINGS REFERENCE FILES AT BrainFLORA LEVEL"
subsect "All CSV/TSV/TXT files within 4 levels of BrainFLORA (excluding sub-* and large recording dirs)"
find "$BRAINFLORA" -maxdepth 4 \
    \( -name "*.csv" -o -name "*.tsv" -o -name "*.txt" \) \
    2>/dev/null | grep -v '/sub-' | grep -v '\.ds/' | head -30

subsect "THINGS concept/category/exemplar reference files (4 levels)"
find "$BRAINFLORA" -maxdepth 4 \
    \( -iname "*things*" -o -iname "*concept*" -o -iname "*category*" -o -iname "*exemplar*" \) \
    -type f 2>/dev/null | head -20

# ──────────────────────────────────────────────────────────────────────────────
section "17. AUDIT COMPLETE"
echo "Full output written to: $OUTFILE"
echo "Audit finished at     : $(date)"
echo ""
echo "QUICK SUMMARY"
echo "$SUBSEP"
echo "EEG root exists   : $([ -d "$EEG_ROOT"  ] && echo YES || echo NO)"
echo "MEG root exists   : $([ -d "$MEG_ROOT"  ] && echo YES || echo NO)"
echo "fMRI root exists  : $([ -d "$FMRI_ROOT" ] && echo YES || echo NO)"
for LABEL in "EEG:$EEG_ROOT" "MEG:$MEG_ROOT" "fMRI:$FMRI_ROOT"; do
    NAME="${LABEL%%:*}"
    PATH_="${LABEL#*:}"
    N=$(ls -1 "$PATH_" 2>/dev/null | grep -c '^sub-' || echo "0")
    echo "$NAME subjects        : $N"
done
