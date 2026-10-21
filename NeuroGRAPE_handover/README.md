# Neuro-GRAPE small-data handover

## Contents
- raw/EEG: one participant's RSVP EEG recording and available metadata.
- raw/MEG: one MEG participant/session/run recording and available metadata.
- raw/fMRI: one participant's THINGS-task BOLD run and event metadata.
- manifests/shared_stimuli.csv: 20 filenames present in the selected EEG,
  MEG and fMRI event/mapping metadata.
- metadata/: source dataset descriptions and the selected MEG stimulus mapping.

## Selected recordings
- EEG: sub-01, task-rsvp.
- MEG: sub-BIGMEG2, session 07, run 06.
- fMRI: sub-01, session things08, task-things, run 10.

## Important limitations
1. This is a small engineering prototype, not the full research dataset.
2. The three modalities come from different participants and experiments.
   Shared filenames align stimulus identity, not simultaneous brain activity.
3. The manifest contains metadata-level matches. It does not yet mean the
   raw signals have been preprocessed, epoched, or temporally aligned.
4. EEG objectnumber is retained as recorded; do not assume it is equivalent
   to MEG/fMRI category or exemplar IDs without separate validation.
5. The fMRI BOLD image is raw. Motion correction, confound handling,
   HRF-aware feature extraction and other preprocessing have not been done.
6. MEG preprocessing and event-to-sample alignment still need validation.
7. Check each dataset's documentation and license before redistribution.

## Suggested next steps
1. Verify file integrity and read the sidecar metadata.
2. Load a small number of EEG/MEG epochs and inspect the fMRI image dimensions.
3. Create modality-specific preprocessing scripts with recorded parameters.
4. Preserve subject/run/stimulus identifiers through all transformations.
5. Have teammates start model prototyping only after agreeing on the
   shared feature format and train/validation split.
