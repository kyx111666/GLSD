# Smoke11 Google Colab extraction

This Colab-only workflow reads the fixed 11-candidate extraction manifest and the `SAMM_longvideos.zip` split archive stored on Google Drive. It indexes only the four required frame directories inside the archive, selectively extracts only the manifest-requested JPG/PNG members, audits them, and creates a deterministic ZIP. It does not unpack the complete dataset, run an MLLM, or modify GLSD, ground truth, candidates, or the smoke40 protocol.

## Run in Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `extract_smoke11_from_drive.ipynb`.
2. Select a **CPU** runtime. No GPU is needed. The notebook installs the small `p7zip-full` system package only if Colab does not already provide `7z`.
3. In the first code cell, change only:

   ```python
   SAMMLV_ARCHIVE_PATH = "/content/drive/MyDrive/.../SAMM_longvideos.zip"
   OUTPUT_DRIVE_DIR = "/content/drive/MyDrive/..."
   ```

   `SAMMLV_ARCHIVE_PATH` must point to the final `.zip` file, not `.z01`. All parts such as `.z01`, `.z02`, `.z03`, `.z04`, and `.zip` must remain together in the same Drive directory with unchanged names. `OUTPUT_DRIVE_DIR` is where the five final artifacts are copied.
4. Leave `USE_MANIFEST_UPLOAD = True` for the preferred upload workflow. Choose **Runtime → Run all**, authorize the Drive mount, and upload `cloud_extraction_manifest.json` when prompted.
5. Alternatively, set `USE_MANIFEST_UPLOAD = False` and set `MANIFEST_PATH` to the manifest's Drive path. This is optional and is not one of the two paths normally changed.
6. The notebook streams the archive member listing and selects only paths under `006_1`, `011_2`, `016_7`, and `020_7`. Review `FRAME_NAMING_AUDIT`. By default, it accepts auto-detection only when every complete archive directory clearly begins at physical index 0 or 1 and all four agree. If it stops with `FRAME_INDEX_CONVENTION_UNRESOLVED`, inspect the audit, set `FRAME_FILE_INDEX_OFFSET` explicitly to `0` or `1`, and rerun. Do not guess or change manifest indices.
7. Confirm the final output is `REMOTE_FRAME_EXTRACTION_PASS` and all seven stage messages were printed.
8. Find `smoke11_remote_frames.zip` and the four selected audit artifacts in `OUTPUT_DRIVE_DIR`.

The runtime also contains `/content/smoke11_remote_frames/`, `/content/smoke11_archive_selection/` (only the selected source members), and `/content/smoke11_remote_frames.zip.sha256`. The complete SAMMLV archive is never copied into `/content`, and the entire runtime directory is not synchronized to Drive.

## Fixed scope and PASS gate

- Expected candidates: **11** across `006_1`, `011_2`, `016_7`, and `020_7`.
- Manifest protocol maximum: **99 frame occurrences** (`11 × 9`), but the manifest is authoritative and no frame is added to reach 99.
- PASS requires 11/11 complete candidates, zero missing frame occurrences, zero image decode failures, zero unresolved index conventions, and zero duplicate-source hash inconsistencies.
- A failed audit is never packaged or copied to Drive. Its audit files remain under `/content/smoke11_remote_frames/audit/` for diagnosis.

## Script entry point

`extract_smoke11_from_drive.py` contains the same Colab core logic as the notebook. It may be uploaded to Colab and run with:

```bash
python extract_smoke11_from_drive.py
```

It is intentionally not a local Drive-access tool.
