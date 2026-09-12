# Viewer-ready ATT bundles

## Required structure

A complete bundle contains non-empty instances of:

```text
<prefix>_results.db
<prefix>_<agent>_shader_engine_<se>_<dispatch>.att
<prefix>_gfx<arch>_code_object_id_<id>.out
stats_ui_output_agent_<agent>_dispatch_<dispatch>.csv
ui_output_agent_<agent>_dispatch_<dispatch>/
  code.json
  filenames.json
  occupancy.json
  wstates*.json
  se*_sm*_sl*_wv*.json
```

`source_*.py` and `snapshots.json` are optional. They may be absent for kernels
loaded from precompiled libraries such as rocBLAS. Their absence alone does not
make the bundle invalid.

Validation must parse every decoded JSON file, run SQLite `integrity_check` on
the results database, and confirm that the stats CSV contains data rows.

## Transfer

Archive the entire trace root so code objects, the database, statistics, and UI
JSON remain adjacent. Preserve the root directory name in the archive and
report a SHA-256 digest. Transferring only `.att` or only `ui_output_agent_*`
does not preserve a complete profile.

## Cycle windows

Wave JSON files encode instruction events under `wave.instructions`; the first
field is the cycle and the last field is the decoded code ID. `code.json` maps
that ID to instruction text and virtual address.

For loop iterations `first..last` inclusive:

1. Select one wave explicitly, normally `se0_sm0_sl0_wv0.json`.
2. Identify an instruction that executes once at the loop boundary.
3. Start at marker occurrence `first`.
4. End at marker occurrence `last + 1`.
5. State whether numbering is zero- or one-based.

Do not compare logical K iterations until accounting for tile depth and loop
unrolling. For example, an outer rocBLAS loop can represent two `BK=64` tiles,
while a TLX loop represents one `BK=32` tile.
