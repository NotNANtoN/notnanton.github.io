# Cosmos T16/T2 action-head learning curves

`training-history.json` contains **all 48 + 37 measured validation points** from
archived W&B runs `ixzworl4` and `jri7vehq`. `plot_training.py` renders
`static/assets/img/video-vam/cosmos-t16-t2-training.svg` with a **960 × 1000**
viewBox, SVG text (not font outlines), and an explicit light background.

## Regenerate

Using the existing research interpreter on abakus, from **any working directory**:

```sh
/home/anton/lerobot-video-vam/.venv/bin/python -B /home/anton/notnanton.github.io/scripts/video-vam-figures/plot_training.py
```

Read-only validation, including byte-for-byte regeneration of the saved SVG:

```sh
/home/anton/lerobot-video-vam/.venv/bin/python -B /home/anton/notnanton.github.io/scripts/video-vam-figures/plot_training.py --check
```

Only Python's standard library and existing Matplotlib are needed. The reference
renderer is **Matplotlib 3.11.1** with bundled DejaVu Sans. No website dependency,
W&B authentication, private archive download, or training is needed. A fixed SVG
hash salt and omitted date make regeneration deterministic in that environment;
other Matplotlib/font versions may change SVG bytes.

## Meaning and timing

Two vertically stacked linear panels show the same unsmoothed validation RMSE
against optimizer steps and elapsed **head-loop minutes**. All measured points,
including early training and actual stopping endpoints, are retained. Blue solid
circles denote T16 pool2; orange dashed squares denote undistilled T2 unpooled.
Both use the same Video-LoRA adaptation: **undistilled does not mean unadapted**.

Point schema: `{step, full30_mixed_rmse, wall_clock_seconds}`. RMSE is copied from
`val_aggregate_rmse_deg`: global masked full-30 error over five angle coordinates
and one gripper coordinate in [0, 100]. It is **mixed units, not pure degrees**.

Both clocks are copied directly from the archive's `wall_clock_seconds`, divided
by 60 only for plotting. The timer begins after pre-loop cache/model/normalizer
setup and W&B initialization. Each timestamp is taken after the optimizer update,
**before the current validation**, and includes earlier evaluations, checkpointing,
cached-feature I/O and head-loop overhead. It excludes **Video-LoRA training and
feature-cache creation**. No `_runtime`, datetime, constant offset, estimated
throughput, or interpolated timestamps are used.

| Run            |          Best RMSE | Optimizer step |     Trainer seconds |    Minutes |
| -------------- | -----------------: | -------------: | ------------------: | ---------: |
| T16 `ixzworl4` | 13.058368685598639 |         38,000 | 6,879.7633914150065 | 114.662723 |
| T2 `jri7vehq`  | 13.740523813264303 |         27,000 | 2,779.7863010689616 |  46.329772 |

## Provenance and limitations

The JSON records research-repository-relative archive/config/metadata paths and
SHA-256 hashes, actual history-field definitions, dataset revision and verified
fixed-split checksum, batch 8 / LR 1e-4 / seed 0, and recorded Git revisions. All
85 step/RMSE pairs were cross-checked against the earlier research exports; both
trainer-time columns were exported directly from the local binary histories.
The old T16 export used W&B `_runtime`, unlike T2; its old hours are not reused.

This is a **single-seed historical comparison**, not a controlled temporal-length
ablation: T16 uses spatial pool2 (4,800 context tokens), while T2 is unpooled
(2,400). The recorded Git revisions are not exact dirty working-tree snapshots.
Best points are validation-selected, not equal-quality timing thresholds. The
original feature tensors/LoRA artifact were unavailable for revalidation, and no
attributable cache-build duration is claimed. Live W&B access was unavailable;
run URLs are documented but not live-verified. No distillation scores are plotted.

The renderer validates finite values, monotonic steps/times, all 85 points,
exact minima and numerical checksums, dynamic axis bounds, figure text bounds,
nonoverlapping best annotations, SVG XML/text and every measured marker in both
panels. It does not contact W&B or reread the private research archives. Browser
integration and website builds are intentionally outside this script's scope.
