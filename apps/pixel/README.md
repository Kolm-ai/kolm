# kolm pixel

Verifiable image-quality claims against a physical camera floor and a
perceptual display ceiling, plus the data pipeline that feeds a restoration /
edit model. The contract is in [SPEC.md](SPEC.md).

## Install

```bash
cd apps/pixel
pip install -e .            # numpy + pillow only
pip install -e ".[data]"    # + requests, pyarrow for the collector
pip install -e ".[train]"   # + torch for the model
pip install -e ".[dev]"     # + pytest
```

## Demo (10 s, no data)

```bash
python -m pixel demo
```

Evaluates the example claims twice, once with the ground truth as candidate and
once with the simulated best-case camera. Expected outcome:

| candidate | still (delivered as JPEG q92) | video (delivered as PNG) |
|---|---|---|
| truth | SUPPORTED on master and delivered; beats the camera envelope on mtf50, shadow noise, shadow detail, dE p95; the two shadow wins do not survive JPEG q92 | SUPPORTED; same wins plus temporal noise |
| camera | NOT SUPPORTED, halts at `tone_color` | same |

The camera fails because one exposure cannot keep both the highlight wedge and
the shadow region: at 0 EV the wedge clips, at -3 EV shadow noise exceeds one
code value. Against the camera envelope (its best exposure per metric) the
truth ties on highlights and wins on shadows, which is the physical statement.
Reports land in `demo_out/<claim>-<candidate>/report.md`.

## Evaluate your own generator

1. Render the scene your claim names: `Scene(SceneSpec.from_dict(claim["scene"]))`,
   frames `render(t)` for `t` in `range(frames)`, reference images
   `display_encode(frame)`. Your model has to reproduce that world.
2. Save frames as `frame_0000.png` ... in a directory.
3. Copy `claims/example_files.json`, point `candidate.path` at it, set the
   delivery format you will actually ship.
4. `python -m pixel evaluate my_claim.json --out out/` exits 0 when the claim
   is supported and 2 when it is not. Read `out/report.md`.

## Preference study

```bash
python -m pixel preference manifest camera gen truth --rounds 20 --out pres.jsonl --key key.json
# show each presentation blind, collect votes as {id, question, choice: left|right|tie}
python -m pixel preference fit --votes votes.jsonl --key key.json --candidate gen --reference camera
```

## Data

```bash
python -m pixel plan-storage --tb 4
python -m pixel collect --store /data/pixel/store --max-bytes 3000000000000 --workers 16
python -m pixel verify  --store /data/pixel/store
python -m pixel shard   --store /data/pixel/store --out /data/pixel/shards
python -m pixel dedup   /data/pixel/shards --out /data/pixel/keep.txt
python -m pixel label   /data/pixel/shards --out /data/pixel/labels --recipes 32
python -m pixel train   --sources /data/pixel/shards --out /data/pixel/runs/v1 --steps 200000
python -m pixel edit photo.png --ckpt /data/pixel/runs/v1/last.pt --direction "blur=-1,noise=-0.7" --out fixed.png
```

Sources are listed in `sources/registry.json`. The Hugging Face and Zenodo
entries are templates (`enabled: false`): set the repo or record, confirm the
license on the dataset card, flip the flag. Everything else in the registry is
CC0 or per-file license-checked at collect time.

## Tests

```bash
python -m pytest tests -q
```

The torch test is skipped when torch is absent. `python -m pixel train --smoke`
runs the training loop on procedural sources and reports first/last loss.

## Reading a report

* **Verdict** says SUPPORTED or where the gate halted, for the master and again
  for the delivered file.
* **Metrics** table: candidate vs camera at the claim exposure vs camera
  envelope vs truth, with beats / ties / loses per metric.
* **Wins over the camera envelope** lists what survived delivery. A win lost in
  delivery is the usual outcome for JPEG at q92 on shadow and highlight
  metrics; that is the point of the section.

The JS Bradley-Terry fitter in `src/bradley-terry.js` is the same model as
`pixel/preference.py`; this one keeps the image pipeline in one language and
adds the bootstrap.
