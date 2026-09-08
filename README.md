<!-- Title -->
# LLM Fine-Tuning: Financial Statement Extraction

Does LoRA fine-tuning a small, local, 3B parameter model beat the free
alternative, a handful of examples put straight in the prompt, at
extracting numbers from financial statements? And does either one close
any of the gap to a hand written rules extractor (0.7765 test accuracy)
or a frontier API language model (0.9321 test accuracy) on the same
task?

This project is a follow-up to [filing-extraction-benchmark](../filing-extraction-benchmark),
reusing that project's labeled data and scoring rule but changing the
method and, necessarily, the input each method sees. That difference is
the first thing to understand before reading any number below.

## What this does and does not have in common with the source project

filing-extraction-benchmark compares three methods for extracting 9
financial statement fields (revenue, operating income, net income,
diluted EPS, total assets, total liabilities, stockholders' equity, cash
and equivalents, operating cash flow) from 24 real companies' 10-K
filings, scored against 1,225 hand labeled ground truth cells, using an
exact scorer: a predicted value counts as correct only if it is within
0.5% relative tolerance of the filed figure (absolute tolerance for
per-share values). That scorer, `filingbench.scoring.values_agree`, is
imported unmodified from the source project and used here too, so a
percentage in this README means the same thing it means there.

What is **not** the same: the source project's 0.9321 API figure is
measured feeding the model the full stripped filing text, up to about
560,000 characters. That is the right comparison for a frontier API
model with a huge context window and a per-call price, but it is not
something a 3B parameter model fine-tuned with LoRA on a laptop can be
usefully trained or evaluated against, the sequences would not fit, and
most of that text is irrelevant to any one of the 9 fields. This project
instead extracts only the tables the source project's own parser already
classified as a financial statement (income statement, balance sheet,
cash flow statement), typically a few thousand characters, not
hundreds of thousands. 6 of the source project's test filings (54 rows,
9 fields each) had no table classified as any statement by that parser
and are excluded from this project's test set entirely, the same way for
every method compared here.

So: same fields, same companies, same ground truth, same scorer, same
0.5% tolerance rule, different, much smaller input. Every number below
should be read against the other numbers below, not against the source
project's full-document 0.9321 figure, which answers a different
question.

## Method

Three methods, all scored on the identical 653-example test set:

1. **Zero-shot**, the base model, `mlx-community/Llama-3.2-3B-Instruct-4bit`,
   given only the instruction and the table.
2. **Few-shot**, the same base model, no training, with 3 worked
   examples from the train split placed in the prompt before the real
   question. Added specifically because it's the obvious cheaper
   alternative to check before crediting fine-tuning with anything.
3. **Fine-tuned**, LoRA fine-tuning of the base model on the train
   split, then evaluated zero-shot (no demonstrations, the point is
   whether training substitutes for showing examples at inference time).

Pipeline:

1. **[src/data_prep.py](src/data_prep.py)**, pulls the source project's
   ground truth and dev/test split, extracts each filing's
   statement-classified tables, tags each table with its own resolved
   scale (`[scale: Dollars in millions]` etc., since different tables in
   the same filing can use different scales), and builds one
   question/answer example per (filing, field) pair. 518 dev examples
   across 60 filings, 653 test examples across 78 filings.
2. **[src/format_for_training.py](src/format_for_training.py)**, splits
   dev into 439 train / 79 valid examples by filing, not by row, so no
   filing's statement text appears in both splits, and writes mlx_lm's
   JSONL prompt/completion format.
3. **[src/few_shot.py](src/few_shot.py)**, selects 3 fixed train-split
   demonstrations (spanning distinct fields) and builds the multi-turn
   prompt for the few-shot baseline. The same 3 demonstrations are reused
   for every test example, so a difference between examples isn't just a
   difference in which demonstrations they happened to get.
4. **[src/run_lora_training.py](src/run_lora_training.py)**, LoRA fine-
   tunes the base model on the train split, evaluating on valid
   periodically.
5. **[src/run_fuse.py](src/run_fuse.py)**, merges a trained adapter
   permanently into a full copy of the base model's weights.
6. **[src/evaluate_model.py](src/evaluate_model.py)**, generates an
   answer for each test example (zero-shot, few-shot, or a fused
   fine-tuned model) and scores it with `values_agree`.
7. **[src/analyze_errors.py](src/analyze_errors.py)**, breaks every
   wrong answer down by failure kind (scaling error, wrong period, wrong
   row, unparseable) using the source project's own `classify()`, so
   "18.7% correct" isn't the only thing this project can say about a
   result.
8. **[src/significance_test.py](src/significance_test.py)**, McNemar's
   exact test for comparing two methods scored on the same examples,
   because all three methods here are paired observations, not
   independent samples, and treating them as independent would get the
   headline comparison wrong (see Results).
9. **[src/evaluate_ollama.py](src/evaluate_ollama.py)**, zero-shot only,
   for the model-size diagnostic below, against a model already on this
   machine via Ollama's local API rather than the MLX pipeline above.

## Environment problems hit building this (see PROGRESS.md for detail)

- `mx.distributed.init()`, called internally by mlx_lm regardless of
  whether this is a single-machine run, hard crashes (SIGABRT) on this
  machine's conda-installed MPICH. Worked around by monkey-patching it
  (and a second, independently crashing call, `mx.distributed.all_sum`)
  to force the single-process `ring` backend before importing mlx_lm's
  training and fuse entry points, see the top of
  [src/run_lora_training.py](src/run_lora_training.py) and
  [src/run_fuse.py](src/run_fuse.py).
- An early 6000-character input budget let some examples exceed the
  2048-token default training sequence length, truncating completions
  and producing NaN validation loss. Fixed by reducing to 3200 characters
  (comfortably above the 1391-token actual maximum in the training set)
  and setting `--max-seq-length 1536` explicitly.
- Training at batch size 4 / 8 trained layers hit a Metal
  "Insufficient Memory" error on this machine. Reduced to batch size 2 /
  4 trained layers, which trained cleanly every time afterward.
- Loading the base model with `adapter_path` set for repeated generation
  calls reproducibly stalled (memory climbing, no generation progress).
  Fusing the adapter into a full checkpoint once and loading that
  checkpoint as a plain model avoided the problem.
- The first attempt at evaluating the fine-tuned model at any scale
  larger than 20 examples repeatedly stalled during model loading.
  Diagnosed with `ps aux -m` rather than assumed: this machine had a
  dozen-plus browser tabs and a few other memory-heavy applications open
  at the time, genuine system-wide memory pressure unrelated to this
  project, not a bug here. The full 653-
  example runs below completed cleanly once retried when that pressure
  had cleared, nothing in the code changed between the stalled attempts
  and the successful one.

## Learning rate tuning

Four real training runs, all 439 train examples, rank 8 LoRA, batch size
2, 4 trained layers, max sequence length 1536:

| run | learning rate | iterations | val loss path |
|---|---|---|---|
| run 1 | 1e-5 (mlx_lm default) | 200 | 1.317 → 1.053, smooth but shallow |
| run 2 | 2e-4 | 200 | 8.401 → oscillates 3-5 → 2.916, never settles |
| **run 3** | **5e-5** | 300 | 8.401 → 1.235 (iter 50) → ~1.0-1.02, smooth |
| run 4 | 1e-4 | 300 | 1.973 → 0.955 (iter 200) → 1.006, smooth |

Checked on a 20-example sanity slice before committing to a full
653-example run on runs 1-3: run 1 produced predictions identical to the
untrained base model on 13 of 20 examples and scored 0/20, the same as
zero-shot, a learning rate too low to change actual generation behavior
in 200 iterations on this much data. Run 2's loss curve alone ruled it
out before evaluating it at all. Run 3 both converged smoothly and
changed the model's actual output behavior (3/20 correct against 0/20
for zero-shot, no identical-to-zero-shot predictions among its correct
answers). Every "fine-tuned" result below (unless labeled run 4) is run
3, fused into `adapters/fused_model_v3`.

Run 4's loss curve looks just as clean as run 3's on paper, smooth,
low, no oscillation, but scored 12/100 on the same 100-example slice
run 3 scores 21/100 on, a real, substantial regression despite the
well-behaved loss. A smooth validation loss curve is necessary but not
sufficient evidence a learning rate is good; run 2 was ruled out by loss
shape alone, but run 4 needed an actual generation-accuracy check to
catch, since its loss curve gave no warning. Not evaluated on the full
653 for this reason, worse on a 100-example check than the already-
established best rate is not worth the machine time to confirm at
scale.

## Results

All three methods scored on the identical 653-example test set:

| method | accuracy |
|---|---|
| zero-shot | 3.5% (23/653) |
| few-shot (3 demonstrations, no training) | 18.7% (122/653) |
| **fine-tuned (LoRA, run 3)** | **21.6% (141/653)** |

**Is fine-tuning actually better than few-shot, or is 21.6% vs 18.7%
noise?** Since all three methods were scored on the identical 653
examples, this is a paired comparison, not three independent samples, the same reason whisper-benchmark's own significance test exists.
McNemar's exact test on the discordant pairs (examples where the two
methods disagreed):

| comparison | p-value | conclusion |
|---|---|---|
| zero-shot vs few-shot | < 0.001 | few-shot is genuinely, significantly better |
| zero-shot vs fine-tuned | < 0.001 | fine-tuned is genuinely, significantly better |
| **few-shot vs fine-tuned** | **0.145** | **not significant, this data cannot distinguish them** |

The honest headline is not "fine-tuning wins": it's that putting 3
examples in the prompt, at zero training cost, closes most of the
distance from 3.5% (15.2 points, to 18.7%), and the further 2.9 points
from fine-tuning on top of that (18.7% → 21.6%) is not something this
653-example test set can confidently attribute to fine-tuning rather
than to noise. A fair statement of what was learned: **both few-shot
prompting and LoRA fine-tuning are dramatically better than zero-shot on
this task, and the gap between few-shot and fine-tuned specifically is
not established by this experiment.**

### What kind of mistake each method makes

Accuracy alone hides what changed. Every wrong answer is also classified
by failure kind (`src/analyze_errors.py`, reusing the source project's
own `classify()`):

| failure kind | zero-shot | few-shot | fine-tuned |
|---|---|---|---|
| correct | 3.5% | 18.7% | 21.6% |
| scaling error (right figure, wrong power of ten) | 35.1% | 7.5% | 1.5% |
| wrong period (right figure, wrong fiscal year) | 3.4% | 10.6% | 7.4% |
| wrong row (a different number entirely) | 55.3% | 60.9% | 69.4% |
| unparseable output | 2.8% | 2.3% | 0.2% |

This is the more interesting result than the accuracy row alone. Both
few-shot and fine-tuning almost eliminate scaling errors and unparseable
output, the model reliably learns "convert the scale, answer with one
bare number" from either 3 in-context examples or from training. Neither
one does much about the dominant failure mode, wrong row: picking a
different line item or figure from the table entirely. That looks like a
table-comprehension limit on a 3B model at this context length, not a
formatting problem, and it's the reason neither cheap fix (a few
examples) nor the more expensive one (LoRA) gets this task anywhere near
the source project's rules (0.7765) or API-model (0.9321) numbers,
measured admittedly on a different, much larger input.

## A table-formatting bug, and a model-size check (neither changes the headline table)

Two follow-up checks after the headline comparison above, kept separate
from it rather than blended in:

- **A real formatting bug, fixed, that turned out not to matter.**
  `data_prep.py`'s row-to-text rendering let a bare "$" and invisible
  zero-width-space characters ride along as their own pipe-separated
  cells on some rows but not others, misaligning column position between
  a table's header row and its data rows. Confirmed against real cached
  data and fixed (`data_prep._is_meaningful_cell`). Re-running the 3B
  zero-shot model on the corrected text (100 examples): 3.0% (3/100),
  against 3.5% on the old text, no real change, and the error-type
  breakdown (33% scaling, 63% wrong_row) is nearly identical to the old
  one too. The bug was real; it wasn't costing this model accuracy.
- **Model size, checked with what was already on disk.** `qwen2.5:14b`
  (downloaded earlier for sar-multiagent-drafting, not fetched for this
  project) scored **40.0% (40/100) zero-shot**, on the identical
  corrected 100 examples, via Ollama's local API
  (`src/evaluate_ollama.py`), roughly 13x the 3B zero-shot number on
  the same data. This is a capability check, not a fourth method in the
  headline comparison: Ollama serves GGUF, not the MLX format LoRA
  fine-tuning here uses, so no fine-tuning of it was attempted.

**Because of this**, `data/dev_examples.json`, `data/test_examples.json`,
and `data/mlx_format/*.jsonl` on disk right now reflect the corrected
table text, not the text the headline table's checked-in prediction
files (`predictions_zeroshot_base.json`, `predictions_fewshot_base_full.json`,
`predictions_finetuned_v3_full.json`) were actually computed against, running `data_prep.py` fresh today and re-scoring will not exactly
reproduce the headline numbers above, only the two follow-up numbers in
this section. The headline comparison is still internally consistent
(all three of its methods were scored on the same, old text); it's the
comparison between "headline" and "follow-up" sections that crosses a
data version and should be read as such.

## Actually fine-tuning a bigger model

The zero-shot check above made a strong prediction: model size, not
formatting, was the real lever, so fine-tuning a bigger MLX-format model
should beat the 3B fine-tuned result by a wide margin. Downloaded
`mlx-community/Qwen2.5-7B-Instruct-4bit` (4.3GB) and ran the same
pipeline, LoRA rank 8, 4 trained layers, learning rate 5e-5, 300
iterations, on the corrected table text, batch size dropped to 1 (from
the 3B run's 2) after the first attempt at batch size 2 hit the same
Metal "Insufficient Memory" error the original 3B tuning did at batch
size 4. Training converged smoothly (val loss 1.301 → 0.716, train loss
→ 0.362), the same well-behaved shape as the successful 3B run.

Evaluating the fused result on the full 653-example test set stalled
three times in a row with a distinct signature from every earlier memory
issue in this project: not a fast, obvious crash, but elapsed wall-clock
time climbing to 8-11 minutes while the process's own accumulated CPU
time stayed under 40 seconds, the process technically alive and not
crashing, just getting almost no actual CPU, the signature of the OS
thrashing something system-wide rather than this script hanging on its
own logic. A 100-example run completed cleanly in a few minutes,
consistently, which narrowed it down: whatever was thrashing correlated
with wall-clock process age (something else on the machine growing over
several minutes), not with example count or this model's memory
footprint at load time. Splitting the same 653 examples into 6 chunks of
~110, each its own fresh `evaluate_model.py` process (`--offset` added
for this), kept every chunk's elapsed process lifetime under the
observed stall threshold and all 6 completed without a single stall.
Concatenated their outputs into `predictions_finetuned_7b_full.json`,
checked to cover the identical 653 (doc_id, field) pairs in the same
order as every other method's result file, so it's directly comparable
to them, not a workaround that quietly changed what was being measured.

| method | accuracy |
|---|---|
| 3B fine-tuned (LoRA, run 3) | 21.6% (141/653) |
| **7B fine-tuned (LoRA, same recipe)** | **43.3% (283/653)** |

McNemar's exact test against both 3B-scale results the 7B fine-tuned
model could be compared to, no ambiguity this time, unlike the earlier
few-shot-vs-fine-tuned result:

| comparison | p-value |
|---|---|
| 3B fine-tuned vs 7B fine-tuned | 2.6 × 10⁻²⁶ |
| 3B few-shot vs 7B fine-tuned | 1.3 × 10⁻³³ |

Error breakdown for the 7B fine-tuned model: correct 43.3%, scaling
error 1.2%, wrong period 5.7%, wrong row 49.8%. Scaling error stays as
low as the 3B fine-tuned model's (both learn the scale-conversion
instruction well from training); wrong row is still the largest single
failure category, but its share of all examples dropped from the 3B
fine-tuned model's 69.4% to 49.8%, a real, if partial, improvement on
the table-comprehension problem the 3B analysis above could only
diagnose, not fix.

One honest caveat this section's comparison carries and the headline
table does not: `finetuned_7b_full` was trained and evaluated on the
corrected table text (the formatting-bug fix above), while
`finetuned_v3_full` and `fewshot_base_full` were computed on the old,
pre-fix text. The formatting fix's own, isolated effect on the 3B model
was checked separately above and found to be ~0 (3.5% → 3.0%), which is
why this comparison is presented as real rather than caveated away
entirely, but it is technically a second changed variable riding along
with model size, not a perfectly isolated one.

## The full 7B comparison, and the same few-shot finding, again, stronger

Only having 7B fine-tuned left an obvious hole: the original 3B headline
comparison's whole point was checking fine-tuning against zero-shot and
free few-shot prompting, and that check hadn't been repeated at 7B. Ran
both (same chunked-evaluation approach, zero stalls across all 12
chunks):

| method | accuracy (n=653) |
|---|---|
| 7B zero-shot | 30.9% (202/653) |
| 7B few-shot (3 demonstrations, no training) | 41.7% (272/653) |
| **7B fine-tuned** | **43.3% (283/653)** |

The 3B finding replicates, and gets stronger, not weaker, at the larger
scale:

| comparison | p-value | conclusion |
|---|---|---|
| 7B zero-shot vs 7B few-shot | 7.5 × 10⁻⁹ | few-shot genuinely, significantly better |
| 7B zero-shot vs 7B fine-tuned | 2.0 × 10⁻¹⁸ | fine-tuned genuinely, significantly better |
| **7B few-shot vs 7B fine-tuned** | **0.367** | **not significant, even more clearly than the 3B version of this comparison (p=0.145)** |

At 7B, few-shot prompting alone (41.7%) gets within 1.6 points of what
LoRA fine-tuning achieves (43.3%), and that gap is, if anything, LESS
distinguishable from noise than the 3B gap was. The honest conclusion
from both model sizes together: **the returns to actually training
something, over just showing a bigger model 3 examples in the prompt,
have not been established by this project at either scale tried**, what
clearly IS established, at both scales, is that either cheap-or-expensive
method beats zero-shot by a lot, and that model size matters enormously
regardless of which of the three methods it's paired with.

That last point is worth stating plainly: **7B zero-shot (30.9%) beats
3B fine-tuned (21.6%)**, significantly (p = 2.3 × 10⁻⁶ vs the 3B
fine-tuned result), a model that received no task-specific training at
all outperforms one that did, on nothing but scale. If the goal is the
best possible accuracy on this task under this project's local-only, no-
paid-API constraint, the evidence here says spend the available compute
on the biggest model that fits, before spending it on fine-tuning a
smaller one.

## Wrong_row isn't a flat ceiling, and part of it isn't the model's fault

Asked whether the dominant failure, wrong_row, is spread evenly or
concentrated (`src/analyze_errors.py --by field` / `--by company`).
Neither. By field, on the 7B fine-tuned model: `operating_cash_flow`
71.8%, `stockholders_equity` 66.7%, down to `revenue` 29.5%, the field
that's almost always the unambiguous first line of an income statement
fails least; subtotal fields with several similarly-sized neighboring
numbers fail most, at every method tried, not just this one. By company,
the same model: Microsoft Corporation 98.1% wrong_row (53/54) against
Cisco Systems' 24.1%, a company-to-company spread far too large to be
random variation in table difficulty.

Investigated Microsoft's number rather than reporting it as a curiosity:
its `statement_text` for the FY2023 10-K turns out to contain no income
statement, balance sheet, or cash flow statement content at all, just
footnote schedule tables (an allowance-for-doubtful-accounts rollforward,
an unrecognized-tax-benefits reconciliation) that the source project's
own table parser tagged as one of the three financial statement types.
Not a hard table the model read wrong; no right answer anywhere in what
it was given.

Checked how far this extends with `src/audit_input_coverage.py`: does
the true value appear anywhere in the given table text at all, at any of
the common financial-statement scales, independent of whether any method
got it right. **89 of 653 test examples (13.6%) have no plausible
representation of their answer anywhere in the text the model was
given**, concentrated exactly where the wrong_row rate is worst:
`total_assets` 42.3%, `stockholders_equity` 30.8% of that field's own
examples unanswerable, against `net_income` 2.6% and `cash_and_equivalents`
1.3%. 44 of 78 test filings have at least one unanswerable field
(usually a couple, never all nine, the real statement content exists in
the filing, it's just not what got classified as "the" balance sheet or
cash flow statement for that filing, or it exists past the 3200-character
budget this project spends on non-footnote content). This is inherited
from the source project's table classifier, not introduced by anything
in this project's own code, and was never checked in either project
until this row/company-level anomaly investigation prompted it.

Recomputing accuracy excluding these 89 unanswerable examples, the
fairer number to credit or blame a model for, separate from a data
problem no amount of model capability fixes:

| method | raw accuracy (n=653) | achievable accuracy (n=564) |
|---|---|---|
| 3B zero-shot | 3.5% | 3.9% |
| 3B few-shot | 18.7% | 21.6% |
| 3B fine-tuned | 21.6% | 25.0% |
| 7B zero-shot | 30.9% | 35.6% |
| 7B few-shot | 41.7% | 47.7% |
| **7B fine-tuned** | **43.3%** | **50.2%** |

The gap between raw and achievable accuracy is small for the weaker
methods (they were failing for their own reasons well before running out
of answerable examples to get right) and largest for the best methods, every 7B method gains 4.7-6.9 points, more than any 3B method, meaning a
meaningfully larger share of the stronger models' remaining errors,
specifically, are against genuinely unanswerable questions rather than
model mistakes.
`value_plausibly_present()`'s own limits are stated in its docstring: a
heuristic substring check at common scales, not an exact match to
whatever rounding the filer's XBRL tag carries versus what's printed, so
this likely understates true coverage problems slightly rather than
overstating them.

## Running it

Requires the source project (`filing-extraction-benchmark`) present as a
sibling directory with its `results/ground_truth.csv`, `results/split.csv`,
and `data/cache/prepared/*.json` already built, this project reads
those, read-only, and does not regenerate them.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/data_prep.py            # writes data/dev_examples.json, test_examples.json
python src/format_for_training.py  # writes data/mlx_format/{train,valid,test}.jsonl

python src/run_lora_training.py --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train --data data/mlx_format --adapter-path adapters/lora_v3 \
    --iters 300 --learning-rate 5e-5 --batch-size 2 --num-layers 4 \
    --max-seq-length 1536 --save-every 100

python src/run_fuse.py --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path adapters/lora_v3 --save-path adapters/fused_model_v3

python src/evaluate_model.py --label zeroshot_base                       # zero-shot, full test set
python src/evaluate_model.py --label fewshot_base_full --few-shot-k 3    # few-shot, full test set
python src/evaluate_model.py --model-path adapters/fused_model_v3 \
    --label finetuned_v3_full                                            # fine-tuned, full test set

python src/analyze_errors.py zeroshot_base fewshot_base_full finetuned_v3_full
python src/significance_test.py fewshot_base_full finetuned_v3_full

# The 7B run, same recipe, batch size 1 (7B needs more memory per step
# than the 3B run's batch size 2 could fit), and evaluated in chunks
# rather than one 653-example run; see "Actually fine-tuning a bigger
# model" above for why.
python src/run_lora_training.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --train --data data/mlx_format --adapter-path adapters/lora_7b_v1 \
    --iters 300 --learning-rate 5e-5 --batch-size 1 --num-layers 4 \
    --max-seq-length 1536 --save-every 100

python src/run_fuse.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path adapters/lora_7b_v1 --save-path adapters/fused_model_7b

for i in 0 1 2 3 4 5; do
  python src/evaluate_model.py --model-path adapters/fused_model_7b \
      --label "finetuned_7b_chunk${i}" --offset $((i * 110)) --limit 110
done
# then concatenate the 6 predictions_finetuned_7b_chunk*.json files, in
# order, into predictions_finetuned_7b_full.json

# 7B zero-shot and few-shot, same chunked approach, no training involved
for i in 0 1 2 3 4 5; do
  python src/evaluate_model.py --model-path mlx-community/Qwen2.5-7B-Instruct-4bit \
      --label "zeroshot_7b_chunk${i}" --offset $((i * 110)) --limit 110
  python src/evaluate_model.py --model-path mlx-community/Qwen2.5-7B-Instruct-4bit \
      --label "fewshot_7b_chunk${i}" --offset $((i * 110)) --limit 110 --few-shot-k 3
done
# concatenate each set of 6 chunks the same way, into
# predictions_zeroshot_7b_full.json and predictions_fewshot_7b_full.json

python src/analyze_errors.py finetuned_7b_full --by field
python src/analyze_errors.py finetuned_7b_full --by company
python src/audit_input_coverage.py    # the 89/653 unanswerable-example finding, and achievable accuracy
```

Run `pytest` from the project root for the test suite (parsing, data
prep integrity, few-shot demonstration selection, error classification,
the significance test, and the numeric claims in this README, checked
against the actual result files, no test loads the MLX model itself,
since that would make the suite depend on this machine's memory state
and take minutes rather than seconds).

## What's left undone

- A fourth learning rate, 1e-4, was tried (see "Learning rate tuning"
  above) and confirmed 5e-5 as the better choice rather than left
  unchecked, but the gap between 5e-5 and 2e-4 above 1e-4 wasn't
  further narrowed; there could be a rate between 1e-4 and 2e-4 that
  does better than either, not tried.
- The wrong-row failure mode (the dominant one at every method tried,
  49.8-69.4% of all examples) improved substantially with model size
  (7B fine-tuned) but was not eliminated. Broken down by field and
  company (see "Wrong_row isn't a flat ceiling" above) and found to be
  concentrated, not uniform, and 13.6% of it turned out to not be the
  model's fault at all (the answer wasn't in the given text). What
  wasn't done: fixing the upstream table-classification issue this
  surfaced in the source project (filing-extraction-benchmark's own
  parser tagging footnote schedules as financial statements for some
  filings), flagged here, not fixed there, since that's a change to a
  different project's code with its own test suite and conventions.
  Also not done: whether the REMAINING wrong_row failures (on examples
  confirmed answerable) cluster by table layout or company beyond what
  the unanswerable-example filtering already explains.
- The root cause of the chunked-evaluation stall (elapsed time climbing
  far past accumulated CPU time on a long-lived `evaluate_model.py`
  process, worked around by running in several short-lived processes
  instead of diagnosed to its actual source) was not identified, some
  other process on this machine growing over several minutes is the
  working theory, not confirmed by inspecting what specifically grew.
- More few-shot demonstrations (5, 10) were not tried at either model
  size; 3 was picked to keep the prompt short, not because it was swept
  against other values.
- Whether an even bigger model (14B+, MLX format, actually fine-tuned
  rather than the zero-shot-only Ollama check) would show the same
  few-shot-catches-up-to-fine-tuned pattern a third time, or whether
  that pattern itself has a ceiling, is unknown, only 3B and 7B were
  actually fine-tuned and compared this way.
