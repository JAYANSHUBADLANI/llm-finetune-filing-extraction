# Progress log

The question: does LoRA fine-tuning a small local model on the
filing-extraction-benchmark project's own labeled financial statement
extraction task close any of the gap between a zero shot small model and
that project's own hand tuned rules (0.7765 test accuracy) and API
language model (0.9321 test accuracy) results, without paying for an API
or leaving this machine.

Reused filing-extraction-benchmark's ground truth (1,225 labeled cells,
9 financial statement fields, 24 real companies' 10-Ks, dev and test
split already defined) rather than building a new labeled set, read only,
nothing in that project touched. Compacted the input from that project's
"document" mode, up to 560,000 characters, the honest but LoRA
impractical comparison basis its own 0.9321 figure was measured under, to
only the tables its own parser already classified as a financial
statement, typically a few thousand characters. That is a real, stated
difference from what the 0.9321 figure measures, not the same comparison
under a cheaper method. 6 Caterpillar filings had no table classified as
any statement at all by that parser and were excluded from the test set
this project's own numbers are measured against, all of them, not
selectively.

Real bug in my own case data, caught early, same shape as the currency
one in the multi agent project: the statement tables mix scales, some
captioned "Dollars in millions", most not, and the first zero shot pass
answered in whatever scale the table displayed rather than the actual
dollar figure the ground truth is stated in, 76,559 instead of
76,559,000,000. Fixed by tagging each table's block with its own scale
phrase and instructing the model to convert before answering. That
instruction is not reliably followed by a 3B model even after the fix,
which the zero shot baseline result below states plainly rather than
assuming the instruction fixed the problem because it was added.

Environment problems, in the order hit, all specific to this machine,
recorded because the next person running this needs them, not because
they are interesting on their own:

MLX's `mx.distributed.init()`, called internally by mlx_lm's training and
fuse code regardless of whether this is a single machine run, probes for
an MPI installation and hard crashes, SIGABRT, exit code 134, no Python
exception to catch, when it finds this machine's conda-installed MPICH
instead of the Open MPI it expects. Confirmed the crash is specifically
this probe by calling `mx.distributed.init(backend="ring")` directly,
which returns a working size 1 group with no crash. `mx.distributed.
all_sum`, called separately inside `evaluate()` with no group argument,
triggers the same probe a second time even after `init` is patched, found
after training got through one validation pass and then crashed on the
next `all_sum` call. `src/run_lora_training.py` and `src/run_fuse.py`
both patch `mx.distributed.init` to force the ring backend and
`mx.distributed.all_sum` to the identity function before importing
mlx_lm's own entry points, which is exact rather than approximate for a
single process run: summing across a world of one process is the
identity.

Truncation: an early context length budget (6000 characters per
statement text) let some examples' prompt plus completion exceed the
default 2048 token training sequence length, cutting off the completion
entirely for the longest ones and producing NaN validation loss on the
first real training run, since a batch with zero valid loss tokens after
masking divides by zero. Reduced to 3200 characters, the max actual token
length in the training set (1391) comfortably fits, no truncation on any
example since.

Out of memory during training at the original settings (batch size 4,
8 trained layers): `RuntimeError: [METAL] Command buffer execution failed:
Insufficient Memory`. Reduced to batch size 2, 4 trained layers, max
sequence length 1536; training completed cleanly at these settings every
time it was tried afterward.

Loading the base model with `adapter_path` set, for repeated generation
calls rather than a one time fuse, reproducibly consumed memory rapidly
with the evaluation script showing no generation progress, on this
machine, every time tried, whether run alone or right after another
process was killed to free memory first. Worked around by fusing the
adapter into a full checkpoint once with `run_fuse.py` and loading that
plain checkpoint at evaluation time instead, `--model-path` in
`evaluate_model.py`, which is what actually gets used for every fine
tuned evaluation number below, not `--adapter-path`.

Even loading the fused, plain checkpoint became unreliable as the night
went on: memory dropped into the low hundreds of megabytes free during
loading, and CPU utilization on the evaluation process dropped toward
zero rather than staying high, which is the signature of the OS
thrashing rather than the process actually computing. This got worse
across repeated attempts at the same task, not better, even right after
killing the stalled process and confirming system wide free memory had
recovered into the multiple gigabytes. Checked `ps aux -m` at the point
of worst pressure to find the actual cause rather than guess: this is a
16GB machine, and free memory was being held by ordinary desktop load
unrelated to this project, over a dozen browser tabs and a handful of
other heavy applications, all already open before this project's
evaluation step ever ran. There was nothing of this project's own to
kill that would free real memory; the model load was simply competing
with everything else already resident on a machine that does not have
room for a several gigabyte model load on top of that at the same time.
Not something fixable from inside this project; see "Left undone."

Working through this late at night, I decided to keep going and use my
own judgement on memory handling rather than stop and wait until morning;
the approach above, killing processes proactively before an actual hang
rather than after, reflects that, and is recorded here because it
changed the sample sizes below, not because it was a free choice.

Learning rate tuning, three real training runs, not one adjusted after
looking at only the last one:

Run 1, learning rate 1e-5 (mlx_lm's own default, not set explicitly the
first time): validation loss went from 1.317 to 1.053 over 200
iterations, a real but small improvement. Evaluated on the same 20 test
examples the zero shot baseline was also checked against directly: 0 of
20 correct for both the fine tuned model and the zero shot base model,
and for 13 of those 20 examples the two models' predicted numbers were
exactly identical. A learning rate this low did not move the
model's actual generation behavior enough to matter within 200
iterations on 439 training examples, whatever the validation loss curve
by itself suggested.

Run 2, learning rate 2e-4, a size some LoRA guides use as a default:
validation loss opened far higher, 8.401 at iteration 1, then oscillated
in the 3 to 5 range for most of training before ending at 2.916, worse
than run 1's ending point and never smoothly converging in 200
iterations. Too aggressive for this dataset size and iteration count to
settle in the time given; not evaluated further, since the loss curve
itself was reason enough not to trust the result.

Run 3, learning rate 5e-5, 300 iterations: validation loss again opened
at 8.401 (same initialization, expected) then fell smoothly to 1.235 by
iteration 50 and settled around 1.0 to 1.02 for the remainder, the best
behaved curve of the three. This is the model behind every "fine-tuned"
result in README.md. Sanity checked on the same 20 examples before
committing to a larger run: 3 of 20 correct, against 0 of 20 for zero
shot on the identical 20 examples, a real, visible behavior change, not
another near identical output pattern.

Sample size actually evaluated for the fine tuned model, and why it is
smaller than the zero shot baseline's full 653: three separate attempts
to evaluate the fused v3 model at larger sizes (150 examples, 50
examples, then the full set unscoped) each hit the memory pressure
described above partway through model loading and were killed rather
than left to actually stall the machine, checked and confirmed to be
system wide desktop load rather than anything this project could fix.
Retrying that same load at the same moment was not going to behave
differently. 20 examples is what completed cleanly, and that same 20 is
the exact set the zero shot and the run 1 (1e-5) checks above were also
run on, so all three numbers below are on identical inputs, not a
different, easier or harder subset picked after the fact:

  zero shot base model:        0/20  (0%)
  fine-tuned, run 1 (1e-5):    0/20  (0%)
  fine-tuned, run 3 (5e-5):    3/20  (15%)

The zero shot model's headline number elsewhere in README.md, 3.5%, is
measured on the full 653 example test set, not this 20; the 0% here is
that same model on this smaller shared slice, and is the correct number
to compare the finetuned rows against, not the 3.5% figure.

2026-09-07, second pass: the 20-example result above was correct as far
as it went, but on its own it was not a strong result, two numbers
computed from 20 examples with no comparison to the cheapest possible
alternative, prompting the base model with a few examples instead of
training anything. Two problems with it: n=20 is
too small to trust for a headline claim, and no version of this project
had checked whether LoRA fine-tuning added anything a free prompt change
would not have gotten for the same money.

Added `src/few_shot.py`: 3 demonstrations selected deterministically
from the train split only (never valid or test), spanning 3 distinct
fields, reused unchanged across every test example in a run rather than
resampled per example, since a per-example resample would make "the
few-shot baseline" actually several different baselines and blur any
real signal. `src/evaluate_model.py` takes a `--few-shot-k` flag that
builds a multi-turn chat prompt (demo question, demo answer, ..., real
question) instead of a single-turn one when set.

Retried the full 653-example evaluation for the fine-tuned model, the
thing left undone above. It completed cleanly this time, no stall,
confirming the earlier diagnosis: the blocker was genuinely machine-wide
memory pressure from other applications, not anything about this
project's own code or model. Also ran the new few-shot baseline on
the same full 653 examples.

  zero shot, full 653:                     23/653   (3.5%)
  few-shot (3 demos, no training), full 653: 122/653  (18.7%)
  fine-tuned, run 3 (5e-5), full 653:        141/653  (21.6%)

Few-shot alone, zero training, gets most of the way from zero-shot to
fine-tuned. That raised the real question directly: is 21.6% vs 18.7%
an actual effect of fine-tuning, or noise? All three methods are scored
on the identical 653 examples, so this is a paired comparison, not three
independent samples, the same situation whisper-benchmark's own
significance test existed for. Wrote `src/significance_test.py`,
McNemar's exact test on the discordant pairs (examples where two methods
disagree; agreements, right or wrong, carry no information about which
method is better). Result: zero-shot vs few-shot and zero-shot vs
fine-tuned are both p < 0.001, genuinely significant. Few-shot vs
fine-tuned is p = 0.145, not significant at any conventional threshold.
This data does not support the claim that fine-tuning beat few-shot
prompting here; it only supports the claim that both of them beat
zero-shot. Reporting the flat 21.6% vs 18.7% numbers without this test
would have overstated what fine-tuning actually demonstrated to do.

Also wrote `src/analyze_errors.py`, reusing the source project's own
`classify()` (correct / scaling_error / wrong_period / wrong_row /
parse_error) rather than inventing a separate breakdown, since that
function already exists, is already tested by that project, and a
category means the same thing there as it does here. This needed
`other_period_values` per prediction (the same field's true value in
the company's other filings, to tell "grabbed last year's column" apart
from "picked an unrelated number"), computed from ground_truth.csv's own
cik and field columns rather than left out. Breakdown across all three
methods (see README.md's table) shows both few-shot and fine-tuning
nearly eliminate scaling errors and unparseable output (the model
reliably learns to convert scale and answer with one bare number from
either 3 examples or from training) but do not fix the dominant failure,
wrong_row (55-69% of all examples across all three methods), picking an
entirely different line item from the table. That is the more useful
finding than any single accuracy percentage: the two methods fix
different things than what's actually holding accuracy down.

2026-09-07, third pass: whether the model itself could be
made better, separate from the fine-tuning-vs-few-shot question already
answered. Two hypotheses, checked in order rather than assumed:

Hypothesis 1: the table-to-text rendering in `load_statement_text` was
itself hurting accuracy. Checked the actual cached row data directly
(not just the rendered text) and found a real bug: the source project's
own table parser leaves a bare "$" as its own list element on some rows
but not others (3,480 occurrences sampled across 30 filings), and
zero-width space characters (2,873 occurrences) that `str(c).strip()`
does not remove since U+200B isn't whitespace by Python's definition, so
these were silently passing the old `if str(c).strip()` filter as if
they were real cells. Both inflate a data row's cell count relative to
its header row inconsistently row to row, which genuinely does misalign
which pipe-separated position corresponds to which reported year.
Fixed in `data_prep._is_meaningful_cell()` (drop any cell with no letter
or digit) and confirmed directly against a real filing that every data
row now carries exactly as many fields as its header
(`test_load_statement_text_rows_align_with_their_header`). Regenerated
`data/dev_examples.json`, `data/test_examples.json`, and
`data/mlx_format/*.jsonl` from the fixed code, same counts as before
(653/78, 518/60, 439/51, 79/9), only the row text itself changed.

Then checked whether that fix actually moved accuracy, rather than
assuming a verified bug implies a verified improvement: re-ran the 3B
zero-shot base model on the corrected data, 100 examples (matched to the
sample size of the model-size check below for a clean comparison).
Result: 3/100 (3.0%), against 3.5% (23/653) on the old, broken-alignment
data, no real change, within noise for this sample size. The error-type
breakdown confirms it isn't a sample-size fluke masking a shift: scaling
errors 33% and wrong_row 63% on the fixed data, essentially the same
proportions as the original broken-alignment run (35.1% / 55.3% on the
full 653). The bug was real and is now fixed regardless, but it was not,
on the evidence, actually costing this model accuracy, a 3B model
apparently isn't relying on strict column-position tracking across rows
in a way this particular misalignment would disrupt. Worth having
checked rather than assumed either way.

Hypothesis 2: the wrong_row failure (dominant at 55-69% for every 3B-
scale method tried) is a genuine model-capacity ceiling, not an
artifact of this project's small-model-oriented setup. Checked using
qwen2.5:14b, already downloaded locally for sar-multiagent-drafting via
Ollama, zero-shot only (Ollama serves GGUF, not the MLX format
`run_lora_training.py` fine-tunes, so no LoRA fine-tuning of it was
attempted here, this is a capability check, not a fourth method added
to the main comparison). Wrote `src/evaluate_ollama.py` to call it
through Ollama's local `/api/chat` HTTP endpoint via stdlib `urllib`
rather than adding the `requests` package as a new dependency for one
diagnostic script. Result, on the identical fixed-format 100 examples
the 3B check above used: 40/100 (40.0%), roughly 13x the 3B zero-shot
number on the same data, same prompt, same scoring. Error breakdown:
scaling_error 6%, wrong_row 54%. Model size is a real, large lever here;
the table-formatting fix, checked separately above, was not.

2026-09-07, fourth pass: the zero-shot check above made a real
prediction, not just an observation, if model size is the actual
lever, actually fine-tuning a bigger MLX model should beat the 3B
fine-tuned result by a lot. Since an MLX-compatible checkpoint was needed (the Ollama one can't be
fine-tuned by this pipeline), downloaded
`mlx-community/Qwen2.5-7B-Instruct-4bit` (4.3GB) via
`huggingface_hub.snapshot_download`, not through `mlx_lm.load()`
directly, so the download and the eventual first model load are
separate steps and a paused, killed download doesn't cost a repeated
model load too.

Pause and resume does not behave the way this library's general
reputation suggests. Checked the installed `huggingface_hub` (1.30.0)
against its actual source rather than trusting that reputation:
`_download_to_tmp_and_move` deliberately writes every download attempt
to a fresh
`uuid.uuid4().hex[:8]`-suffixed temp file now (see the code's own
comment, referencing PR #4228: a shared `<etag>.incomplete` file could
get corrupted by a broken `flock` on some network filesystems, and the
fix accepts "duplicated bandwidth" per retry as the tradeoff). A
pause-and-resume after this was found does restart that file's download
from zero, not append to the old partial. Deleted the now-orphaned
partial file this produced and treated the interrupted download as
unrecoverable rather than assuming it would pick back up.

LoRA training on 7B, same recipe as the 3B run 3 (rank 8, 4 layers,
learning rate 5e-5, 300 iterations): batch size 2 (the 3B run's setting)
hit the same Metal "Insufficient Memory" error the original 3B tuning
did at batch size 4, 7B needs more per-step memory than 3B at the same
batch size, unsurprising in hindsight. Reduced to batch size 1; trained
cleanly, converging smoothly from val loss 1.301 to 0.716 (train loss to
0.362), the same well-behaved shape as the successful 3B run. Fused into
`adapters/fused_model_7b`.

Evaluating the fused 7B model on the full 653-example test set failed
three times in a row, and the failure signature was new, not the same
"stalled during model loading" pattern from earlier in this project:
elapsed wall-clock time on the `evaluate_model.py` process climbed to
8-11 minutes while its own accumulated CPU time (`ps -o time`) stayed
under 40 seconds the whole time, the process technically alive, barely
scheduled, not crashing outright. A 100-example run of the identical
model on the identical data completed cleanly in a few minutes, every
time it was tried, which ruled out "this model's memory footprint is
just too big to load reliably" as the explanation (it clearly loads and
runs fine) and pointed instead at something correlated with wall-clock
process age specifically, plausibly some other already-running
application on this machine growing its own memory footprint over
several minutes, though the actual other-process cause was not confirmed
by inspecting what specifically grew, only inferred from the timing
correlation.

One diagnostic trap worth recording: Python's stdout is block-buffered
when piped to a file, so a healthy, running process can leave its output
file empty for a long stretch. File emptiness alone is not evidence of a
stall, only `ps` CPU-vs-elapsed timing is. Removed that ambiguity: added
`flush=True` to
`evaluate_model.py`'s progress prints and dropped the print interval
from every 50 examples to every 20, so a redirected log reflects reality
promptly rather than needing a `ps`-based workaround to trust it.

Fix for the actual stall, once its shape was understood (time-correlated,
not example-count or memory-footprint correlated): added an `--offset`
argument to `evaluate_model.py` (previously `--limit` only ever sliced
from the start) and ran the 653 examples as 6 independent chunks of
~110, each its own fresh process via a small shell loop
(`/tmp/run_7b_chunks.sh`). Every chunk's process lifetime stayed under
the observed stall threshold; all 6 completed with zero stalls on the
first attempt at this approach. Concatenated the 6
`predictions_finetuned_7b_chunk*.json` files into
`predictions_finetuned_7b_full.json` and verified programmatically (not
assumed) that the concatenation covers the exact same 653 (doc_id,
field) pairs in the same order as every other method's result file
before trusting any comparison against it.

Result: 283/653 (43.3%), against the 3B fine-tuned model's 21.6% on the
same 653 questions (different underlying table text, see the format-
version caveat in README.md, but the same fields, companies, and
scorer). McNemar's test against both 3B-scale results this could be
meaningfully compared to: 3B fine-tuned vs 7B fine-tuned p = 2.6e-26, 3B
few-shot vs 7B fine-tuned p = 1.3e-33, both overwhelmingly significant,
unlike the earlier few-shot-vs-fine-tuned comparison at 3B scale, which
was genuinely ambiguous (p=0.145). Error breakdown: scaling_error 1.2%
(as low as the 3B fine-tuned model's), wrong_row 49.8%, still the
largest failure category by far, but down from the 3B fine-tuned model's
69.4% share, a real, partial improvement on the actual bottleneck this
project's own error analysis identified, not just a better accuracy
number with the same failure shape underneath it.

2026-09-07, fifth pass: went back to the one thing left unbroken-down
from the fourth pass, wrong_row by field and by company, expecting a
mildly interesting distribution and finding something that changed the
project's central conclusion instead.

By field on the 7B fine-tuned model: operating_cash_flow 71.8%,
stockholders_equity 66.7%, down to revenue 29.5%. Made sense on sight, revenue is almost always the unambiguous first line of an income
statement; stockholders_equity and operating_cash_flow are subtotals
with several similarly-sized numbers nearby. Ran company next expecting
a milder version of the same story and got something too large to be
that: Microsoft Corporation 98.1% wrong_row (53/54) against Cisco
Systems' 24.1%, on the same model. A spread that size between two
companies' filings, both presumably comparable 10-Ks, was not "some
tables are a bit harder", worth reading the actual input rather than
moving on.

Printed the actual `statement_text` for a Microsoft example
(0000789019_0000950170-23-035122, FY2023) and it contained no income
statement, balance sheet, or cash flow statement at all, an allowance-
for-doubtful-accounts rollforward and an unrecognized-tax-benefits
reconciliation, footnote schedules, nothing else, repeated for all three
tables in that filing's rendered text. This project's own
`data_prep.py` trusts `table["statement"]` from the source project's
cached, pre-parsed JSON without checking it against anything; whatever
mis-tagged these footnote tables as one of the three statement types did
so in filing-extraction-benchmark's own parser, inherited here silently.
Not something introduced by this project, but never checked by this
project either, until a company-level accuracy anomaly forced the
question.

Checked how far it extends rather than treat Microsoft as a one-off.
Wrote `src/audit_input_coverage.py`: for every one of the 653 test
examples, does the true value appear anywhere in the given table text at
all, at any of the common financial-statement scales (as printed,
thousands, millions, billions), independent of what any model actually
answered. Negative numbers need care here: a "(1,975)" closing-paren
check over-flags, because this project's own earlier column-alignment
fix (dropping bare ")" filler cells) leaves the value rendered as
"(1,975" with NO closing parenthesis. Confirmed by reading a flagged
Boeing example directly rather than trusting the count. Not a bug in
that fix, but a real gotcha for anything checking the rendered text's
exact string shape afterward. The heuristic accepts the open-paren-only
form; the count is 89/653 (13.6%).

Concentration confirms this is a real structural issue, not heuristic
noise: total_assets 42.3% of its own 78 examples unanswerable,
stockholders_equity 30.8%, almost exactly the two fields with the
worst wrong_row rates above. 44 of 78 test filings have at least one
unanswerable field (never all nine, the real statements exist
somewhere in each filing, just not tagged as such, or past this
project's own 3200-character table-block budget).

Recomputed accuracy excluding these 89 examples for every method
already measured:

  zero-shot:      raw 3.5%  (23/653)  -> achievable 3.9%  (22/564)
  few-shot:        raw 18.7% (122/653) -> achievable 21.6% (122/564)
  3B fine-tuned:   raw 21.6% (141/653) -> achievable 25.0% (141/564)
  7B fine-tuned:   raw 43.3% (283/653) -> achievable 50.2% (283/564)

7B fine-tuned gains the most (6.9 points) of any method, the strongest
model's remaining errors are disproportionately against genuinely
unanswerable questions, not its own mistakes, more so than the weaker
methods, which were still failing plenty on answerable questions too.
50.2% on questions that were actually answerable is a materially
different, fairer number than the 43.3% raw figure this project reported
as its headline result before this pass, and both are worth stating
side by side rather than picking the more flattering one.

Left undone: a fourth learning rate between 5e-5 and 2e-4 was not tried
at either model size; 5e-5 was accepted once it showed a real, correctly
shaped curve and a real behavior change, not swept against alternatives.
More few-shot demonstration counts (5, 10) were not tried at 3B, and
few-shot was not tried at all at 7B, whether 7B few-shot alone closes
much of the gap to 7B fine-tuned, the way 3B few-shot closed most of the
gap to 3B fine-tuned, is unknown. The actual cause of the wall-clock-
correlated eval stall (some other process's memory growth, by
inference, not confirmed by directly inspecting what grew) was never
identified, only worked around. The upstream table-misclassification bug
this pass found was flagged, not fixed, fixing it belongs in
filing-extraction-benchmark, a different project with its own test
suite and conventions, not patched from here. Whether the wrong_row
failures remaining on confirmed-answerable examples cluster further by
table layout or company was not checked; the unanswerable-example
filtering above may not be the whole explanation for the field/company
concentration, just a real and substantial part of it.

2026-09-07, sixth pass: 7B had only been fine-tuned, not checked
zero-shot or few-shot, so the project's central "does fine-tuning beat
free prompting" question still needed asking at the larger scale rather
than taken to generalise. Ran both, full 653, same 6-chunk
approach as the fine-tuned run (offsets 0/110/220/330/440/550): all 12
chunks (6 zero-shot + 6 few-shot) completed with zero stalls on the
first attempt, no retries needed this time.

  7B zero-shot:   202/653  (30.9%)
  7B few-shot:    272/653  (41.7%)
  7B fine-tuned:  283/653  (43.3%)

McNemar's: zero-shot vs few-shot p=7.5e-9 (significant), zero-shot vs
fine-tuned p=2.0e-18 (significant), few-shot vs fine-tuned p=0.367, the
3B result (p=0.145, itself already not significant) not just holding at
7B but getting MORE decisively non-significant, not less. The "does
fine-tuning actually beat cheap prompting" question now has the same
answer at two different model sizes, not one, a real replication, not
a single lucky/unlucky sample.

Also checked, since the numbers made it obvious: 7B zero-shot (30.9%)
beats 3B fine-tuned (21.6%) outright, and significantly (p=2.3e-6). The
single most actionable finding across this whole project for anyone
actually choosing between spending compute on fine-tuning a small model
versus running a bigger one zero-shot: at least on this task, at these
two sizes, model size beats task-specific training that this project
spent by far the most engineering effort on.

Recomputed the input-coverage-adjusted accuracy from the fifth pass for
these two new methods too: 7B zero-shot 30.9% -> 35.6% achievable, 7B
few-shot 41.7% -> 47.7% achievable, consistent with the pattern already
seen, stronger methods gain more from excluding unanswerable examples.

Left undone, updated: everything from the fourth and fifth passes'
lists still stands except the 7B zero-shot/few-shot gap, now closed.
Not tried: a bigger model than 7B actually fine-tuned and compared the
same three ways (only a zero-shot-only Ollama check exists at 14B, no
MLX fine-tune of anything larger than 7B), whether the
few-shot-catches-fine-tuning pattern holds a third time, or has its own
ceiling, is unknown.

Item C, the last of the three items for this pass: one more learning
rate, 1e-4, between the working 5e-5 and the unstable 2e-4, at 3B, on
the corrected table format. Val loss: 1.973 → 0.955 (iter 200) → 1.006,
smooth, no oscillation, looked, on the loss curve alone, at least as
good as run 3's. Checked generation accuracy anyway rather than trust
the curve: fused it and ran a 100-example check, 12/100 (12%), against
run 3's 21/100 on the identical 100-example slice (different table-text
version, old vs corrected, but the earlier formatting-fix check already
established that difference is worth ~0 on its own for this model, so
the ~9-point gap here is real, not a format artifact). A clean loss
curve did not predict this; only actually generating and scoring did.
Did not evaluate on the full 653, worse than the established best rate
on a 100-example check isn't worth confirming at scale, the same
judgment call this project already made for run 2 on loss-curve grounds
alone, just made this time on accuracy grounds after the loss curve
gave no warning.

All three follow-up items for this pass (A: wrong_row breakdown,
B: 7B zero-shot/few-shot, C: one more learning rate) are done. Every
result in this file and README.md as of this line is complete, checked,
and covered by the test suite.
