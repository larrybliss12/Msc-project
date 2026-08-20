# Run Guide: Claude CLI prompt, dataset access, and Colab steps

---

## PART 1 — Where the dataset is

There is **no dataset file in the repository**. EdAcc is downloaded at runtime.
Nothing is bundled because the corpus is ~40 hours of audio under a CC-BY-SA
licence that requires you to accept terms in your own account.

### Primary source (what the code uses)

| | |
|---|---|
| Hugging Face dataset ID | `edinburghcstr/edacc` |
| Set in | `src/config.py`, field `dataset_id` |
| Licence | CC-BY-SA 4.0 |
| Size | ~40 h total; ~29 h in the released dev + test partitions |
| Access | May require accepting terms on the dataset page while signed in |

> **Verify this ID before your first run.** I could not reach Hugging Face from
> the environment where this code was written, so the ID has not been confirmed
> against a live response. Open `https://huggingface.co/datasets/edinburghcstr/edacc`
> in a browser. If it 404s, search Hugging Face for "EdAcc" or "Edinburgh
> accents" and put the correct ID in `src/config.py`, or pass
> `--dataset-id <correct/id>` on the command line. The pipeline raises a clear
> error rather than silently producing nothing, so a wrong ID fails loudly at
> stage 1.

### Fallback source

If the Hugging Face route is unavailable, EdAcc is distributed directly by the
University of Edinburgh through Edinburgh DataShare. Search for "Edinburgh
International Accents of English Corpus DataShare". That route gives you audio
plus metadata files rather than a `datasets` object, and would require writing a
small local loader in `src/data.py`. Try Hugging Face first.

### Authentication

```python
from huggingface_hub import login
login()          # paste a token from huggingface.co/settings/tokens
```

Only needed if stage 1 fails with a 401 or a gated-dataset error.

---

## PART 2 — What to do in Colab

### Step 1: GPU runtime

Runtime → Change runtime type → **T4 GPU** (or better). Without a GPU the
inference stage is impractically slow. The audit and analysis stages run fine on
CPU.

### Step 2: Get the code into Colab

Upload `edacc-fairness.zip` using the Files pane in the left sidebar, then open
`notebooks/EdAcc_Study_Colab.ipynb`. The first cell unzips it and sets the
working directory.

Alternative, if you push the repo to GitHub:

```python
!git clone https://github.com/<you>/edacc-fairness.git
%cd edacc-fairness
```

### Step 3: Run the cells in order

| Cell | What it does | Time | Needs GPU |
|---|---|---|---|
| 1–2 | Setup, install dependencies | ~3 min | no |
| 3a | Unit tests (44) | ~10 s | no |
| 3b | Synthetic validation | ~30 s | no |
| 4 | HF login, only if needed | — | no |
| 5 | **Audit** — this is the one that matters first | ~20–60 min | no |
| 5b | Inspect raw labels and group counts | instant | no |
| 6 | Build manifest | ~10–30 min | no |
| 7a | Smoke run, 200 utterances both systems | ~10 min | yes |
| 7b | Full Whisper inference | 1–3 h | yes |
| 7c | Full wav2vec 2.0 inference | 30–60 min | yes |
| 8 | Analysis, emits all tables | ~2 min | no |
| 9–10 | Display and download results | instant | no |

**Do not skip 3a and 3b.** They take under a minute and confirm the analysis
chain is correct before you spend hours of GPU time.

**Stop after cell 5b and look at the output.** The group audit tells you how many
utterances and speakers each of the five varieties actually has. If a group is
flagged below the viability threshold, that changes what the paper can claim, and
it is better to know before inference than after.

### Step 4: Handle disconnection

Colab disconnects. Inference checkpoints continuously, so re-running the same
cell resumes from where it stopped rather than restarting. Keep the tab active
during long runs.

### Step 5: Download

Cell 10 zips `outputs/` and downloads it. That archive contains every table the
paper needs, plus the reproducibility record for the appendices.

---

## PART 3 — Claude CLI prompt

Put `edacc-fairness/` in a folder, `cd` into it, run `claude`, and paste the
block below.

```
I'm running an MSc research project measuring accent disparity in automatic
speech recognition. This repository contains the complete pipeline. Read
README.md first, then help me execute the study end to end.

CONTEXT
- Corpus: EdAcc (Edinburgh International Accents of English Corpus), a single
  conversational corpus. Dataset ID is in src/config.py.
- Systems: Whisper Large V3 and wav2vec 2.0, evaluated zero-shot.
- Five accent groups: US (reference), Indian, Scottish, Nigerian, Jamaican.
- This is a MEASUREMENT study, not a fine-tuning study. Do not add fine-tuning.

WHAT I NEED YOU TO DO, IN ORDER

1. Verify the environment
   - Run: python -m pytest tests/ -v
   - Run: python validate_synthetic.py
   - Both must pass before anything else. If either fails, diagnose and fix
     before proceeding. Do not continue past a failure.

2. Confirm corpus access
   - Run: python run.py audit
   - If it fails with a dataset-not-found error, the dataset ID in
     src/config.py is wrong. Search for the correct EdAcc dataset ID and update
     it. If it fails with an auth error, tell me to run huggingface-cli login.

3. Interpret the audit
   - Show me the group audit table: utterances, speakers and viability per group.
   - Show me the top 40 raw accent and first-language strings.
   - Tell me whether any of the five groups are under-counted because the
     matching tokens in STUDY_GROUPS do not cover how speakers actually
     described themselves.
   - If so, propose specific token additions and explain each one. Then re-run
     audit_dictionary() to confirm no new substring collisions were introduced.
     Two collisions already exist and are handled by a longest-token rule:
     "usa" inside "hausa", and "india" inside "west indian". Do not break that.

4. Build the manifest
   - Run: python run.py manifest
   - Report utterances, speakers and minutes per group.

5. Smoke test before the expensive run
   - Run: python run.py all --smoke 200
   - Confirm hypotheses are non-empty and the tables populate sensibly.

6. Full inference
   - Run: python run.py infer --system whisper
   - Run: python run.py infer --system wav2vec2
   - These are long. If interrupted, re-run the same command; it resumes.

7. Analysis
   - Run: python run.py analyse
   - Show me every table it produces.

8. Interpret the results for the paper
   - For each of the four research questions, tell me what the output does and
     does not support:
     RQ1 magnitude of disparity vs the US reference
     RQ2 whether disparity survives length adjustment and speaker clustering
     RQ3 whether the intersectional interaction was estimable
     RQ4 whether the two architectures agree on the ordering
   - Flag specifically: any overlapping confidence intervals (which preclude a
     claim of separation), whether overdispersion triggered the negative
     binomial fallback, and whether Scottish English patterns with the reference
     or with the outer-circle varieties, since that distinguishes a nativeness
     account from a distributional-distance account.

RULES
- Never fabricate, estimate or fill in a result. If a number is not in the
  pipeline output, say so.
- If a group is below the viability threshold, say plainly that it cannot carry
  an inferential claim, rather than reporting its ratio as if it could.
- Report what the data shows even when it contradicts the hypotheses in the
  paper. A disconfirmed hypothesis is a finding.
- Do not change the statistical specification to obtain a preferred result. The
  overdispersion escalation rule is automatic and stated; leave it alone.
```

### Follow-up prompt, once results exist

```
The pipeline has produced outputs/. Using ONLY the actual numbers in those
files, draft the Results section for my IEEE-format paper.

- Follow the structure already in the paper: IV-A composition, IV-B WER by
  group, IV-C adjusted rate ratios, IV-D intersectional, IV-E error composition.
- Format tables in IEEE style with captions above, numbered continuing from
  Table III.
- Report point estimates with confidence intervals throughout. Never report a
  point estimate alone.
- State explicitly where intervals overlap.
- Where a group was excluded from modelling, state the exclusion and the reason.
- Report results only. No interpretation; that belongs in the Discussion.
- Do not round away uncertainty. If an interval is wide, let it look wide.
```

---

## PART 4 — Order of work

```
1. Verify        pytest + validate_synthetic.py          minutes, no GPU
2. Audit         python run.py audit                     ← decides the study
3. Review        adjust STUDY_GROUPS if needed, re-audit
4. Manifest      python run.py manifest
5. Smoke         python run.py all --smoke 200           GPU
6. Inference     whisper, then wav2vec2                  GPU, hours
7. Analyse       python run.py analyse
8. Write         Results → Discussion → Conclusion → Abstract
```

The abstract is written last, from findings. Everything from step 8 onward
depends on step 2, which is why the audit is the only thing that matters today.
