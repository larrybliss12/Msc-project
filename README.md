# EdAcc Accent Fairness Measurement Study

Measurement audit of accent-related word error rate disparity in automatic
speech recognition, using the Edinburgh International Accents of English
Corpus (EdAcc) as a single conversational corpus.

Two systems are evaluated in inference mode with no adaptation: Whisper Large
V3 and wav2vec 2.0. Five English varieties are compared against a United States
English reference: Indian, Scottish, Nigerian and Jamaican English.

---

## Why a single corpus

Conversational speech yields substantially higher error rates than read speech
for all speakers. Where accent groups are drawn from corpora differing in
register, the register effect is confounded with the accent effect and no
cross-accent claim is identifiable. Holding corpus, register, channel and
recording protocol constant makes residual between-group variation attributable
to variety rather than provenance.

---

## Repository layout

```
edacc-fairness/
├── run.py                     staged CLI entry point
├── validate_synthetic.py      end-to-end validation without corpus access
├── requirements.txt
├── src/
│   ├── config.py              all study parameters and the accent dictionary
│   ├── mapping.py             label normalisation and group assignment
│   ├── data.py                loading, schema probing, audit, manifest
│   ├── inference.py           ASR transcription with resumption
│   ├── scoring.py             normalisation, WER, speaker-level bootstrap
│   ├── stats.py               Poisson GLM, clustered SE, overdispersion
│   └── report.py              emits the paper's numbered tables
├── tests/
│   └── test_pipeline.py       44 tests, the verification evidence
└── notebooks/
    └── EdAcc_Study_Colab.ipynb
```

---

## Running in VSCode

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verification first. No corpus or GPU needed.
python -m pytest tests/ -v
python validate_synthetic.py

# Then the study, one stage at a time.
python run.py audit                      # schema + accent enumeration
python run.py manifest                   # utterance manifest
python run.py infer --system whisper     # GPU strongly recommended
python run.py infer --system wav2vec2
python run.py analyse                    # scores, models, writes all tables
```

Fast end-to-end check on a small subset:

```bash
python run.py all --smoke 200
```

Every stage writes to `outputs/` and can be re-run independently. Inference
resumes from partial output, so an interrupted transcription pass does not
restart.

---

## Running in Colab

Open `notebooks/EdAcc_Study_Colab.ipynb` and set the runtime to GPU
(Runtime → Change runtime type → T4 GPU or better). The notebook clones or
uploads this repository and calls the same modules, so Colab and VSCode execute
identical code rather than divergent copies.

Whisper Large V3 over the full corpus takes roughly one to three hours on a T4.
Run the smoke stage first.

---

## Corpus access

EdAcc is distributed under CC-BY-SA. If loading fails with an authentication
error:

```python
from huggingface_hub import login
login()
```

and accept the licence terms on the dataset page.

---

## Design commitments enforced in code

**One normaliser for everything.** A single normaliser instance is applied to
every reference and every hypothesis of every system. Scoring two systems under
different normalisation regimes would make any comparison an artefact of text
processing rather than a measurement of recognition quality.

**Bootstrap resamples speakers, not utterances.** Utterances within a speaker
are correlated. Utterance-level resampling produces intervals that are too
narrow and overstates precision. `tests/test_pipeline.py` contains a test that
fails if this is ever weakened.

**Token-weighted aggregation.** Group WER is total errors over total reference
tokens, not the mean of per-utterance rates, which over-weights short utterances
where a single error produces an extreme rate.

**Log reference length as an offset, not a covariate.** Its coefficient is fixed
at 1, constraining the model to describe errors per reference token, which is
exactly a word error rate.

**Fail loudly.** Every stage asserts non-empty output and raises
`EmptyStageError` otherwise. The predecessor pipeline completed successfully
while producing a 22-byte artefact; these guards exist to prevent that class of
silent failure.

**Specificity-based label matching.** Study tokens overlap as substrings.
`audit_dictionary()` detected two collisions before any experiment was run:
`"usa"` inside `"hausa"`, which would have assigned Nigerian Hausa speakers to
the US reference group and contaminated the baseline; and `"india"` inside
`"west indian"`, which would have assigned Caribbean speakers to the Indian
group. Both are resolved by the longest-matching-token rule and both are covered
by regression tests.

---

## Verification status

```
44 tests passing
validate_synthetic.py: recovers injected disparity ratios within tolerance
```

The synthetic validation injects known disparities (1.4, 2.0, 2.4, 3.0) and
confirms the analysis chain recovers them and their ordering. This validates
scoring, aggregation and modelling independently of the corpus.

---

## Outputs

| File | Contents |
|---|---|
| `schema.json` | Discovered EdAcc columns and resolved role mapping |
| `raw_accent_labels.csv` | Every raw accent and L1 string with frequency (paper appendix) |
| `group_audit.csv` | Utterances, speakers and viability per group |
| `manifest.csv` | One row per mapped utterance |
| `hyp_whisper.csv`, `hyp_wav2vec2.csv` | Per-system hypotheses |
| `scored_utterances.csv` | Per-utterance error breakdown |
| `table_v_1.csv` … `table_v_5.csv` | The paper's numbered tables |
| `paper_tables.md` | All tables in markdown, ready to paste |
| `run_config.json`, `normaliser.json` | Reproducibility record |
