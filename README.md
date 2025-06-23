# podcast_analysis
Analyze podcasts for discourse markers. 

## Quick Start

```bash
# 1. Create / update the conda environment
conda env update -f environment.yml
conda activate podcast_analysis

# 2. Run the end-to-end analysis pipeline (ElevenLabs transcript)
python scripts/main.py --source elevenlabs
```

The command above will:
1. Parse `data/processed/full_podcast/elevenlabs_transcript.json`.
2. Generate `elevenlabs_analysis_results.json` with word-level instances of "right".
3. Produce three plots in `data/processed/full_podcast/visualizations_accurate/`.
4. Build a high-accuracy video montage (and FCPXML) for Dylan Patel's "right" tokens.

---

## ElevenLabs JSON Transcript Format

The file `data/processed/full_podcast/elevenlabs_transcript.json` is a single JSON object:

```json
{
  "words": [ { … 125 869 objects … } ],
  "utterances": []
}
```

Each element of `words` has the schema:

| key          | type   | example           | notes |
|--------------|--------|-------------------|-------|
| `text`       | str    | "Right?"         | literal token including punctuation / spaces |
| `start`      | float  | 1559.319          | seconds (inclusive) |
| `end`        | float  | 1559.559          | seconds (exclusive) |
| `type`       | str    | `word` / `spacing` / `audio_event` |
| `speaker_id` | str    | `speaker_2`       | maps to human names in `transcript_loader.py` |
| `logprob`    | float  | 0.0               | ignored in current analysis |
| `characters` | null   | –                 | reserved |

Important quirks:
1. **Spaces are explicit tokens** – every space in the transcript appears as a record with `type == "spacing"`. These make up 62 927 entries.
2. **Punctuation counts as `word`** – e.g., a period (`text = "."`) has `type == "word"`.
3. **Filler tokens** like "uh" / "um" are included. ElevenLabs is verbatim, so the total "real" words (letters or digits) is **62 552**, while `word` tokens total 62 700.

### How we counted words
```python
import json, re
with open('elevenlabs_transcript.json') as f:
    words = json.load(f)['words']
word_tokens   = [w for w in words if w['type'] == 'word']
real_words    = [w for w in word_tokens if re.search('[A-Za-z0-9]', w['text'])]
print(len(real_words))   # 62 552
```
• Spacing tokens are excluded entirely.
• Punctuation-only tokens are removed by regex.
• The remaining objects are treated as words for frequency stats.

To collect "right" tokens we further cleaned punctuation:
```python
import string, re
translator = str.maketrans('', '', string.punctuation)
rightish = [w for w in word_tokens if 
            w['text'].translate(translator).lower() == 'right']
```
This yields **999** instances after phrase-level exclusions (`all right`, `right now`, etc.).

---

## Repository Layout (high-level)
```
├── data/                  # Raw & processed media / transcripts
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── main.py            # Entry-point orchestrating the pipeline
│   ├── analysis/          # Core analysis modules
│   ├── create_video_*.py  # Video & FCPXML utilities
│   └── diagnostics/       # One-off debugging scripts (suggested)
├── tests/                 # Pytest suite (add soon)
├── environment.yml        # Conda env spec
└── README.md
```

---

## House-Keeping Suggestions
1. **Diagnostics folder** – move ad-hoc scripts (e.g., `compare_word_counts.py`, `deep_analyze_json.py`) into `scripts/diagnostics/`.
2. **Archive legacy code** – keep `scripts/archive/` but mark it "read-only" in the README.
3. **Automated tests** – add word-count and montage-integrity tests in `tests/`.
4. **CI** – GitHub Actions running the smoke tests on every push.
5. **Tags / releases** – create a `v1-right-analysis` tag once results are final.

---

## Contributing / Next Steps
* Commit & push the latest documentation updates:
  ```bash
  git add analysis_summary.md lab_notebook.md README.md
  git commit -m "docs: update counts & add ElevenLabs JSON explanation"
  git push origin feature/elevenlabs-transcription
  ```
* Open an issue to track the refactor of diagnostics and CI setup.
