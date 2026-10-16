# Manga page reader

Reads three consecutive English manga pages and returns the story text in reading order, with a speaker label on every line. The same character keeps the same label across all three pages.

**Live app:** `https://bytedemourl.streamlit.app/`

**Model used for the submission:** Qwen3-VL-8B with a LoRA adapter fine-tuned on our labelled sequences (run v3 below). The test predictions, the adapter and this app all come from that run. We also trained a larger Qwen3-VL-32B model as an experiment (run v4). It scored higher on the held-out data, but it is not part of the submission.

## What the app does

The app is a viewer for the model's output on the 15 test sequences.

- **3D page stack.** The three pages float in a fan, with page 1 on the right because manga reads right to left. Move the pointer to tilt the stack. Choosing a page brings it to the front.
- **Transcript.** The text for the chosen page appears as speech bubbles. Each character gets one colour for the whole sequence, and bubbles for the same character stay on the same side. Narration appears as a boxed caption.
- **Empty pages.** If the model found no story text on a page, the app says so instead of showing a blank panel.
- **Score table.** The sidebar lists how each model run scored on held-out data.

The app does not run the model. It reads `predictions/test_predictions.jsonl`, which the training notebook produced, so it needs no GPU.

## What is in this repo

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `.streamlit/config.toml` | Dark theme settings |
| `requirements.txt` | `streamlit`, `pillow` |
| `predictions/test_predictions.jsonl` | Model output for the 15 test sequences, in the required format |
| `dataset/sequences.json`, `dataset/test/` | Page lists and test page images, used by the 3D view |
| `notebook.py` | The marimo notebook that trained the model and generated the predictions |
| `outputs/` | Score files, training logs and debug files from the runs |

If the page images are missing, the app still works: the 3D view shows blank pages and the transcript is unaffected.

## Run it yourself

```
pip install -r requirements.txt
streamlit run app.py
```

To deploy: push this repo to GitHub, open share.streamlit.io, sign in with GitHub, choose the repo and the branch, set the main file to `app.py`, and deploy.

## Results

Every run below was scored with the task's `score.py` on the same 10 development sequences, which were kept out of training. "No tuning" means the model ran with our prompt and no training.

| Run | Model | Text recovery (`text_order_score`) | Speaker + text (`balanced_joint_f1`) | Speaker accuracy on matched text | `joint_f1` | Token coverage |
|---|---|---|---|---|---|---|
| v1, no tuning | Qwen2.5-VL-7B | 0.404 | 0.157 | 0.531 | 0.257 | 0.440 |
| v1 | Qwen2.5-VL-7B + LoRA | 0.530 | 0.250 | 0.617 | 0.423 | 0.608 |
| v2, no tuning | Qwen2.5-VL-7B, higher resolution | 0.442 | 0.195 | 0.590 | 0.337 | 0.510 |
| v2 | Qwen2.5-VL-7B + LoRA, higher resolution | 0.562 | 0.250 | 0.583 | 0.476 | 0.631 |
| v3, no tuning | Qwen3-VL-8B | 0.139 | 0.038 | 0.116 | 0.073 | 0.136 |
| **v3, submitted** | **Qwen3-VL-8B + LoRA** | **0.816** | **0.463** | **0.706** | **0.678** | **0.858** |
| v4, no tuning (experiment) | Qwen3-VL-32B | 0.463 | 0.339 | 0.724 | 0.457 | 0.573 |
| v4 (experiment, not submitted) | Qwen3-VL-32B + LoRA | 0.867 | 0.551 | 0.773 | 0.754 | 0.871 |

The two main scores are text recovery and speaker plus text. The other columns are the task's remaining metrics.

Facts to keep in mind when reading the table:

- Ten sequences is a small sample, so the numbers are rough.
- Early stopping and the choice of the best epoch used the same 10 sequences, so the LoRA scores are slightly optimistic.
- The untuned Qwen3-VL-8B produced no text at all on 8 of the 10 sequences, so its row measures the missing output format, not its reading ability. The untuned 32B followed the format, so it is the fairer baseline for the LoRA gain.
- The v1 LoRA used rank 16 and 3 epochs. v2 added a higher image resolution, rank 32 and early stopping. v3 and v4 changed only the model.
- v3, the submitted run, stopped at epoch 7 and restored epoch 4, which had the lowest validation loss (0.131). The v4 experiment reached a best validation loss of 0.108.
- Speaker consistency is the weakest part. For the submitted 8B model, text recovery is about 0.82 but speaker plus text is 0.46. The 32B experiment improved both (0.87 and 0.55) and still shows the same gap.

## Approach and experiments

*Write this part in your own words. The task asks for a README written by you.*

**Approach.** What does the system do, from three images to JSONL? Why one model that sees all three pages at once?

**What I built and what I reused.** Which parts are your own (prompt, split, loss on answer tokens only, parser, early stopping, evaluation) and which are external (Qwen models, PEFT, transformers, marimo, three.js, Streamlit)?

**Experiments.** What did you change between v1 to v4, and what did each change do to the scores?

**What failed.** Pick two or three real examples from `outputs/` (for example sequences where several characters were merged into one label, or a page the model missed).

**What I would try next.** One or two ideas, and why.

## Credits

Qwen2.5-VL and Qwen3-VL (Alibaba Qwen team), Hugging Face transformers and PEFT, marimo and molab for training, three.js for the 3D view, Streamlit for the app.
