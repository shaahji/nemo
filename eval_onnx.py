#!/usr/bin/env python3
"""FLEURS WER eval for a Nemotron ONNX (onnxruntime-genai) model.

Streaming inference, per-utterance TSV + JSON summary, BasicMultilingualText
normalization (same normalizer as the NeMo streaming eval).

This is the per-language (monolingual) evaluator used for the INT4
weight-only quantization study: each ONNX export already targets a single
language and a single streaming-latency tier (the chunk size is baked into
``genai_config.json``), so no language-id / prompt override is applied.

Usage:
    python eval_fleurs_onnx.py \
        --onnx_model <ONNX_ROOT>/es_500h_new_ml_1e4_int4 \
        --lang es_419 \
        --name es_500h_new_ml_1e4_int4 \
        --output_dir ./results

Requires onnxruntime-genai and a model dir containing genai_config.json
(plus encoder.onnx / decoder.onnx / joint.onnx / tokenizer.* / etc.).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import onnxruntime_genai as og


# Multilingual exports may emit "<es-ES>" / "<de-DE>" tokens; strip from text.
# (Harmless no-op for the monolingual exports evaluated here.)
_LANG_TAG_RE = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>\s*")

# WER normalization uses the Whisper BasicMultilingualTextNormalizer
# packaged in the Open ASR Leaderboard repo, used unchanged:
#   https://github.com/huggingface/open_asr_leaderboard
# Clone it and add the repo root to PYTHONPATH (or set OPEN_ASR_LB_ROOT).
_oalb = os.environ.get("OPEN_ASR_LB_ROOT")
if _oalb:
    sys.path.insert(0, _oalb)
try:
    from normalizer import BasicMultilingualTextNormalizer  # noqa: E402
except ImportError as e:
    raise RuntimeError(
        "Could not import normalizer.BasicMultilingualTextNormalizer. "
        "Clone https://github.com/huggingface/open_asr_leaderboard and set "
        "OPEN_ASR_LB_ROOT=/path/to/open_asr_leaderboard."
    ) from e

_norm = BasicMultilingualTextNormalizer()


def normalize_text(t: str) -> str:
    return _norm(t)


def compute_wer_ids(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


def load_fleurs(lang: str):
    from datasets import load_dataset, Audio
    cache = (os.environ.get("HF_DATASETS_CACHE")
             or os.environ.get("HF_HOME"))
    print(f"  Loading FLEURS {lang} test (cache_dir={cache})")
    ds = load_dataset("google/fleurs", lang, split="test",
                      trust_remote_code=True, cache_dir=cache)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def transcribe(model, params, tokenizer, sample_rate, chunk_samples, audio):
    processor = og.StreamingProcessor(model)
    processor.set_option("use_vad", "false")
    gen = og.Generator(model, params)
    tok_stream = tokenizer.create_stream()

    text = ""

    def _drain():
        nonlocal text
        while not gen.is_done():
            gen.generate_next_token()
            toks = gen.get_next_tokens()
            if len(toks) > 0:
                t = tok_stream.decode(toks[0])
                if t:
                    t = _LANG_TAG_RE.sub("", t)
                    if t:
                        text += t

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples].astype(np.float32)
        inp = processor.process(chunk)
        if inp is not None:
            gen.set_inputs(inp)
            _drain()

    inp = processor.flush()
    if inp is not None:
        gen.set_inputs(inp)
        _drain()

    del gen, processor
    return text


def evaluate(onnx_dir: str, dataset, name: str, lang: str, max_samples=None):
    with open(os.path.join(onnx_dir, "genai_config.json")) as f:
        cfg = json.load(f)
    sample_rate = cfg["model"]["sample_rate"]
    chunk_samples = cfg["model"]["chunk_samples"]

    print(f"  Model: {onnx_dir}")
    print(f"  chunk_samples: {chunk_samples}")
    model = og.Model(og.Config(onnx_dir))
    tokenizer = og.Tokenizer(model)
    params = og.GeneratorParams(model)

    total_subs = total_dels = total_ins = total_words = 0
    total_audio_sec = total_proc_sec = 0.0
    errors = 0
    utts = []

    n_total = len(dataset)
    if max_samples:
        n_total = min(n_total, max_samples)
        dataset = dataset.select(range(n_total))
    print(f"  Samples: {n_total}")

    for i, sample in enumerate(dataset):
        try:
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != sample_rate:
                new_len = int(len(audio) * sample_rate / sr)
                audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                                  np.arange(len(audio)), audio).astype(np.float32)

            t0 = time.perf_counter()
            hyp = transcribe(model, params, tokenizer, sample_rate, chunk_samples, audio)
            proc = time.perf_counter() - t0
            dur = len(audio) / sample_rate
            total_audio_sec += dur
            total_proc_sec += proc

            ref_n = normalize_text(sample["transcription"])
            pred_n = normalize_text(_LANG_TAG_RE.sub(" ", hyp).strip())
            ref_w = ref_n.split()
            pred_w = pred_n.split()

            sid = sample.get("id", str(i))
            utts.append({"idx": i, "sample_id": str(sid),
                         "duration": f"{dur:.2f}", "ref": ref_n, "hyp": pred_n})

            if ref_w:
                s, d, ins_ = compute_wer_ids(ref_w, pred_w)
                total_subs += s; total_dels += d; total_ins += ins_
                total_words += len(ref_w)

            if (i + 1) % 25 == 0:
                running = (total_subs + total_dels + total_ins) / max(total_words, 1) * 100
                rtf = total_audio_sec / max(total_proc_sec, 1e-9)
                print(f"    [{i+1}/{n_total}] WER: {running:.2f}%  RTF: {rtf:.2f}x",
                      flush=True)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    [error] {type(e).__name__}: {e}")

    del model
    total_edits = total_subs + total_dels + total_ins
    wer = total_edits / max(total_words, 1) * 100
    return {
        "name": name, "dataset": "fleurs", "lang": lang,
        "wer": wer,
        "subs": total_subs, "dels": total_dels, "ins": total_ins,
        "sub_rate": total_subs / max(total_words, 1) * 100,
        "del_rate": total_dels / max(total_words, 1) * 100,
        "ins_rate": total_ins / max(total_words, 1) * 100,
        "total_edits": total_edits, "total_words": total_words,
        "errors": errors,
        "audio_sec": total_audio_sec, "proc_sec": total_proc_sec,
        "rtf": total_audio_sec / max(total_proc_sec, 1e-9),
        "utterances": utts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_model", required=True,
                   help="Path to ONNX model dir (must contain genai_config.json).")
    p.add_argument("--lang", required=True,
                   help="FLEURS lang code, e.g. es_419, de_de, fr_fr.")
    p.add_argument("--name", default="onnx_model",
                   help="Display/file name for outputs.")
    p.add_argument("--output_dir", default=None,
                   help="Write per-utterance TSV + JSON summary here.")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    print(f"\n  === {args.name} (FLEURS {args.lang}) — ONNX streaming ===")
    ds = load_fleurs(args.lang)
    r = evaluate(args.onnx_model, ds, name=args.name, lang=args.lang,
                 max_samples=args.max_samples)

    print(f"\n  WER: {r['wer']:.2f}%  "
          f"(S={r['sub_rate']:.2f}% D={r['del_rate']:.2f}% I={r['ins_rate']:.2f}%)")
    print(f"  Words: {r['total_words']}  Errors: {r['errors']}  "
          f"Audio: {r['audio_sec']:.1f}s  Proc: {r['proc_sec']:.1f}s  "
          f"RTF: {r['rtf']:.2f}x")

    if args.output_dir and r["utterances"]:
        os.makedirs(args.output_dir, exist_ok=True)
        tsv = os.path.join(args.output_dir, f"{args.name}_fleurs_streaming.tsv")
        with open(tsv, "w") as f:
            f.write("idx\tsample_id\tduration_sec\tref\thyp\n")
            for u in r["utterances"]:
                f.write(f"{u['idx']}\t{u['sample_id']}\t{u['duration']}\t"
                        f"{u['ref']}\t{u['hyp']}\n")
        print(f"  Saved {len(r['utterances'])} utterances -> {tsv}")
        summary = {k: v for k, v in r.items() if k != "utterances"}
        with open(os.path.join(args.output_dir,
                               f"{args.name}_fleurs_streaming.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()