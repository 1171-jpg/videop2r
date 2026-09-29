# VideoP2R: Video Understanding from Perception to Reasoning

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-1f6feb.svg)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jiang_VIDEOP2R_Video_Understanding_from_Perception_to_Reasoning_CVPRF_2026_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2511.11113-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.11113)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://videop2r.github.io/videop2r/)
[![GitHub](https://img.shields.io/badge/GitHub-Data%20%26%20Code-181717.svg?logo=github)](https://github.com/1171-jpg/videop2r)

Official data release for VideoP2R, CVPR 2026 Findings.

VideoP2R treats perception and reasoning as two separate processes in video understanding. We build a process-aware chain-of-thought dataset that separates what the model observes from how it reasons, then train with SFT followed by RL.

## Dataset

**VideoP2R-CoT-162K** contains 162K chain-of-thought annotations in which the perception segment is explicitly separated from the reasoning segment. It is filtered down from 260K initial generations by an observation sufficiency check.

**VideoP2R-RL-263K** contains the 263K samples used in the RL stage. It has the same fields as the SFT file.

This release is text only. No video, audio, images, or frames are redistributed. To use the dataset, download the source media from the original releases listed under Attribution below.

The videos and images can be downloaded from [Video-R1-data](https://huggingface.co/datasets/Video-R1/Video-R1-data). The `path` field of each record follows its directory layout.

### Format

Each record is a JSON object:

| Field | Source | Description |
|---|---|---|
| `problem` | upstream | Question text |
| `options` | upstream | Answer choices, empty for non multiple-choice |
| `solution`, `answer` | upstream | Ground-truth answer |
| `path` | upstream | Relative path identifying the source sample. A string, not a file. |
| `data_source` | upstream | Upstream subset label |
| `data_type` | ours | `image` or `video` |
| `problem_type` | ours | `multiple choice`, `numerical`, `OCR`, `free-form`, or `regression` |
| `observation` | ours | Perception segment of the chain-of-thought |
| `process` | ours | Reasoning segment of the chain-of-thought |
| `output` | ours | Full chain-of-thought trace |
| `problem_id`, `reward`, `select`, `clean_version` | ours | Pipeline metadata |

In the RL file all values are stored as strings, including `options`, `reward` and `select`.

The chain-of-thought traces were generated with Qwen2.5-VL-72B-Instruct (Apache-2.0).

## Download

The files are in [`data/`](data). Each file is a gzip-compressed JSON list. The parts of one dataset are split in order, so concatenating them gives the full dataset.

| Dataset | Files | Records |
|---|---|---|
| VideoP2R-CoT-162K (SFT) | `VideoP2R-162k_sft.part1of2.json.gz`, `VideoP2R-162k_sft.part2of2.json.gz` | 162,062 |
| VideoP2R-RL-263K (RL) | `VideoP2R-263k_rl.part1of3.json.gz`, `VideoP2R-263k_rl.part2of3.json.gz`, `VideoP2R-263k_rl.part3of3.json.gz` | 263,055 |

```python
import glob, gzip, json

def load(prefix):
    records = []
    for path in sorted(glob.glob(f"data/{prefix}.part*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            records += json.load(f)
    return records

sft = load("VideoP2R-162k_sft")   # 162,062 records
rl = load("VideoP2R-263k_rl")     # 263,055 records
```

## Code

The training and inference code is not included in this repository yet. It will be released separately.

## License

The annotation files in this repository are released under the Apache License 2.0. See [LICENSE](LICENSE).

## Attribution

The question text, answer options, ground-truth answers, and source identifiers come from Video-R1-data. The chain-of-thought traces are ours.

| Source | License |
|---|---|
| Video-R1-data | Apache-2.0 |
| CLEVRER | CC0 |
| LLaVA-Video-178K | Apache-2.0 |
| NExT-QA | MIT |
| Perception Test | Apache-2.0 (code), CC-BY-4.0 (annotations) |
| STAR | Apache-2.0 |

Please cite the original datasets in addition to our paper.

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [R1-V](https://github.com/Deep-Agent/R1-V), [Video-R1](https://github.com/tulerfeng/Video-R1)

## Citation

```bibtex
@inproceedings{jiang2026videop2r,
  title={Videop2r: Video understanding from perception to reasoning},
  author={Jiang, Yifan and Wang, Yueying and Zhao, Rui and Parag, Toufiq and Chen, Zhimin and Liao, Zhenyu and Unnikrishnan, Jayakrishnan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8303--8313},
  year={2026}
}
```
