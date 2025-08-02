# 📂 Few-Shot Episode Definitions (`splits/`)

This folder contains predefined and dynamically generated **few-shot evaluation episodes** for the ISIC 2019 skin lesion classification task using **Prototypical Networks**.

---

## 📄 Files Overview

| File                          | Description |
|-------------------------------|-------------|
| `5way_5shot_episodes.json`    | Predefined 5-way 5-shot few-shot evaluation episodes. |
| `5way_1shot_episodes.json`    | Predefined 5-way 1-shot few-shot evaluation episodes. |
| `generate_fewshot_splits.py`  | Utility script to dynamically generate few-shot episodes. |
| `split_config.yaml`           | YAML config for dynamic episode generation. |

---

## 🔁 Dynamic Episode Generation

Episodes can be automatically generated from your dataset using `generate_fewshot_splits.py` and a YAML config.

### Example:

```bash
python generate_fewshot_splits.py \
  --config splits/split_config.yaml \
  --output_path splits/generated/test_episode_01.json
