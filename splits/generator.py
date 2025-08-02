import os
import random
import json
import yaml
from collections import defaultdict
from glob import glob
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_class_images(data_root):
    class_to_images = defaultdict(list)
    for class_dir in os.listdir(data_root):
        class_path = os.path.join(data_root, class_dir)
        if os.path.isdir(class_path):
            images = glob(os.path.join(class_path, "*.jpg")) + \
                     glob(os.path.join(class_path, "*.png")) + \
                     glob(os.path.join(class_path, "*.jpeg"))
            if images:
                class_to_images[class_dir] = sorted(images)
    return class_to_images

def generate_episode(class_to_images, n_way, k_shot, n_query):
    selected_classes = random.sample(list(class_to_images.keys()), n_way)
    support_set, query_set = [], []

    for cls in selected_classes:
        images = class_to_images[cls]
        if len(images) < k_shot + n_query:
            continue  # Skip this class if not enough images
        sampled = random.sample(images, k_shot + n_query)
        support = sampled[:k_shot]
        query = sampled[k_shot:]
        support_set.extend([(img, cls) for img in support])
        query_set.extend([(img, cls) for img in query])

    return {"support": support_set, "query": query_set}

def generate_all_episodes(config):
    random.seed(config["seed"])
    class_to_images = get_class_images(config["data_root"])

    all_episodes = []
    for _ in tqdm(range(config["num_episodes"]), desc="Generating Episodes"):
        episode = generate_episode(
            class_to_images,
            config["n_way"],
            config["k_shot"],
            config["n_query"]
        )
        all_episodes.append(episode)

    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
    with open(config["output_path"], "w") as f:
        json.dump(all_episodes, f, indent=2)

    print(f"[INFO] Saved {len(all_episodes)} episodes to {config['output_path']}")

if __name__ == "__main__":
    config = load_config("splits/split_config.yaml")
    generate_all_episodes(config)
