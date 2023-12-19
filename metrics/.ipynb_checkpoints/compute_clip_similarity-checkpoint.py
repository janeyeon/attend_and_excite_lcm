import json
import sys
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import pdb
import clip
import numpy as np
import pandas as pd
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")
    dataset: str = './datasets/phrases1.csv'
    condition: str = 'SynGen_LCM'

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")
    
    csv = pd.read_csv(config.dataset)
    prompts = csv.prompt
    num_phrases = int(config.dataset[config.dataset.index('.csv')-1])
    # prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")
    
    full_similarities = []
    partial_similarities = []
    results_per_prompt = {}
    for i, prompt in enumerate(tqdm(prompts)):

        print(f'Running on: "{prompt}"')

        # get all images for the given prompt
        image_paths = [p for p in (config.output_path).rglob(f'./phrases{num_phrases}/{i:003}/{config.condition}*') if p.suffix in ['.png', '.jpg']]
        images = [Image.open(p) for p in image_paths]
        image_names = [p.name for p in image_paths]
        queries = [preprocess(image).unsqueeze(0).to(device) for image in images]

        with torch.no_grad():

            # split prompt into first and second halves
            if ' and ' in prompt:
                prompt_parts = prompt.split(' and ')
            elif ' with ' in prompt:
                prompt_parts = prompt.split(' with ')
            else:
                print(f"Unable to split prompt: {prompt}. "
                      f"Looking for 'and' or 'with' for splitting! Skipping!")
                prompt_parts = [prompt]

            # extract texture features
            full_text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)
            partial_features = []
            if num_phrases > 1:
                for i in range(num_phrases):
                    partial_features.append(get_embedding_for_prompt(model, prompt_parts[i], templates=imagenet_templates))

            # extract image features
            images_features = [model.encode_image(image) for image in queries]
            images_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in images_features]

            # compute similarities
            full_text_similarities = [(feat.float() @ full_text_features.T).item() for feat in images_features]
            partial_similarities = []
            if num_phrases > 1:
                for partial_feature in partial_features:
                    partial_similarities.append([(feat.float() @ partial_feature.T).item() for feat in images_features])
            
            results_per_prompt[prompt] = {
                'full_text': full_text_similarities,
                'image_names': image_names,
            }
            # full_similarities += full_text_similarities
            
            if num_phrases > 1:
                for i, partial_similarity in enumerate(partial_similarities):
                    results_per_prompt[prompt][f"text_part{i}"] = partial_similarity
                    # partial_similarities += partial_similarity

    # results_per_prompt['mean_score'] = np.mean(full_similarities)
    with open(config.metrics_save_path / f"{config.condition}_{num_phrases}_clip_raw_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
        
    # aggregate results
    aggregated_results = {'full_text_aggregation': aggregate_by_full_text(results_per_prompt)}
    if num_phrases > 1:
        # aggregated_results[]
        aggregated_results['min_partial_aggregation'] = aggregate_by_min_partial(deepcopy(results_per_prompt))
        # 'min_first_second_aggregation': aggregate_by_min_half(results_per_prompt),
    with open(config.metrics_save_path / f"{config.condition}_{num_phrases}_clip_aggregated_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_by_min_partial(similarities):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_total = []
    keys = [*filter(lambda x: 'part' in x, similarities[[*similarities.keys()][0]].keys())]

    for prompt in similarities:
        min_total.append(min([similarities[prompt][key] for key in keys]))
    min_total = np.concatenate(min_total)
    min_per_half_res = np.array(min_total).flatten()
    return np.average(min_per_half_res)

def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_per_half_res = [[min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)


def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.concatenate(full_text_res)
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


if __name__ == '__main__':
    run()