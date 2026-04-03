"""GEdit-Bench score aggregation script.

Adapted from model/Step1X-Edit/GEdit-Bench/calculate_statistics.py.
Changes: removed megfile dependency, uses standard open().
"""
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math

GROUPS = [
    "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
]

def analyze_scores(save_path_dir, evaluate_group, language, file_ext):
    group_dict_sub = {}
    group_scores_semantics = defaultdict(lambda: defaultdict(list))
    group_scores_quality = defaultdict(lambda: defaultdict(list))
    group_scores_overall = defaultdict(lambda: defaultdict(list))

    group_scores_semantics_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_quality_intersection = defaultdict(lambda: defaultdict(list))
    group_scores_overall_intersection = defaultdict(lambda: defaultdict(list))
    length_total = 0

    for group_name in GROUPS:
        csv_path = os.path.join(save_path_dir, f"{evaluate_group[0]}_{group_name}_{file_ext}_vie_score.csv")
        df = pd.read_csv(csv_path)

        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        filtered_semantics_scores_intersection = []
        filtered_quality_scores_intersection = []
        filtered_overall_scores_intersection = []

        for _, row in df.iterrows():
            source_image = row['key']
            edited_image = row['edited_image']
            instruction = row['instruction']
            semantics_score = row['sementics_score']
            quality_score = row['quality_score']
            intersection_exist = row['intersection_exist']
            instruction_language = row['instruction_language']

            if instruction_language == language:
                pass
            else:
                continue

            overall_score = math.sqrt(semantics_score * quality_score)

            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)
            if intersection_exist:
                filtered_semantics_scores_intersection.append(semantics_score)
                filtered_quality_scores_intersection.append(quality_score)
                filtered_overall_scores_intersection.append(overall_score)

        avg_semantics_score = np.mean(filtered_semantics_scores)
        avg_quality_score = np.mean(filtered_quality_scores)
        avg_overall_score = np.mean(filtered_overall_scores)
        group_scores_semantics[evaluate_group[0]][group_name] = avg_semantics_score
        group_scores_quality[evaluate_group[0]][group_name] = avg_quality_score
        group_scores_overall[evaluate_group[0]][group_name] = avg_overall_score

        avg_semantics_score_intersection = np.mean(filtered_semantics_scores_intersection)
        avg_quality_score_intersection = np.mean(filtered_quality_scores_intersection)
        avg_overall_score_intersection = np.mean(filtered_overall_scores_intersection)
        group_scores_semantics_intersection[evaluate_group[0]][group_name] = avg_semantics_score_intersection
        group_scores_quality_intersection[evaluate_group[0]][group_name] = avg_quality_score_intersection
        group_scores_overall_intersection[evaluate_group[0]][group_name] = avg_overall_score_intersection


    # --- Overall Model Averages ---

    # Semantics
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics[model_name]["avg_semantics"] = model_avg

    # Semantics Intersection
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics_intersection[model_name]["avg_semantics"] = model_avg

    # Quality
    for model_name in evaluate_group:
        model_scores = [group_scores_quality[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality[model_name]["avg_quality"] = model_avg

    # Quality Intersection
    for model_name in evaluate_group:
        model_scores = [group_scores_quality_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality_intersection[model_name]["avg_quality"] = model_avg

    # Overall
    for model_name in evaluate_group:
        model_scores = [group_scores_overall[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall[model_name]["avg_overall"] = model_avg

    # Overall Intersection
    for model_name in evaluate_group:
        model_scores = [group_scores_overall_intersection[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall_intersection[model_name]["avg_overall"] = model_avg

    return group_scores_semantics, group_scores_quality, group_scores_overall, group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="step1x")
    parser.add_argument("--backbone", type=str, default="qwen25vl", choices=["qwen25vl"])
    parser.add_argument("--save_path", type=str, default="/results/")
    parser.add_argument("--language", type=str, default="all", choices=["all", "en", "cn"])
    args = parser.parse_args()
    model_name = args.model_name
    backbone = args.backbone
    save_path_dir = args.save_path
    if args.language == "all":
        languages = ["en", "cn"]
    else:
        languages = [args.language]

    for language in languages:
        print("="*10 + f" backbone:{backbone} - model_name:{model_name} - language:{language} " + "="*10)

        save_path_new = os.path.join(save_path_dir, model_name, backbone)
        group_scores_semantics, group_scores_quality, group_scores_overall, group_scores_semantics_intersection, group_scores_quality_intersection, group_scores_overall_intersection = analyze_scores(save_path_new, [model_name], language=language, file_ext=args.language)

        print("\nOverall:")
        for group_name in GROUPS:
            print(f"{group_name}: {group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")
        print(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}, {group_scores_quality[model_name]['avg_quality']:.3f}, {group_scores_overall[model_name]['avg_overall']:.3f}")

        print("\nIntersection:")
        for group_name in GROUPS:
            print(f"{group_name}: {group_scores_semantics_intersection[model_name][group_name]:.3f}, {group_scores_quality_intersection[model_name][group_name]:.3f}, {group_scores_overall_intersection[model_name][group_name]:.3f}")
        print(f"Average Intersection: {group_scores_semantics_intersection[model_name]['avg_semantics']:.3f}, {group_scores_quality_intersection[model_name]['avg_quality']:.3f}, {group_scores_overall_intersection[model_name]['avg_overall']:.3f}")
