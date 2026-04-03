"""GEdit-Bench scoring script.

Adapted from model/Step1X-Edit/GEdit-Bench/run_gedit_score.py.
Changes: removed megfile dependency, parameterized model/dataset paths.
"""
from viescore import VIEScore
import PIL
import os
import glob
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sys
import csv
import time
import argparse
import math
from collections import defaultdict

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)

def process_single_item(item, vie_score, max_retries=10000):

    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    intersection_exist = item['Intersection_exist']
    edit_image_path = item['edited_image_path']

    for retry in range(max_retries):
        try:
            pil_image = item['input_image_raw'].convert("RGB")
            pil_image_edited = Image.open(edit_image_path).convert("RGB")
            source_img_width, source_img_height, source_img_area = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
            edited_img_width, edited_img_height, edited_img_area = calculate_dimensions(512 * 512, pil_image_edited.width / pil_image_edited.height)
            pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
            pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))
            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image, pil_image_edited], text_prompt)
            sementics_score, quality_score, overall_score = score_list

            print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")

            return {
                "key": key,
                "edited_image": edit_image_path,
                "instruction": instruction,
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "intersection_exist" : item['Intersection_exist'],
                "instruction_language" : item['instruction_language']
            }
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {edit_image_path} after {max_retries} attempts: {e}")
                return

def find_files_with_given_basename(folder_path, basename):
    pattern = os.path.join(folder_path, f"{basename}.*")
    matched_files = glob.glob(pattern)
    return [os.path.basename(f) for f in matched_files]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="step1x", help="edit model name")
    parser.add_argument("--edited_images_dir", type=str, default="edit_images", help="path to edited images")
    parser.add_argument("--instruction_language", type=str, default="all", choices=["all", "en", "cn"])
    parser.add_argument("--task_type", type=str, default="all",  choices=["all", "background_change", "color_alter", "material_alter", "motion_change",
    "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"])
    parser.add_argument("--save_dir", type=str, default="csv_results")
    parser.add_argument("--backbone", type=str, default="qwen25vl", choices=["qwen25vl"])
    parser.add_argument("--model_path", type=str, default=None, help="path to evaluator model weights")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to pre-downloaded GEdit-Bench dataset (load_from_disk)")
    args = parser.parse_args()
    model_name = args.model_name
    edited_images_dir = args.edited_images_dir
    instruction_language = args.instruction_language
    save_dir = args.save_dir
    backbone = args.backbone
    if args.task_type == "all":
        groups = ["background_change", "color_alter", "material_alter", "motion_change", "ps_human",
        "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"]
    else:
        groups = [args.task_type]

    # Load GEdit-Bench dataset and group by task type
    vie_score = VIEScore(backbone=backbone, task="tie", model_path=args.model_path)
    if args.dataset_path and os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset("stepfun-ai/GEdit-Bench")
    dataset_by_group = defaultdict(list)
    # Handle both Dataset and DatasetDict
    data_split = dataset['train'] if 'train' in dataset else dataset
    for i, item in tqdm(enumerate(data_split), desc=f"Loading GEdit-Bench dataset..."):
        if instruction_language == "all" or item['instruction_language'] == instruction_language:
            dataset_by_group[item['task_type']].append(item)
    for k, v in dataset_by_group.items():
        print(f"Number of samples in {k} - {instruction_language}:", len(v))

    # Evaluate each group
    save_path_new = os.path.join(save_dir, model_name, backbone)
    for group_name in groups:
        group_csv_list = []
        group_dataset_list = dataset_by_group[group_name]

        # Load existing group CSV if it exists, if csv exists, skip this group
        group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_{instruction_language}_vie_score.csv")
        if os.path.exists(group_csv_path):
            with open(group_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                group_results = list(reader)
            print(f"{model_name} - {group_name} exists, skip this group")
            continue

        for item in tqdm(group_dataset_list, desc=f"Processing {model_name} - {group_name}"):
            key = item['key']
            try:
                # Should organize edited image directory, please refer EVAL.md for details
                edited_images_path = os.path.join(edited_images_dir, model_name, 'fullset', group_name, item['instruction_language'])
                item['edited_image_path'] = os.path.join(edited_images_path, find_files_with_given_basename(edited_images_path, key)[0])
            except:
                print(key, "not found in", edited_images_path)
                continue

            result = process_single_item(item, vie_score)
            if result:
                group_csv_list.append(result)

        # Save group-specific CSV
        os.makedirs(os.path.dirname(group_csv_path), exist_ok=True)
        with open(group_csv_path, 'w', newline='') as f:
            fieldnames = ["key", "edited_image", "instruction", "sementics_score", "quality_score", "intersection_exist", "instruction_language"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in group_csv_list:
                writer.writerow(row)
        print(f"Saved group CSV for {group_name}, length: {len(group_csv_list)}")
