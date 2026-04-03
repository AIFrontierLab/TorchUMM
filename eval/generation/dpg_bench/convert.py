import os

def rename_grid_images(folder):
    for filename in os.listdir(folder):
        if filename.endswith("_grid.png"):
            old_path = os.path.join(folder, filename)
            new_filename = filename.replace("_grid.png", ".png")
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder = "/home/yinyil/projects/OmniGen2/eval/ELLA/dpg_bench/generated_baseline"  
    rename_grid_images(folder)
