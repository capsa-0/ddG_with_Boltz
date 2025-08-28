#!/usr/bin/env python3
import os
import shutil

def remove_processed_dir():
    processed_dir = "data/processed"
    if os.path.isdir(processed_dir):
        shutil.rmtree(processed_dir)
        print(f"Removed directory: {processed_dir}")
    else:
        print(f"No directory found at: {processed_dir}")

def remove_tmp_files_and_dirs():
    cwd = os.getcwd()
    for item in os.listdir(cwd):
        if "tmp" in item:
            path = os.path.join(cwd, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                elif os.path.isfile(path):
                    os.remove(path)
                    print(f"Removed file: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")


    
if __name__ == "__main__":
    remove_processed_dir()
    remove_tmp_files_and_dirs()
