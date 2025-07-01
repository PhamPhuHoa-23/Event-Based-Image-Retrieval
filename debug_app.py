#!/usr/bin/env python3
import pandas as pd
import os
import json

def debug_track2_file(result_set_name="pipeline_20250618_154004"):
    """Debug track2 file to see why images aren't showing"""
    
    result_path = os.path.join("app_results", result_set_name)
    print(f"Checking result set: {result_path}")
    
    # Check files in directory
    files = os.listdir(result_path)
    print(f"Files found: {files}")
    
    # Find track2 file
    track2_files = [f for f in files if 'track2' in f.lower() and f.endswith('.csv')]
    print(f"Track2 files: {track2_files}")
    
    if track2_files:
        track2_file = track2_files[0]
        track2_path = os.path.join(result_path, track2_file)
        
        # Load and examine track2 data
        df = pd.read_csv(track2_path)
        print(f"\nTrack2 file: {track2_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Check image columns
        image_cols = [col for col in df.columns if 'image_id' in col]
        print(f"\nImage columns: {image_cols}")
        
        # Check first query's images
        if len(df) > 0:
            first_query = df.iloc[0]
            print(f"\nFirst query ID: {first_query.get('query_id', 'N/A')}")
            images = []
            for i in range(1, 11):
                col = f'image_id_{i}'
                if col in first_query and not pd.isna(first_query[col]) and first_query[col] != "#":
                    images.append(first_query[col])
            print(f"Images for first query: {images}")
    
    # Check mapping files
    print(f"\n=== Checking mapping files ===")
    mapping_files = [
        "database_images_to_article_v.0.1.json",
        "database_article_to_url.json", 
        "database_article_to_images_v.0.1.json"
    ]
    
    for file_path in mapping_files:
        if os.path.exists(file_path):
            print(f" {file_path} exists")
            # Check file size
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   Size: {size:.2f} MB")
        else:
            print(f" {file_path} missing")
    
    # Check image directory
    print(f"\n=== Checking image directories ===")
    possible_dirs = [
        "D:/database_compressed_images/database_images_compressed90",
        "D:\\database_compressed_images\\database_images_compressed90",
        "./images",
        "./static/images"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f" {dir_path} exists")
            # Count image files
            try:
                img_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
                print(f"   Contains {len(img_files)} .jpg files")
            except:
                print(f"   Could not list files")
        else:
            print(f" {dir_path} not found")

def test_image_lookup():
    """Test image to article mapping"""
    print(f"\n=== Testing Image Lookup ===")
    
    mapping_file = "database_images_to_article_v.0.1.json"
    if os.path.exists(mapping_file):
        # Load a small sample
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Mapping contains {len(data)} image entries")
        
        # Show first few entries
        sample_keys = list(data.keys())[:5]
        print("Sample mappings:")
        for key in sample_keys:
            print(f"  {key} -> {data[key]}")
    else:
        print("Mapping file not found")

if __name__ == "__main__":
    debug_track2_file()
    test_image_lookup() 