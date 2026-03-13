#!/usr/bin/env python3
"""
Hard Negative Image Collector
Downloads clean knife images from public sources for training.
"""

import csv
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from io import BytesIO

# Output directory
OUTPUT_DIR = Path("datasets/hard_negatives")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Collection tracker
TRACKER_FILE = OUTPUT_DIR / "collection_tracker.csv"


def init_tracker():
    """Initialize collection tracker."""
    if not TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'category', 'source', 'url', 'verified', 'date'])


def add_to_tracker(filename, category, source, url, verified=False):
    """Add entry to tracker."""
    with open(TRACKER_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename, category, source, url, verified, time.strftime('%Y-%m-%d')])


def download_image(url, filename, category, min_size=600):
    """Download and save an image."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ImageCollector/1.0)'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Open and validate
        img = Image.open(BytesIO(response.content))
        
        # Check size
        if min(img.size) < min_size:
            print(f"  ✗ Too small: {img.size}")
            return False
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P', 'LA', 'L'):
            img = img.convert('RGB')
        
        # Save
        output_path = IMAGES_DIR / filename
        img.save(output_path, 'JPEG', quality=95)
        
        # Create empty label file (negative example)
        label_path = LABELS_DIR / f"{Path(filename).stem}.txt"
        label_path.touch()  # Empty file
        
        print(f"  ✓ Saved: {filename} ({img.size[0]}x{img.size[1]})")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return False


def main():
    init_tracker()
    print("=" * 60)
    print("HARD NEGATIVE IMAGE COLLECTOR")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Tracker: {TRACKER_FILE}")
    print("\nTo add images manually:")
    print("  1. Save image to: datasets/hard_negatives/images/")
    print("  2. Create empty file: datasets/hard_negatives/labels/{name}.txt")
    print("  3. Update tracker: {name},category,source,url,verified,date")
    print()
    
    # Show current status
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'r') as f:
            reader = csv.DictReader(f)
            entries = list(reader)
        
        if entries:
            print(f"Current collection: {len(entries)} images")
            categories = {}
            for e in entries:
                cat = e['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            print("\nBy category:")
            for cat, count in sorted(categories.items()):
                print(f"  {cat}: {count}")
        else:
            print("No images collected yet.")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Manual download from knife retailers:")
    print("   - Korin.com (Shun, Masakage)")
    print("   - Chef's Knives To Go")
    print("   - Williams Sonoma")
    print("   - Zwilling.com")
    print("\n2. Save clean images as:")
    print("   neg_satin_001.jpg")
    print("   neg_hammered_001.jpg")
    print("   neg_grind_001.jpg")
    print("   neg_glare_001.jpg")
    print("   neg_weld_001.jpg")
    print("\n3. Create empty label files:")
    print("   neg_satin_001.txt (empty)")
    print("   neg_hammered_001.txt (empty)")
    print("\n4. Update tracker CSV")


if __name__ == "__main__":
    main()
