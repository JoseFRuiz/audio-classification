#!/usr/bin/env python3
"""
Download and prepare BirdCLEF dataset using Kaggle API.
BirdCLEF is an annual challenge for bird species identification.
"""

import os
import subprocess
import zipfile
from pathlib import Path
import argparse
import pandas as pd
import json

def run_kaggle_command(cmd):
    """Run a kaggle command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle command failed: {e}")
        print(f"   Error output: {e.stderr}")
        return None

def check_kaggle_installed():
    """Check if kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Kaggle CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Kaggle CLI not found or not working")
            return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed")
        return False

def download_birdclef_kaggle(year, output_dir):
    """
    Download BirdCLEF dataset using Kaggle API.
    
    Args:
        year: BirdCLEF year (2020, 2021, 2022, 2023)
        output_dir: Output directory for dataset
    """
    print(f"🔹 Downloading BirdCLEF-{year} from Kaggle...")
    
    # Kaggle competition name
    competition_name = f"birdclef-{year}"
    
    # Check if kaggle is installed
    if not check_kaggle_installed():
        print("\n📥 To install Kaggle CLI:")
        print("   pip install kaggle")
        print("\n🔑 To configure Kaggle API:")
        print("   1. Go to https://www.kaggle.com/settings/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Download kaggle.json")
        print("   4. Place it in ~/.kaggle/kaggle.json")
        print("   5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the competition files
    print(f"🔹 Downloading competition files...")
    cmd = f"kaggle competitions download -c {competition_name} -p {output_dir}"
    result = run_kaggle_command(cmd)
    
    if result is None:
        print(f"❌ Failed to download BirdCLEF-{year}")
        return False
    
    print(f"✅ Downloaded BirdCLEF-{year} files to {output_dir}")
    
    # List downloaded files
    downloaded_files = list(Path(output_dir).glob("*.zip"))
    print(f"🔹 Downloaded files:")
    for file in downloaded_files:
        print(f"   - {file.name}")
    
    # Extract zip files
    print(f"🔹 Extracting files...")
    for zip_file in downloaded_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"✅ Extracted {zip_file.name}")
            
            # Remove zip file after extraction
            zip_file.unlink()
            print(f"🗑️ Removed {zip_file.name}")
        except Exception as e:
            print(f"❌ Error extracting {zip_file.name}: {e}")
    
    return True

def create_birdclef_csv(audio_dir, output_csv, metadata_file=None):
    """
    Create CSV file for BirdCLEF dataset.
    
    Args:
        audio_dir: Directory containing BirdCLEF audio files
        output_csv: Output CSV file path
        metadata_file: Optional metadata file with species information
    """
    print(f"🔹 Preparing BirdCLEF dataset from: {audio_dir}")
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(audio_dir).rglob(ext))
    
    print(f"🔹 Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"❌ No audio files found in {audio_dir}")
        return None
    
    # Load metadata if provided
    species_info = {}
    if metadata_file and os.path.exists(metadata_file):
        print(f"🔹 Loading metadata from: {metadata_file}")
        try:
            if metadata_file.endswith('.csv'):
                metadata_df = pd.read_csv(metadata_file)
            elif metadata_file.endswith('.json'):
                with open(metadata_file, 'r') as f:
                    metadata_df = pd.DataFrame(json.load(f))
            
            # Extract species information
            for _, row in metadata_df.iterrows():
                filename = row.get('filename', row.get('id', ''))
                species = row.get('species', row.get('primary_label', ''))
                if filename and species:
                    species_info[filename] = species
        except Exception as e:
            print(f"⚠️ Warning: Could not load metadata: {e}")
    
    # Extract species from filenames if no metadata
    if not species_info:
        print("🔹 Extracting species from filenames...")
        for audio_file in audio_files:
            filename = audio_file.stem
            # Common BirdCLEF filename patterns:
            # species_name_rest, species-name_rest, etc.
            parts = filename.replace('_', ' ').replace('-', ' ').split()
            if len(parts) >= 2:
                species = ' '.join(parts[:2])  # Assume first two words are species
                species_info[filename] = species
    
    # Get unique species
    unique_species = sorted(list(set(species_info.values())))
    print(f"🔹 Found {len(unique_species)} unique species")
    
    # Create DataFrame
    data = []
    for audio_file in audio_files:
        clip_id = audio_file.stem
        species = species_info.get(clip_id, 'Unknown')
        
        # Create one-hot encoded labels
        labels = [1 if species == s else 0 for s in unique_species]
        
        data.append({
            'clip_id': clip_id,
            'species': species,
            **{species: label for species, label in zip(unique_species, labels)}
        })
    
    df = pd.DataFrame(data)
    
    # Filter out unknown species
    df_filtered = df[df['species'] != 'Unknown']
    
    if len(df_filtered) < len(df):
        print(f"⚠️ Warning: {len(df) - len(df_filtered)} files had unknown species and were filtered out")
    
    if len(df_filtered) == 0:
        print(f"❌ Error: No valid files with species labels found")
        return None
    
    # Remove the species column for the final CSV (keep only clip_id and one-hot labels)
    df_final = df_filtered.drop('species', axis=1)
    df_final.to_csv(output_csv, index=False)
    
    print(f"✅ Created CSV with {len(df_final)} entries: {output_csv}")
    
    # Print species distribution
    species_counts = df_filtered['species'].value_counts()
    print(f"🔹 Top 10 species by count:")
    for species, count in species_counts.head(10).items():
        print(f"   {species}: {count}")
    
    return df_final

def main():
    parser = argparse.ArgumentParser(description="Download BirdCLEF dataset using Kaggle")
    parser.add_argument("--output_dir", type=str, default="./birdclef_data", 
                       help="Output directory for dataset")
    parser.add_argument("--year", type=str, default="2023", 
                       help="BirdCLEF challenge year (2020, 2021, 2022, 2023)")
    parser.add_argument("--download_only", action="store_true",
                       help="Only download, don't prepare CSV")
    parser.add_argument("--extract_only", action="store_true",
                       help="Only extract existing zip file, don't download")
    
    args = parser.parse_args()
    
    print("🔹 BirdCLEF Dataset Downloader (Kaggle)")
    print("=" * 60)
    print(f"📅 Year: {args.year}")
    print("🌐 Source: Kaggle Competitions")
    print("🐦 Content: Bird species identification challenge")
    print("📊 Format: Audio files with species annotations")
    print("=" * 60)
    
    # Check if zip file already exists
    zip_file = os.path.join(args.output_dir, f"birdclef-{args.year}.zip")
    if os.path.exists(zip_file):
        print(f"✅ Found existing zip file: {zip_file}")
        if args.extract_only:
            print("🔹 Extracting existing zip file...")
            success = extract_existing_zip(zip_file, args.output_dir)
        else:
            print("🔹 Zip file already exists. Use --extract_only to extract it.")
            success = True
    else:
        # Download dataset using Kaggle
        success = download_birdclef_kaggle(args.year, args.output_dir)
    
    if not success:
        print(f"\n❌ Failed to process BirdCLEF-{args.year}")
        return
    
    # Check if dataset was extracted successfully
    audio_dir = os.path.join(args.output_dir, "train_audio")
    metadata_file = os.path.join(args.output_dir, "train_metadata.csv")
    
    # Look for audio directory in common locations
    possible_audio_dirs = [
        audio_dir,
        os.path.join(args.output_dir, "audio"),
        os.path.join(args.output_dir, "train"),
        os.path.join(args.output_dir, "data", "train_audio")
    ]
    
    audio_dir_found = None
    for dir_path in possible_audio_dirs:
        if os.path.exists(dir_path) and len(list(Path(dir_path).rglob("*.wav"))) > 0:
            audio_dir_found = dir_path
            break
    
    if audio_dir_found:
        print(f"✅ Audio files found in: {audio_dir_found}")
        
        if not args.download_only:
            # Create CSV
            output_csv = os.path.join(args.output_dir, f"birdclef_{args.year}_dataset.csv")
            df = create_birdclef_csv(audio_dir_found, output_csv, metadata_file if os.path.exists(metadata_file) else None)
            
            if df is not None:
                print(f"\n🚀 Ready to run experiments!")
                print(f"   python run_birdclef_experiment.py --audio_dir {audio_dir_found} --csv_path {output_csv} --use_gpu")
            else:
                print(f"\n❌ Failed to create CSV. Check audio file structure.")
    else:
        print(f"❌ Audio files not found in expected locations")
        print(f"   Searched in: {possible_audio_dirs}")
        print(f"\n📁 Check the downloaded files in: {args.output_dir}")
        
        # List contents of output directory
        if os.path.exists(args.output_dir):
            print(f"🔹 Contents of {args.output_dir}:")
            for item in os.listdir(args.output_dir):
                item_path = os.path.join(args.output_dir, item)
                if os.path.isdir(item_path):
                    wav_count = len(list(Path(item_path).rglob("*.wav")))
                    print(f"   📁 {item}/ ({wav_count} WAV files)")
                else:
                    print(f"   📄 {item}")

def extract_existing_zip(zip_file, output_dir):
    """Extract an existing zip file."""
    try:
        print(f"🔹 Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ Successfully extracted {zip_file}")
        
        # Remove zip file after extraction
        os.remove(zip_file)
        print(f"🗑️ Removed {zip_file}")
        
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_file}: {e}")
        return False

if __name__ == "__main__":
    main()
