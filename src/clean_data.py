import os
import shutil
import hashlib
import kagglehub

def clean_and_get_dataset_path(dataset_name="ibadeus/pcos-xai-ultrasound-dataset"):
    """
    Downloads the dataset from KaggleHub, scans for duplicate images using MD5 hashing,
    and copies only unique images to a local workspace folder: data/cleaned_pcos/.
    
    Returns the absolute path to the local cleaned dataset directory.
    """
    # 1. Download dataset via KaggleHub
    print("Downloading/checking latest version of dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download(dataset_name)
    print(f"Path to dataset files in cache: {downloaded_path}")
    
    raw_pcos_dir = os.path.join(downloaded_path, "PCOS")
    if not os.path.exists(raw_pcos_dir):
        # In case the directory structure is flat or different, check files
        raw_pcos_dir = downloaded_path
        
    # Define local cleaned destination directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    cleaned_dir = os.path.join(project_root, "data", "cleaned_pcos")
    
    print(f"Commencing data cleaning process. Target local directory: {cleaned_dir}")
    
    # Check if cleaned directory already has files. If it exists and already has unique files,
    # we can skip rebuilding to save time, unless forced.
    # Let's see if we can check the count to decide if we reuse.
    if os.path.exists(cleaned_dir):
        existing_count = sum(len(files) for r, d, files in os.walk(cleaned_dir))
        # We know there are exactly 3,996 unique images. If we find them, we can reuse.
        if existing_count == 3996:
            print("Cleaned dataset already exists locally with all 3,996 unique images. Skipping reconstruction.")
            return cleaned_dir
        else:
            print(f"Cleaned dataset found but count ({existing_count}) doesn't match expected (3996). Rebuilding...")
            shutil.rmtree(cleaned_dir)

    # Re-create cleaned directories
    os.makedirs(os.path.join(cleaned_dir, "infected"), exist_ok=True)
    os.makedirs(os.path.join(cleaned_dir, "noninfected"), exist_ok=True)
    
    hashes = set()
    total_scanned = 0
    copied_count = 0
    skipped_count = 0
    
    # Supported image extensions
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    for root, dirs, files in os.walk(raw_pcos_dir):
        for file in files:
            if file.lower().endswith(img_extensions):
                total_scanned += 1
                src_path = os.path.join(root, file)
                
                # Compute MD5
                with open(src_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash not in hashes:
                    hashes.add(file_hash)
                    
                    # Determine target class subdirectory based on path
                    # Normally either 'infected' or 'noninfected'
                    class_name = os.path.basename(root)
                    if class_name not in ["infected", "noninfected"]:
                        # Fallback check
                        if "infected" in root.lower() or "infected" in file.lower():
                            class_name = "infected"
                        else:
                            class_name = "noninfected"
                            
                    dst_path = os.path.join(cleaned_dir, class_name, file)
                    
                    # Ensure filename uniqueness in the destination folder (in case different duplicates have the same name)
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(cleaned_dir, class_name, f"{base}_{counter}{ext}")
                        counter += 1
                        
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                else:
                    skipped_count += 1
                    
    print(f"Data cleaning complete!")
    print(f"   - Total images scanned: {total_scanned}")
    # We should have exactly 3,996 copied images and 7,788 skipped duplicates
    print(f"   - Unique images copied: {copied_count}")
    print(f"   - Duplicate images skipped: {skipped_count}")
    
    return cleaned_dir

if __name__ == "__main__":
    clean_and_get_dataset_path()
