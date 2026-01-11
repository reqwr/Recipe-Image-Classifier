# uploader_selective.py
import time
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm
import tempfile
import shutil
from datetime import datetime

# ---------------- CONFIG ----------------
REPO_ID = "cyuangli/WebEats-v3"
LOCAL_ROOT = Path("notebooks/data/images")
BATCH_SIZE = 500  # files per batch folder upload
SLEEP_SECONDS = 5  # reduced - HF handles rate limiting
CHECKPOINT_FILE = Path("uploaded_files.txt")
MAX_RETRIES = 3

# ============================================
# MANUAL SELECTION: Specify which meta_topics to upload
# ============================================
# Option 1: Upload specific meta_topics (uncomment and modify)
SELECTED_META_TOPICS = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]  # e.g., upload meta_topic_1, meta_topic_2, etc.

# Option 2: Upload a range (uncomment to use instead)
# SELECTED_META_TOPICS = list(range(1, 11))  # uploads meta_topic_1 through meta_topic_10

# Option 3: Upload all (uncomment to use instead)
# SELECTED_META_TOPICS = None  # uploads all meta_topics
# ============================================

api = HfApi()

# Track commits in this session
commits_made = 0

def load_checkpoint():
    """Load set of relative paths that have been uploaded."""
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_to_checkpoint(relative_paths):
    """Append relative paths to checkpoint file."""
    with open(CHECKPOINT_FILE, "a") as f:
        for path in relative_paths:
            f.write(f"{path}\n")

def get_relative_path(file_path, base_path):
    """Get path relative to base_path."""
    return str(file_path.relative_to(base_path))

def get_pending_files(meta_topic_path, uploaded_set, base_path):
    """Get list of files not yet uploaded."""
    all_files = [p for p in meta_topic_path.rglob("*") if p.is_file()]
    pending = []
    for p in all_files:
        rel_path = get_relative_path(p, base_path)
        if rel_path not in uploaded_set:
            pending.append(p)
    return pending

def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def upload_batch_via_folder(batch, meta_topic_name, base_path):
    """
    Upload batch using upload_folder (much faster than individual uploads).
    Creates temporary folder structure and uploads in one operation.
    """
    global commits_made
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "images" / meta_topic_name
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Copy batch files to temp structure
        for file_path in batch:
            dest = temp_path / file_path.name
            shutil.copy2(file_path, dest)
        
        # Upload entire temp folder
        api.upload_folder(
            folder_path=str(Path(temp_dir) / "images"),
            path_in_repo="images",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # Increment commit counter
        commits_made += 1
        
        # Save checkpoint
        relative_paths = [get_relative_path(p, base_path) for p in batch]
        save_to_checkpoint(relative_paths)

def upload_batch_with_retry(batch, meta_topic_name, base_path):
    """Upload batch with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            upload_batch_via_folder(batch, meta_topic_name, base_path)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    ❌ Failed after {MAX_RETRIES} attempts: {e}")
                return False
            print(f"    ⚠️  Attempt {attempt} failed: {e}")
            print(f"    Retrying in {SLEEP_SECONDS * attempt}s...")
            time.sleep(SLEEP_SECONDS * attempt)
    return False

def upload_meta_topic(meta_topic_folder, uploaded_set, base_path):
    """Upload all pending images in a meta_topic folder."""
    global commits_made
    
    pending_files = get_pending_files(meta_topic_folder, uploaded_set, base_path)
    
    if not pending_files:
        print(f"✓ {meta_topic_folder.name}: No new images")
        return
    
    print(f"\n📁 {meta_topic_folder.name}: {len(pending_files)} images to upload")
    
    batches = list(chunk_list(pending_files, BATCH_SIZE))
    failed_batches = []
    
    for i, batch in enumerate(tqdm(batches, desc=f"  Uploading batches"), start=1):
        success = upload_batch_with_retry(batch, meta_topic_folder.name, base_path)
        if not success:
            failed_batches.append(i)
        else:
            # Print commit count after successful upload
            print(f"    📊 Total commits made: {commits_made}")
        
        # Sleep between batches (except after last batch)
        if i < len(batches):
            time.sleep(SLEEP_SECONDS)
    
    if failed_batches:
        print(f"  ⚠️  Failed batches: {failed_batches}")
    else:
        print(f"  ✅ All batches uploaded successfully")

def get_selected_meta_topics():
    """Get list of meta_topic folders based on SELECTED_META_TOPICS."""
    all_meta_topics = sorted([d for d in LOCAL_ROOT.iterdir() if d.is_dir()])
    
    # If no selection, return all
    if SELECTED_META_TOPICS is None:
        return all_meta_topics
    
    # Filter based on selected numbers
    selected_folders = []
    for num in SELECTED_META_TOPICS:
        folder_name = f"meta_topic_{num}"
        folder_path = LOCAL_ROOT / folder_name
        if folder_path.exists() and folder_path.is_dir():
            selected_folders.append(folder_path)
        else:
            print(f"⚠️  Warning: {folder_name} not found, skipping")
    
    return sorted(selected_folders)

def main():
    global commits_made
    
    print(f"🚀 Starting upload to {REPO_ID}")
    print(f"📂 Local root: {LOCAL_ROOT}")
    print(f"⚙️  Batch size: {BATCH_SIZE} files")
    print(f"⏱️  Sleep between batches: {SLEEP_SECONDS}s\n")
    
    # Load checkpoint
    uploaded_set = load_checkpoint()
    print(f"📋 Checkpoint loaded: {len(uploaded_set)} files already uploaded\n")
    
    # Get selected meta_topic folders
    meta_topics = get_selected_meta_topics()
    
    if SELECTED_META_TOPICS is None:
        print(f"📌 Mode: Uploading ALL meta_topics")
    else:
        print(f"📌 Mode: Uploading SELECTED meta_topics: {SELECTED_META_TOPICS}")
    
    print(f"Found {len(meta_topics)} meta_topic folder(s) to process\n")
    
    if not meta_topics:
        print("❌ No meta_topic folders to upload. Check your SELECTED_META_TOPICS configuration.")
        return
    
    # Upload each meta_topic
    start_time = datetime.now()
    for meta_topic_folder in meta_topics:
        upload_meta_topic(meta_topic_folder, uploaded_set, LOCAL_ROOT)
    
    elapsed = datetime.now() - start_time
    print(f"\n" + "="*60)
    print(f"✅ Upload complete!")
    print(f"📊 Total commits made in this session: {commits_made}")
    print(f"⏱️  Total time: {elapsed}")
    print(f"="*60)
    
    # Warning if approaching limit
    if commits_made >= 80:
        print(f"\n⚠️  WARNING: You've made {commits_made} commits.")
        print(f"    HuggingFace typically has a limit around 100 commits per hour.")
        print(f"    Consider waiting before uploading more.")
    elif commits_made >= 50:
        print(f"\n💡 Note: You've made {commits_made} commits. Approaching the hourly limit.")

if __name__ == "__main__":
    main()