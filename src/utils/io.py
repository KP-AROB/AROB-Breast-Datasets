import os, tempfile, shutil

def preload_to_local(src_paths, max_files=100, custom_dir=None):
    """
    Preloads files into a custom directory (SSD, /tmp, etc.) for faster processing.

    Args:
    - src_paths: List of file paths to be copied.
    - max_files: Maximum number of files to preload in one batch.
    - custom_dir: Optional custom directory path for caching files. If None, default system temp is used.
    
    Returns:
    - local_paths: List of paths to the locally cached files.
    - local_dir: Directory where files were cached.
    """
    if custom_dir:
        local_dir = custom_dir
        os.makedirs(local_dir, exist_ok=True)
    else:
        local_dir = tempfile.mkdtemp()

    local_paths = []
    for src in src_paths[:max_files]:
        try:
            filename = os.path.basename(src)
            dst = os.path.join(local_dir, filename)
            shutil.copy2(src, dst)
            local_paths.append(dst)
        except Exception as e:
            continue
    return local_paths, local_dir