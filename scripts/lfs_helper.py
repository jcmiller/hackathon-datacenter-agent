#!/usr/bin/env python3
import os
import re

def get_lfs_cache_path(repo_dir: str, file_path: str) -> str:
    """
    Given a git repository path and a file path inside that repository,
    resolves the path to the underlying git-lfs cache object.
    
    This allows reading the file directly from .git/lfs/objects/ without
    performing a full 'git checkout/pull' (saving disk space).
    """
    pointer_file = os.path.join(repo_dir, file_path)
    if not os.path.exists(pointer_file):
        raise FileNotFoundError(f"Pointer file not found: {pointer_file}")
        
    with open(pointer_file, "r", errors="ignore") as f:
        # Just read the first 200 chars to check if it's a pointer
        content = f.read(200)
        
    # If the file is already checked out (real file), it won't start with LFS header
    if not content.startswith("version https://git-lfs.github.com/spec/v1"):
        return pointer_file
        
    # LFS pointer files look like:
    # version https://git-lfs.github.com/spec/v1
    # oid sha256:8d37bcbb9e3d368d184ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0
    # size 123456
    match = re.search(r"oid sha256:([a-f0-9]{64})", content)
    if not match:
        return pointer_file # Fallback in case of custom formats or real file
        
    sha256 = match.group(1)
    
    # Git LFS stores files under: .git/lfs/objects/xx/yy/xxyy...
    cache_path = os.path.join(
        repo_dir, 
        ".git", 
        "lfs", 
        "objects", 
        sha256[0:2], 
        sha256[2:4], 
        sha256
    )
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"LFS object cache not found for {file_path}. "
            "Run 'git lfs fetch' to download it."
        )
        
    return cache_path

# Example Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: lfs_helper.py <repo_dir> <file_path>")
        sys.exit(1)
        
    try:
        path = get_lfs_cache_path(sys.argv[1], sys.argv[2])
        print(f"Resolved Cache Path: {path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
