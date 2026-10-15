"""
Upload a staged folder to the Hugging Face Space (used by .github/workflows/deploy-hf-space.yml).

Usage: HF_TOKEN=... HF_SPACE_ID=owner/space python deploy/hf_space/upload.py <staged_dir> [<commit_sha>]

The Space mirrors the staged folder exactly: remote files that are not in the
folder are deleted in the same commit. huggingface_hub stores binary files
(model checkpoints) through the Hub's large-file storage automatically.
"""
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    folder = sys.argv[1]
    sha = sys.argv[2] if len(sys.argv) > 2 else "local"
    repo_id = os.environ["HF_SPACE_ID"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=folder,
        commit_message=f"Deploy {sha} from GitHub",
        delete_patterns="*",
    )
    print(f"Uploaded {folder} to https://huggingface.co/spaces/{repo_id} ({commit})")


if __name__ == "__main__":
    main()
