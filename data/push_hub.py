# pip install huggingface_hub[hf_transfer]
# huggingface-cli upload --repo-type dataset ustc-zhangzm/HybRank NQ_DPR-Multi .

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, CommitOperationCopy


api = HfApi()


repo_id = "ustc-zhangzm/HybRank"
repo_type = "dataset"

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type=repo_type,
)

folder_names = ["NQ_DPR-Multi", "TRECDL2019_ANCE", "TRECDL2020_ANCE", "MSMARCO_ANCE"]
for folder_name in folder_names:
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=folder_name,
        path_in_repo=folder_name,
        multi_commits=True,
        multi_commits_verbose=True,
    )


# code for operating on HuggingFace Repo
# https://huggingface.co/docs/huggingface_hub/guides/upload#createcommit
# operations = [
#     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
#     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
#     CommitOperationDelete(path_in_repo="old-weights.h5"),
#     CommitOperationDelete(path_in_repo="logs/"),
#     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
# ]
# api.create_commit(
#     repo_id=repo_id,
#     repo_type=repo_type,
#     operations=operations,
#     commit_message="move files",
# )
