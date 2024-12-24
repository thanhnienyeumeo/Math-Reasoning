from huggingface_hub import HfApi, HfFolder, Repository
from credential import HUGGINGFACE_TOKEN
# Define your repository details
repo_name = "Colder203/phi_metamatk_gsm8k"
checkpoint_path = "phi/results/checkpoint-43455"
token = HUGGINGFACE_TOKEN

# Initialize the API and repository
api = HfApi()
# repo_url = api.create_repo(repo_name, token=token)
# repo = Repository(local_dir=checkpoint_path, clone_from=repo_url)

# upload_file
import os
list_dir = os.listdir(checkpoint_path)
for file in list_dir:
    api.upload_file(
        path_or_fileobj= os.path.join(checkpoint_path, file),
        path_in_repo='optimizer.pt',
        repo_id = 'Colder203/llama_finetune_metamath',
        repo_type='model'
    )