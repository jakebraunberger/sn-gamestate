---
license: gpl-3.0
---

# SoccerNet Challenge 2025 - Game State Reconstruction

## Download the dataset

Install the huggingface_hub pip package:

```bash
pip install huggingface_hub[cli]
```

Download the dataset with the following Python code :

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-GSR-2025",
                  repo_type="dataset", revision="main",
                  local_dir="SoccerNet/SN-GSR-2025")
```