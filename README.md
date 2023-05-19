# RL Template
A standard template I use for my RL projects.

## Quickstart
First, edit `pyproject.toml` and the folder names to your project.
Next, install the project. You'll need Poetry and >=Python3.9.
```bash
poetry install
poetry shell
```
To see if everything works, run the PPO test. You'll need to log into wandb first.
```bash
python PROJECT_NAME/experiments/test_ppo.py
```

## License
This template repository is MIT licensed.