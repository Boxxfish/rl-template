# RL Template

A standard template I use for my RL projects.

## Quickstart

First, edit `pyproject.toml` and the folder names to your project. There are
also some import statements you'll have to change, but the easiest way to do
that is a global replacement of `rl_template` with your new project name.

Next, install the project. You'll need `uv` and >=Python3.11.

```bash
uv sync
cd RUST_FOLDER_NAME && maturin build --release  # Builds the Rust wheel for Python.
```

To see if everything works, run the PPO test. If you want logging, you'll need
to log into wandb, then uncomment the appropriate sections.

```bash
python PROJECT_NAME/experiments/test_ppo.py
```

## License

This template repository is MIT licensed.
