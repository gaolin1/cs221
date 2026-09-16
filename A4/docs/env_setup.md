# 🧪 Setting Up the Python Environment with Conda and `uv`

This repository uses **Conda** (Anaconda, Miniconda, or Miniforge) as the primary toolchain for environment management, leveraging [`uv`](https://docs.astral.sh/uv/) instead of standard `pip` for fast Python package installations.

By combining Conda and `uv`:
- **Conda** manages the core environment (`XCS221`), Python runtime, and C/system dependencies defined in `environment.yml`.
- **`uv`** acts as a drop-in, high-performance replacement for `pip` (`uv pip install`), resolving and installing Python packages from `requirements.txt`.

---

## 📋 Prerequisites

1. **Conda**: Install [Anaconda](https://www.anaconda.com/), [Miniconda](https://docs.anaconda.com/miniconda/), or [Miniforge](https://github.com/conda-forge/miniforge).
2. **uv**: `uv` is included in `environment.yml` and installed automatically into the Conda environment. Alternatively, you can install [`uv`](https://docs.astral.sh/uv/getting-started/installation) globally on your system.

---

# 📦 Creating the Environment

From the root of this repository, run the following command to set up and activate the environment:

```bash
source install.sh
```

### What `install.sh` Does:
1. **Locates `environment.yml`**: Checks for `gradescope/environment.yml` first, falling back to `src/environment.yml`.
2. **Locates `requirements.txt`**: Checks for `gradescope/requirements.txt` first, falling back to `src/requirements.txt`.
3. **Creates/Updates Conda Environment**: Builds or updates the `XCS221` Conda environment from `environment.yml`.
4. **Activates Environment**: Activates `XCS221` in your current shell session.
5. **Installs Python Packages via `uv`**: Uses `uv pip install -r <requirements_file>` instead of `pip` to install all Python dependencies into the active `XCS221` environment.

---

## 🚀 Activating the Environment

For new terminal sessions, activate your environment with:

```bash
conda activate XCS221
```

*(You can also re-run `source install.sh` at any time.)*

To deactivate:
```bash
conda deactivate
```

> [!IMPORTANT]  
> Remember to activate the `XCS221` Conda environment in every new terminal session before running assignment code or tests.

---

## 🔄 Refreshing the Environment

If you need to tear down and perform a clean re-installation of the environment:

```bash
source install.sh -r
```

This will remove the existing `XCS221` Conda environment, recreate it from scratch from `environment.yml`, activate it, and re-install all requirements with `uv`.

---

## 🤖 Autograder Compatibility

The Gradescope autograder uses the exact same Conda + `uv` setup. By running `source install.sh` locally, your local environment will match the autograder environment.