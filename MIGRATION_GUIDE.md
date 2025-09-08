# Migration Guide: From Conda to UV

This document explains how to migrate from the Conda-based environment setup to the new UV-based setup.

## Why UV?

UV is a fast Python package installer and resolver built by Astral (the creators of Ruff). It provides several advantages over traditional pip and conda:

- **Speed**: UV is significantly faster than pip for dependency resolution and installation
- **Better dependency resolution**: More reliable handling of version conflicts
- **Lock file support**: Reproducible builds with `uv.lock`
- **Virtual environment management**: Built-in venv support
- **PyPI compatibility**: Works with all PyPI packages

## Migration Steps

### For New Users

Simply follow the updated installation instructions in the README.md using UV.

### For Existing Users

If you have an existing Conda environment, you can migrate as follows:

1. **Backup your current environment** (optional):
   ```bash
   conda env export -n oct > backup_environment.yaml
   ```

2. **Remove the old Conda environment** (optional):
   ```bash
   conda env remove -n oct
   ```

3. **Install UV**:
   ```bash
   pip install uv
   ```

4. **Set up the new UV environment**:
   ```bash
   chmod +x make_env_uv.sh
   ./make_env_uv.sh
   ```

5. **Activate the new environment**:
   ```bash
   source .venv/bin/activate
   ```

6. **Install system dependencies**:
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Key Differences

| Aspect | Conda (Old) | UV (New) |
|--------|-------------|----------|
| Environment file | `environment.yaml` | `pyproject.toml` + `uv.lock` |
| Setup script | `make_env.sh` | `make_env_uv.sh` |
| Activation | `conda activate oct` | `source .venv/bin/activate` |
| Package management | `conda install` / `pip install` | `uv pip install` |
| Dependency locking | Manual export | Automatic `uv.lock` |

## Conda Dependencies Handled

The following conda-specific packages have been addressed:

- **ffmpeg**: Now requires system-level installation (see installation guide)
- **ffmpeg-python**: Remains as a pip package, automatically installed
- **python=3.11.8**: Handled by UV virtual environment creation
- **pip**: Not needed, UV handles package installation

## Backward Compatibility

The original Conda setup is still available for users who prefer it:

- `environment.yaml` is preserved for reference
- `make_env.sh` still works for conda users
- Installation instructions include both methods

## Troubleshooting

### Common Issues

1. **Import errors after migration**: Ensure you've activated the UV virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. **FFmpeg not found**: Install ffmpeg at the system level as described in the installation guide.

3. **Dependency conflicts**: UV provides better error messages. Check the conflict resolution suggestions.

### Getting Help

If you encounter issues with the migration:

1. Check that UV is properly installed: `uv --version`
2. Verify Python version: `python --version` (should be 3.11.x)
3. Ensure virtual environment is activated
4. For dependency issues, try recreating the environment:
   ```bash
   rm -rf .venv
   ./make_env_uv.sh
   ```

## Performance Benefits

Users should expect:
- Faster initial environment setup
- Quicker package installations and updates
- More reliable dependency resolution
- Reproducible builds across different machines

## Next Steps

After migration:
1. Test your existing workflows to ensure compatibility
2. Use `uv pip install <package>` for new package installations
3. Commit the `uv.lock` file for reproducible environments
4. Update any CI/CD scripts to use the new UV-based setup