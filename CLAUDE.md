# RVC WebUI Project

## Dependencies

- Python: >=3.9,<3.11
- PyTorch: 2.4.0 (pinned for RVC compatibility)

## Project Structure

- Root project: RVC WebUI with specific PyTorch requirements
- executorch/: Independent sub-project with modern Python/PyTorch requirements
- Projects are managed separately (no UV workspace) to avoid dependency conflicts

## Installation

```bash
uv sync
```

## Notes

- The executorch subdirectory is a separate project with its own pyproject.toml
- Each project should be installed and managed independently
