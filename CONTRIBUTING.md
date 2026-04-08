# Contributing to VDrift RL

Thank you for your interest in contributing to VDrift RL! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

Feature requests and enhancement suggestions are welcome! Please:
- Check if a similar suggestion already exists
- Clearly describe the feature and its benefits
- Provide use cases or examples

### Pull Requests

1. **Fork the repository** and create a new branch for your feature/fix
2. **Make your changes** following the code style guidelines below
3. **Test your changes** thoroughly
4. **Update documentation** if needed (README, docs/, docstrings)
5. **Commit your changes** with clear, descriptive commit messages
6. **Submit a pull request** with a description of your changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/kubasienki/aiRacing.git
cd vdrift-rl

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,training,visualization]"

# Run tests (when available)
pytest tests/
```

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use `black` for code formatting:
  ```bash
  black src/ examples/ tests/
  ```
- Use `flake8` for linting:
  ```bash
  flake8 src/ examples/ tests/
  ```
- Add type hints where appropriate
- Write docstrings for public functions and classes (Google style)

### Example Docstring

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Longer description with more details about what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this specific error occurs
    """
    pass
```

### C++ (VDrift modifications)

- Follow existing VDrift code style
- Use meaningful variable names
- Add comments for complex logic
- Test changes with both Python and standalone VDrift

## Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues/PRs when relevant

Example:
```
Add support for custom reward functions

- Add RewardFunction base class
- Update VDriftEnv to accept reward_fn parameter
- Add examples of custom reward functions
- Update documentation

Closes #42
```

## Testing

### Manual Testing

Test your changes with:
```bash
# Run random agent
python examples/random_agent.py

# Test basic environment creation
python -c "import gym; import vdrift_rl; env = gym.make('VDrift-v0'); print('Success!')"
```

### Unit Tests (Future)

When writing new features, consider adding tests in `tests/` directory.

## Documentation

### Code Documentation

- Add docstrings to all public APIs
- Update type hints
- Include usage examples in docstrings

### User Documentation

Update relevant files in `docs/` when:
- Adding new features
- Changing APIs
- Modifying behavior

### README Updates

Update the main README.md if:
- Adding major features
- Changing installation steps
- Adding new examples

## Project Structure

Understanding the project structure:

```
vdrift-rl/
├── src/vdrift_rl/         # Main Python package
│   ├── __init__.py
│   └── vdriftenv.py        # Core environment
├── vdrift/                 # VDrift simulator (submodule)
├── examples/               # Example scripts
├── docs/                   # Documentation
├── tests/                  # Unit tests
└── tools/                  # Utility scripts
```

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- Unit tests for the environment
- Additional reward function implementations
- Performance optimizations
- Better error handling and logging
- Documentation improvements

### Medium Priority
- Additional wrappers (e.g., action smoothing)
- Visualization tools
- Training utilities and callbacks
- Support for different VDrift tracks
- Multi-agent racing support

### Future Ideas
- Gymnasium (newer Gym) compatibility
- Support for other racing simulators
- Imitation learning from human demonstrations
- Curriculum learning helpers

## License

By contributing to VDrift RL, you agree that your contributions will be licensed under the MIT License (for Python code) or GPL-3.0 (for VDrift modifications). See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to the maintainers

Thank you for contributing! 🏎️
