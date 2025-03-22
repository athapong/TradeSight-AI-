# Contributing to the Trading Entry Point Prediction Project

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/yourusername/trading.git
   cd trading
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

## Project Structure

The project is organized into the following modules:

- `src/data/`: Data loading and feature engineering
- `src/models/`: Model implementation
- `src/visualization/`: Visualization utilities
- `src/backtesting/`: Backtesting functionality
- `src/utils/`: Shared utilities

## Coding Guidelines

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Add type hints where possible
- Write unit tests for new functionality

## Git Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request from your fork to the main repository

## Running Tests

We use pytest for testing. Run the tests with:

```bash
pytest
```

## Adding New Features

### Adding a New Model

1. Create a new file in `src/models/` (e.g., `gradient_boosting_model.py`)
2. Implement the model class by extending `BaseModel`
3. Register the model in `src/models/__init__.py`
4. Add tests in `tests/models/`

### Adding New Technical Indicators

1. Add the indicator calculation to `src/data/features.py`
2. Update the `add_technical_indicators` function
3. Add the indicator to the feature list in `src/data/features.py`

## Releasing New Versions

1. Update the version number in `setup.py`
2. Update the changelog
3. Create a new release on GitHub

## Questions or Need Help?

Feel free to open an issue for any questions or problems you encounter.

Thank you for contributing!
