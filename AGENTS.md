# Repository Guidelines

## Project Structure & Module Organization
- Core app code is in the repo root: `app.py` (Streamlit UI), `data_processing.py`, `database.py`, `vision_api.py`, and `imagen_api.py`.
- Input data files live in `data/` (for example `data/insta_dummy_data(in).csv`). Keep raw inputs stable and add derived exports in clearly named files alongside the source.
- Images for analysis live in `image/`; generated outputs are written to `generated_images/` (auto-created).
- No test directory exists yet; create `python/tests/` for new test suites.

## Build, Test, and Development Commands
- Create and activate a virtual environment: `python3 -m venv venv` then `source venv/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`.
- Run the app locally: `streamlit run app.py`.
- (If/when tests exist) run Python tests with `python3 -m pytest`.

## Coding Style & Naming Conventions
- Python follows PEP 8 with 88-column max; prefer snake_case for variables/functions and descriptive module names.
- Keep data-processing functions deterministic and reusable; avoid hidden global state.
- Name derived outputs after their source (e.g., `insta_dummy_data(in).csv` → `insta_dummy_data_enriched.csv`) to preserve provenance.

## Testing Guidelines
- No automated tests are present yet, so add coverage for new features.
- Place suites in `python/tests/` and name them after the module (e.g., `test_data_processing.py`).
- Run tests with `python3 -m pytest`.

## Commit & Pull Request Guidelines
- No commit-message convention is established yet; use short, scoped summaries like `<area>: <change>` (example: `app: add time analysis chart`).
- Keep commits single-purpose and avoid committing large binary assets unless required.
- PRs should describe datasets touched, list rerun commands, and include screenshots of UI changes.

## Security & Configuration Tips
- Keep API keys and service-account JSON files out of git; use untracked `.env` or local files.
- Generated artifacts (like `generated_images/`) should be excluded if they contain sensitive data.
