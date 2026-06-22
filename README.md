# Hendrix Mechanical Analytics

Interactive Streamlit application for analyzing stress-strain experiments, extracting material properties, and generating research-ready outputs.

The app allows users to upload mechanical-testing files, clean noisy stress-strain data, estimate Young's modulus, review curve-quality warnings, and export processed results directly from a web browser.

## Features

- Upload CSV, Excel, TXT, DAT, or TSV files
- Automatically detect strain and stress columns
- Convert stress values to MPa when units are available
- Apply baseline correction and negative-stress cleanup
- Remove spike outliers
- Use moving-average or Savitzky-Golay smoothing
- Detect early stress drops, unusually early peaks, and post-peak failure drops
- Calculate Young's modulus from a user-selected strain region
- Validate the modulus fit using slope and R²
- Visualize stress-strain curves with Plotly
- Download cleaned CSV, summary CSV, and interactive chart HTML
- Use included demonstration data when no file is uploaded

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select the repository and set the entrypoint file to `app.py`.
4. Deploy the app.
5. Share the resulting `streamlit.app` URL.

## Public-safety note

Only upload data that is approved for external use. Use generic file names and synthetic or public example data for demonstrations. Do not include proprietary sample labels, internal organization names, confidential metadata, or restricted instrument exports.
