# 🌦️ MEGHA-AI
## (Multi-model Ensemble for Geospatial & Hyperlocal Agentic Atmospheric Intelligence)

MEGHA-AI is the self-contained Streamlit application. The application does **not require Ollama**. The application uses a
built-in rule-based intent parser for interaction with users.

## Repository structure

```text
MEGHA-AI-Streamlit/
├── app.py
├── requirements.txt
├── IN.txt
├── IND-DIS-732.json
├── ecmwf_aifs_india_YYYYMMDD_00z_merged.nc
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

Keep the supporting data files in the **same directory as `app.py`**.

### Required data files

1. `IN.txt` — GeoNames India location database used for location matching.
2. `IND-DIS-732.json` — India district boundary GeoJSON used for spatial/district operations.
3. ECMWF forecast NetCDF — the application searches for:

```text
ecmwf_aifs_india_YYYYMMDD_00z_merged.nc
```

Example:

```text
ecmwf_aifs_india_20260923_00z_merged.nc
```

The app automatically selects the latest file matching that pattern.

## Important: NetCDF size

GitHub rejects individual files larger than 100 MB on normal repositories. ECMWF NetCDF files can easily exceed this limit.

If your `.nc` file is >100 MB, **do not force it into normal GitHub history**. For Streamlit Community Cloud, a better deployment design is to keep the application code and small JSON/TXT assets in GitHub and host the large NetCDF file on an accessible object/file store, then modify `app.py` to download/cache it at startup.

If your NetCDF is below GitHub's file-size limit, it can be committed normally.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository, for example `megha-ai-streamlit`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `IN.txt`
   - `IND-DIS-732.json`
   - your matching `.nc` forecast file (only if it is small enough)
   - `.streamlit/config.toml`
   - `.gitignore`
3. Push the repository to GitHub.
4. In Streamlit Community Cloud, create a new app.
5. Select your GitHub repository.
6. Set the main file to:

```text
app.py
```

7. Deploy.


## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## No Ollama

Ollama is not required by this deployment. There is no `langchain-ollama` dependency in the deployment requirements.
