# RRF Savant Deployment

This repository packages a logistic regression model (stored in `logreg_rrf_savant.joblib`) trained on RRF-Savant meta-state features. The included FastAPI service exposes the model for online inference, making it easy to deploy the model behind an HTTP endpoint.

## Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   Set `MODEL_PATH` to override the default `logreg_rrf_savant.joblib` location if you store the model elsewhere.
   Use `MODEL_CONFIG_PATH` to point at a custom config file when the expected feature dimension differs from the bundled
   `config.json`.

3. **Send a prediction request**
   ```bash
   curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}' \
        http://localhost:8000/predict
   ```

   The API returns the predicted label and class probabilities from the logistic regression model.

## Endpoints

- `GET /health`: Simple health check.
- `POST /predict`: Accepts a 15-length `features` array of floats and returns the predicted class plus probabilities.

## Development notes

- The service validates that the incoming feature vector has the dimension specified in `config.json` (defaults to 15) so the
  model receives the correct shape.
- Model loading happens at startup, and requests will return an error if the model artifact is missing or cannot be loaded.

## Deploy as an API service

You can containerize the FastAPI service for deployment with Docker:

```bash
docker build -t rrf-savant-api .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/logreg_rrf_savant.joblib \
  -e MODEL_CONFIG_PATH=/app/config.json \
  rrf-savant-api
```

- Override `PORT` at runtime (`-e PORT=8080`) to change the exposed port.
- Mount or replace `logreg_rrf_savant.joblib` if you want to serve a different model (update `MODEL_PATH` accordingly).
- Point `MODEL_CONFIG_PATH` at the matching configuration when swapping models so feature validation aligns with the
  new artifact.
