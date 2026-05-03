#  Solar Power Generation Forecasting — MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![DVC](https://img.shields.io/badge/DVC-3.x-purple)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF)
![AWS](https://img.shields.io/badge/AWS-ECR%20%2B%20EC2-FF9900)

An end-to-end MLOps pipeline for predicting solar power (AC output) from a photovoltaic plant using irradiation, ambient temperature, and time-of-day features.

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started (Local)](#-getting-started-local)
- [FastAPI — REST Endpoints](#-fastapi--rest-endpoints)
- [CI/CD Pipeline](#-cicd-pipeline)
- [AWS Deployment](#-aws-deployment)
- [CloudWatch Monitoring](#-cloudwatch-monitoring)
- [Environment Variables](#-environment-variables)

---

##  Project Overview

This project forecasts AC power output from a 5MW solar plant using machine learning. It is built with production MLOps practices:

- **Data versioning** with DVC
- **Experiment tracking** with MLflow
- **REST API** with FastAPI
- **Interactive UI** with Streamlit
- **Automated CI/CD** with GitHub Actions
- **Cloud deployment** on AWS EC2 via ECR
- **Live monitoring** with AWS CloudWatch

---

##  Architecture

```
Local Development
        │
        ├── DVC          → data versioning
        ├── MLflow       → experiment tracking
        └── model.pkl    → trained model exported
                │
                │  git push
                ▼
        GitHub Actions
                │
                ├── Install dependencies
                ├── Build Docker image
                ├── Push to AWS ECR
                └── Deploy to AWS EC2
                            │
                ┌───────────┴────────────┐
                :8501                 :8000
            Streamlit UI           FastAPI
            (humans)               (machines)
                            │
                    AWS CloudWatch
                    (logs & monitoring)
```

---

##  Tech Stack

| Category | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| ML Framework | Scikit-learn | 1.4+ |
| Experiment Tracking | MLflow | 2.11 |
| Data Versioning | DVC | 3.x |
| Web UI | Streamlit | 1.32 |
| REST API | FastAPI + Uvicorn | 0.111 |
| Containerization | Docker | — |
| CI/CD | GitHub Actions | — |
| Container Registry | AWS ECR | — |
| Cloud Server | AWS EC2 (t3.micro) | — |
| Monitoring | AWS CloudWatch | — |

---

##  Project Structure

```
Solar_Power_Generation_Forecasting/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          ← GitHub Actions CI/CD pipeline
│
├── data/
│   ├── raw/                   ← DVC-tracked raw data
│   │   ├── Plant_1_Generation_Data.csv.dvc
│   │   └── Plant_1_Weather_Sensor_Data.csv.dvc
│   └── processed/
│       └── train.csv
│
├── app/                       ← Production application
│   ├── app.py                 ← Streamlit UI (Dashboard, Manual, Real-Time)
│   ├── api.py                 ← FastAPI REST API
│   ├── config.py              ← Configuration & constants
│   ├── model_loader.py        ← Loads model.pkl
│   ├── utils.py               ← Feature engineering + CSV helpers
│   ├── weather_service.py     ← OpenWeatherMap integration
│   ├── Dockerfile             ← Container definition
│   ├── start.sh               ← Launches FastAPI + Streamlit together
│   └── model/
│       └── model.pkl          ← Exported trained model
│
├── src/
│   ├── preprocess.py          ← Data preprocessing
│   ├── train.py               ← Model training + MLflow logging
│   ├── registry.py            ← MLflow model registry
│   ├── export_model.py        ← Export model.pkl from MLflow
│   └── main.py                ← Pipeline orchestrator
│
├── tests/
│   └── test_pipeline.py       ← pytest test suite
│
├── requirements.txt           ← All dependencies
└── README.md
```

---

##  Getting Started (Local)

### Prerequisites

- Python 3.10+
- Docker Desktop
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/solar-power-forecasting-mlops.git
cd solar-power-forecasting-mlops
```

### 2. Create and activate virtual environment

```bash
python -m venv .solarprojvenv

# Windows
.solarprojvenv\Scripts\activate

# Mac/Linux
source .solarprojvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
# Windows
set WEATHER_API_KEY=your_openweathermap_key

# Mac/Linux
export WEATHER_API_KEY=your_openweathermap_key
```

### 5. Run Streamlit app

```bash
cd app
streamlit run app.py
```

Open: `http://localhost:8501`

### 6. Run FastAPI (separate terminal)

```bash
cd app
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://localhost:8000/docs`

---

##  FastAPI — REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API and model status |
| `POST` | `/predict` | Predict AC power (manual inputs) |
| `POST` | `/predict/now` | Predict using current time + live weather |
| `GET` | `/weather` | Fetch live weather data |
| `GET` | `/history` | Last N predictions from CSV |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "datetime_str": "2024-06-15 13:30:00",
    "irradiation": 0.75,
    "amb_temp": 32.5,
    "save": true
  }'
```

### Example Response

```json
{
  "ac_power_kw": 3240.50,
  "capacity_pct": 64.81,
  "hour": 13,
  "hour_sin": 0.9659,
  "hour_cos": -0.2588,
  "saved": true
}
```

---

##  CI/CD Pipeline

Every `git push` to `main` triggers the full pipeline automatically:

```
git push → GitHub Actions
    │
    ├── ✅ Install Python dependencies
    ├── ✅ Build Docker image
    ├── ✅ Run container (smoke test)
    ├── ✅ Push image to AWS ECR
    └── ✅ SSH into EC2 → pull → deploy
```

### GitHub Secrets Required

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_REGION` | e.g. `eu-north-1` |
| `ECR_REGISTRY` | e.g. `435998720725.dkr.ecr.eu-north-1.amazonaws.com` |
| `EC2_HOST` | EC2 public IP (update on each restart) |
| `EC2_SSH_KEY` | Contents of `.pem` key file |
| `WEATHER_API_KEY` | OpenWeatherMap API key |

---

## ☁️ AWS Deployment

### Services Used

| Service | Purpose |
|---|---|
| **ECR** | Stores Docker images |
| **EC2 (t3.micro)** | Runs the containerized app |
| **CloudWatch** | Logs and monitoring |
| **IAM** | Role-based access for EC2 → CloudWatch |

### Deployed App URLs

```
Streamlit UI  →  http://EC2_PUBLIC_IP:8501
FastAPI Docs  →  http://EC2_PUBLIC_IP:8000/docs
```

### Docker — How Both Apps Run Together

The container runs **both Streamlit and FastAPI** using a startup script:

```bash
# app/start.sh
#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

```dockerfile
# app/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh
EXPOSE 8501
EXPOSE 8000
CMD ["./start.sh"]
```

### Docker — How Both Apps Run Together

The container runs **both Streamlit and FastAPI** using a startup script:

```bash
# app/start.sh
#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

```dockerfile
# app/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh
EXPOSE 8501
EXPOSE 8000
CMD ["./start.sh"]
```

### Manual Deployment Steps

```bash
# 1. Login to ECR
aws ecr get-login-password --region eu-north-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ECR_REGISTRY

# 2. Pull latest image
docker pull YOUR_ECR_REGISTRY/mlops/solar-power-forecasting:latest

# 3. Run container
docker run -d -p 8000:8000 -p 8501:8501 \
  -e WEATHER_API_KEY=your_key \
  --name solar-container \
  --restart always \
  YOUR_ECR_REGISTRY/mlops/solar-power-forecasting:latest
```

---

##  CloudWatch Monitoring

Prediction logs are streamed to AWS CloudWatch automatically:

```
Log group:  /solar-app/container
Log stream: solar-container
```

Each prediction logs:

```
[PREDICTION] id=1 | datetime=2024-06-15 13:30:00 |
irradiation=0.75 | amb_temp=32.5 | ac_power=3240.50 kW | mode=manual
```

Filter logs in CloudWatch console using `[PREDICTION]` to see only predictions.

---

##  Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WEATHER_API_KEY` | Optional | OpenWeatherMap API key for live weather |
| `SOLAR_CITY` | Optional | City for weather (default: Mumbai) |

> App runs without `WEATHER_API_KEY` — falls back to last CSV row or random values.

---

## 📄 License

This project is for academic/MSc purposes.

---

<p align="center">Built with ☀️ for MSc AI — Solar Power MLOps Project</p>
