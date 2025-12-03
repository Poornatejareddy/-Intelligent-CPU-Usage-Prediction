# CPU Usage Prediction System ⚡

A machine learning system designed to predict CPU usage for Kubernetes workloads based on resource requests, limits, and controller types. This project uses **DVC (Data Version Control)** for experiment tracking and **Streamlit** for model deployment.

## 🚀 Features
- **ML Pipeline**: Automated preprocessing, training, and evaluation using DVC.
- **Model Comparison**: Trains and evaluates Linear Regression, SVR, and Random Forest.
- **Interactive Dashboard**: Streamlit app for real-time predictions with dynamic visualizations.
- **Experiment Tracking**: Metrics (R², MAE, RMSE) are tracked and versioned.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Versioning**: DVC, Git
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit

## 📂 Project Structure
```
├── data/               # Dataset and processed files
├── models/             # Trained models
├── plots/              # Evaluation plots
├── src/                # Source code
│   ├── preprocess.py   # Data cleaning and splitting
│   ├── train.py        # Model training
│   └── evaluate.py     # Model evaluation
├── app.py              # Streamlit dashboard
├── dvc.yaml            # DVC pipeline definition
├── dvc.lock            # DVC lock file (reproducibility)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Running the ML Pipeline
To reproduce the entire pipeline (preprocessing → training → evaluation):
```bash
dvc repro
```
This will check for changes and run only the necessary stages.

### Running the Dashboard
To launch the Streamlit app:
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

## ☁️ Deployment (Azure Free Tier)
This app is ready to be deployed on Azure App Service.
1.  Create a Web App on Azure (Free F1 tier).
2.  Connect your GitHub repository.
3.  Set the startup command:
    ```bash
    python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
    ```

## 📊 Results
The current best model is **Linear Regression** with an R² score of ~0.90.
Evaluation metrics are stored in `evaluation.json`.

## 🤝 Contributing
1.  Fork the repo.
2.  Create a feature branch.
3.  Commit your changes.
4.  Push to the branch.
5.  Create a Pull Request.
