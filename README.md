# MLOps Tools & Platforms

This guide covers essential MLOps (Machine Learning Operations) tools and platforms for managing complete ML lifecycle from experimentation to production deployment. MLOps bridges gap between data science and production engineering. When you train model on laptop it works fine. When you need to deploy it for thousands of users, retrain regularly with new data, track experiments, version datasets, monitor production performance - that's where MLOps tools become critical.

---

## Why MLOps Needed

**Understanding MLOps:** Traditional software engineering has DevOps for continuous integration, deployment, monitoring. Machine learning adds complexity - models degrade over time, need retraining, data versioning, experiment tracking, hyperparameter tuning, A/B testing. MLOps provides tools and practices to operationalize ML systems reliably at scale.

Real-world problem: Data scientist trains object detection model achieving 95% mAP on test set. Works great on laptop. Deploy to production. After 2 months accuracy drops to 70%. Why? Data distribution changed, no monitoring detected it, no automated retraining pipeline exists, can't reproduce original results because training parameters lost, dataset version unknown. This is chaos without MLOps.

MLOps solves:
- Experiment tracking: Compare hundreds of training runs, hyperparameters, metrics
- Model versioning: Track which model version deployed where
- Data versioning: Reproduce training with exact dataset version
- Pipeline automation: Automate training, testing, deployment workflows
- Monitoring: Track model performance in production, detect drift
- Collaboration: Share experiments, models, pipelines across team
- Reproducibility: Recreate any experiment or model from history
- Governance: Compliance, audit trails, model approval workflows

---

## MLflow

**Open-source ML lifecycle platform:** MLflow is most widely adopted MLOps tool. Built by Databricks, now Linux Foundation project. Tracks experiments, packages models, deploys anywhere. Works with any ML library - PyTorch, TensorFlow, scikit-learn, XGBoost. Four main components solving different MLOps problems.

Why MLflow dominates: Simple to start, powerful at scale. Used by thousands of companies from startups to enterprises. Open source with no vendor lock-in. Integrates with everything. Can run locally or in cloud. Minimal code changes needed.

### MLflow Tracking

**Experiment tracking and comparison:** Log parameters, metrics, artifacts, models from training runs. Compare experiments in web UI. Answer questions like "which learning rate gave best accuracy?" or "what hyperparameters did production model use?".

**Core concepts:**

**Experiment**: Group of related runs. Example: "yolov8_training" experiment contains all YOLOv8 training attempts.

**Run**: Single training execution. Logs parameters (learning rate, batch size), metrics (loss, mAP, precision), artifacts (model weights, plots, logs).

**Artifact**: Any file generated during run. Model checkpoints, confusion matrices, training curves, preprocessed data samples.

**Metrics**: Numeric values logged over time. Training loss per epoch, validation mAP per iteration.

**Parameters**: Hyperparameters used. Learning rate, epochs, architecture config.

**Installation:**
```bash
pip install mlflow

# Start tracking server
mlflow server --host 0.0.0.0 --port 5000
```

**Basic tracking example:**
```python
"""
MLflow Tracking - Basic Usage
------------------------------
Track object detection training
"""
import mlflow
import mlflow.pytorch
import torch
from ultralytics import YOLO

# Set tracking URI (local or remote server)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("yolov8_training")

# Start run
with mlflow.start_run(run_name="yolov8n_640_lr0.01"):

    # Log parameters
    mlflow.log_param("model", "yolov8n")
    mlflow.log_param("img_size", 640)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 100)

    # Train model
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01
    )

    # Log metrics
    mlflow.log_metric("mAP50", results.results_dict['metrics/mAP50(B)'])
    mlflow.log_metric("mAP50-95", results.results_dict['metrics/mAP50-95(B)'])
    mlflow.log_metric("precision", results.results_dict['metrics/precision(B)'])
    mlflow.log_metric("recall", results.results_dict['metrics/recall(B)'])

    # Log artifacts
    mlflow.log_artifact("runs/train/weights/best.pt")
    mlflow.log_artifact("runs/train/results.png")
    mlflow.log_artifact("runs/train/confusion_matrix.png")

    # Log model
    mlflow.pytorch.log_model(model, "model")

    print(f"Run logged to: {mlflow.get_tracking_uri()}")
```

**Advanced tracking with autolog:**
```python
"""
MLflow Autolog - Automatic tracking
------------------------------------
Automatically logs parameters, metrics, models
"""
import mlflow
import mlflow.pytorch

# Enable autologging for PyTorch
mlflow.pytorch.autolog(
    log_models=True,
    log_every_n_epoch=1,
    log_every_n_step=None
)

# Train model - automatically logged
with mlflow.start_run():
    # Your training code here
    # MLflow automatically logs:
    # - Model architecture
    # - Optimizer parameters
    # - Loss values
    # - Training metrics
    # - Model checkpoints
    pass
```

**Comparing experiments programmatically:**
```python
"""
Query and compare experiments
------------------------------
"""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("yolov8_training")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.mAP50 > 0.9",  # Filter high-performing runs
    order_by=["metrics.mAP50 DESC"],       # Sort by mAP
    max_results=10
)

# Compare top runs
for run in runs:
    print(f"\nRun ID: {run.info.run_id}")
    print(f"mAP50: {run.data.metrics['mAP50']:.4f}")
    print(f"Learning Rate: {run.data.params['learning_rate']}")
    print(f"Batch Size: {run.data.params['batch_size']}")
```

**Nested runs for hyperparameter search:**
```python
"""
Nested runs for grid search
----------------------------
Parent run contains multiple child runs
"""
import mlflow

with mlflow.start_run(run_name="hyperparameter_search") as parent_run:

    # Grid search
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [8, 16, 32]

    for lr in learning_rates:
        for bs in batch_sizes:

            # Child run for each combination
            with mlflow.start_run(nested=True, run_name=f"lr{lr}_bs{bs}"):

                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("batch_size", bs)

                # Train and log results
                # ...
                mlflow.log_metric("mAP50", mAP)

    # Log best combination to parent
    best_mAP = max_mAP_from_children
    mlflow.log_metric("best_mAP50", best_mAP)
```

### MLflow Models

**Model packaging and deployment:** Standardized format for packaging ML models. Works with any framework. Deploy to REST API, AWS SageMaker, Azure ML, Kubernetes. Model includes all dependencies, preprocessing code, inference logic.

**Model flavors:** MLflow supports multiple "flavors" - PyTorch, TensorFlow, ONNX, scikit-learn, custom Python functions.

**Saving models:**
```python
"""
MLflow Models - Save and load
------------------------------
"""
import mlflow
import mlflow.pytorch
import torch

# Save PyTorch model
with mlflow.start_run():
    model = YourModel()
    # Train model...

    # Save with MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="yolov8_detector",
        conda_env={
            'dependencies': [
                'python=3.9',
                'pytorch=2.0',
                'torchvision=0.15',
                'onnxruntime=1.15'
            ]
        }
    )

# Load model for inference
loaded_model = mlflow.pytorch.load_model("runs:/<run_id>/model")
predictions = loaded_model(input_data)
```

**Custom Python function models:**
```python
"""
Custom model with preprocessing
--------------------------------
Package custom inference logic
"""
import mlflow
from mlflow.pyfunc import PythonModel
import numpy as np
import cv2

class ObjectDetectorWrapper(PythonModel):
    """Custom model wrapper with preprocessing"""

    def load_context(self, context):
        """Load model and dependencies"""
        import onnxruntime as ort
        self.session = ort.InferenceSession(context.artifacts["model"])

    def preprocess(self, image_bytes):
        """Preprocess input image"""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize and normalize
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, context, model_input):
        """Run inference with preprocessing"""
        # Preprocess
        input_tensor = self.preprocess(model_input['image'][0])

        # Inference
        outputs = self.session.run(None, {'images': input_tensor})

        # Postprocess
        detections = self.postprocess(outputs)

        return detections

# Save custom model
artifacts = {"model": "model.onnx"}

mlflow.pyfunc.log_model(
    artifact_path="detector",
    python_model=ObjectDetectorWrapper(),
    artifacts=artifacts,
    registered_model_name="custom_detector"
)
```

**Serving models as REST API:**
```bash
# Serve model locally
mlflow models serve -m "models:/yolov8_detector/1" -p 5001

# Test endpoint
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[0.1, 0.2, 0.3, ...]]}'
```

### MLflow Model Registry

**Centralized model store:** Registry for managing model lifecycle. Version models, stage transitions (Staging → Production), approval workflows, model lineage.

**Model stages:**
- **None**: Initial registration
- **Staging**: Testing in staging environment
- **Production**: Deployed to production
- **Archived**: Deprecated models

**Using model registry:**
```python
"""
MLflow Model Registry
---------------------
Manage model lifecycle
"""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
run_id = "abc123"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="yolov8_detector"
)

model_name = result.name
version = result.version

print(f"Registered model: {model_name} version {version}")

# Transition to staging
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

# Add description
client.update_model_version(
    name=model_name,
    version=version,
    description="YOLOv8n trained on custom dataset, 95% mAP50"
)

# After testing, promote to production
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)
```

**Model approval workflow:**
```python
"""
Approval workflow for model deployment
---------------------------------------
"""
def approve_model_for_production(model_name, version, approver):
    """
    Approve model after validation.

    Steps:
    1. Run validation tests
    2. Check performance benchmarks
    3. Get approval
    4. Transition to production
    """
    client = MlflowClient()

    # Get model version details
    mv = client.get_model_version(model_name, version)

    # Run validation
    validation_passed = run_model_validation(mv)

    if not validation_passed:
        print("Model failed validation")
        return False

    # Add approval tag
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="approved_by",
        value=approver
    )

    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True  # Archive old production models
    )

    print(f"Model {model_name} v{version} promoted to Production")
    return True
```

### MLflow Projects

**Reproducible ML code:** Package ML code in reusable format. Specify dependencies, parameters, entry points. Run locally, remotely, on cloud.

**MLproject file:**
```yaml
# MLproject
name: yolov8_training

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "dataset.yaml"}
      epochs: {type: int, default: 100}
      img_size: {type: int, default: 640}
      batch_size: {type: int, default: 16}
      learning_rate: {type: float, default: 0.01}
    command: "python train.py --data {data_path} --epochs {epochs} --imgsz {img_size} --batch {batch_size} --lr {learning_rate}"

  inference:
    parameters:
      model_path: {type: str}
      image_path: {type: str}
    command: "python inference.py --model {model_path} --image {image_path}"
```

**Running MLflow project:**
```bash
# Run locally
mlflow run . -P epochs=50 -P learning_rate=0.001

# Run from GitHub
mlflow run https://github.com/user/ml-project -P epochs=100

# Run on remote (Kubernetes, Databricks, etc)
mlflow run . --backend kubernetes --backend-config kubernetes_config.json
```

**When to use MLflow:**
- Need experiment tracking and comparison
- Want framework-agnostic model packaging
- Require model versioning and registry
- Building ML pipelines with reproducibility
- Team collaboration on ML projects
- Gradual MLOps adoption (start simple, scale up)

---

## Comet

**ML experiment tracking platform:** Comet is cloud-based experiment tracking similar to MLflow but with more advanced features. Better visualization, collaboration tools, model comparison UI. Commercial product with free tier. Stronger focus on deep learning workflows.

Why Comet: Superior UI/UX compared to MLflow. Built-in hyperparameter optimization. Real-time collaboration features. Great for research teams. Hosted solution (no infrastructure needed). Rich visualization for comparing hundreds of experiments.

**Installation:**
```bash
pip install comet-ml
```

**Basic tracking:**
```python
"""
Comet Tracking - Object Detection
----------------------------------
"""
from comet_ml import Experiment
import torch

# Create experiment
experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="object-detection",
    workspace="your-workspace"
)

# Log parameters
experiment.log_parameters({
    "learning_rate": 0.01,
    "batch_size": 16,
    "epochs": 100,
    "model": "yolov8n",
    "img_size": 640
})

# Training loop
for epoch in range(epochs):
    # Train
    loss, mAP = train_epoch()

    # Log metrics
    experiment.log_metric("loss", loss, step=epoch)
    experiment.log_metric("mAP50", mAP, step=epoch)

# Log confusion matrix
experiment.log_confusion_matrix(
    y_true=ground_truths,
    y_predicted=predictions,
    labels=class_names,
    title="Confusion Matrix"
)

# Log image samples
experiment.log_image("predictions.png", "Sample Predictions")

# Log model
experiment.log_model("yolov8_detector", "best.pt")

# End experiment
experiment.end()
```

**Advanced features:**
```python
"""
Comet advanced features
-----------------------
"""
# Log dataset
experiment.log_dataset_info(
    name="custom_dataset",
    version="v1.2",
    path="s3://bucket/dataset"
)

# Log hyperparameter search
from comet_ml import Optimizer

config = {
    "algorithm": "bayes",
    "parameters": {
        "learning_rate": {"type": "float", "min": 0.0001, "max": 0.1},
        "batch_size": {"type": "integer", "min": 8, "max": 64}
    },
    "spec": {
        "metric": "mAP50",
        "objective": "maximize"
    }
}

opt = Optimizer(config)

for experiment in opt.get_experiments():
    # Get suggested parameters
    lr = experiment.get_parameter("learning_rate")
    bs = experiment.get_parameter("batch_size")

    # Train and log
    mAP = train_model(lr, bs)
    experiment.log_metric("mAP50", mAP)

# Compare experiments in UI
# Comet automatically creates comparison views
```

**When to use Comet:**
- Want managed solution (no server setup)
- Need advanced visualization and comparison
- Team collaboration important
- Hyperparameter optimization required
- Budget for commercial tool
- Deep learning focused workflows

---

## Weights & Biases (W&B)

**ML developer tools platform:** W&B is popular experiment tracking and visualization platform. Excellent for deep learning. Real-time dashboards, hyperparameter tuning, model versioning. Used by OpenAI, Toyota, NVIDIA. Commercial with generous free tier.

Why W&B popular: Best-in-class visualizations. Real-time experiment monitoring. Great for remote teams. Integrates with PyTorch Lightning, Hugging Face, fastai. Powerful sweeps for hyperparameter tuning. Model registry and artifact tracking.

**Installation:**
```bash
pip install wandb

# Login
wandb login
```

**Basic tracking:**
```python
"""
Weights & Biases tracking
--------------------------
"""
import wandb

# Initialize run
run = wandb.init(
    project="object-detection",
    name="yolov8n-training",
    config={
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 100,
        "architecture": "yolov8n"
    }
)

# Training loop
for epoch in range(epochs):
    # Train
    train_loss = train_epoch()
    val_mAP = validate()

    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/mAP50": val_mAP,
        "val/mAP50-95": mAP50_95
    })

# Log images with bounding boxes
images = []
for img, pred_boxes, gt_boxes in zip(sample_images, predictions, ground_truths):
    images.append(wandb.Image(
        img,
        boxes={
            "predictions": {
                "box_data": pred_boxes,
                "class_labels": class_names
            },
            "ground_truth": {
                "box_data": gt_boxes,
                "class_labels": class_names
            }
        }
    ))

wandb.log({"predictions": images})

# Log model
wandb.save("best_model.pt")

# Finish run
wandb.finish()
```

**Hyperparameter sweeps:**
```yaml
# sweep.yaml
program: train.py
method: bayes
metric:
  name: val/mAP50
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [8, 16, 32, 64]
  epochs:
    value: 100
```

```python
"""
Run hyperparameter sweep
------------------------
"""
import wandb

def train():
    """Training function for sweep"""
    run = wandb.init()

    # Get config from sweep
    config = wandb.config

    # Train with sweep parameters
    model = train_model(
        lr=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # Log results
    wandb.log({"val/mAP50": mAP})

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="object-detection")

# Run sweep (multiple agents for parallel execution)
wandb.agent(sweep_id, function=train, count=50)
```

**When to use W&B:**
- Need real-time experiment monitoring
- Want beautiful visualizations
- Team collaboration critical
- Hyperparameter tuning important
- Deep learning workflows
- Remote team with cloud requirements

---

## Apache Airflow

**Workflow orchestration platform:** Airflow schedules and monitors workflows (DAGs - Directed Acyclic Graphs). Originally for data engineering, now used for ML pipelines. Schedule training jobs, automate retraining, orchestrate multi-step ML workflows.

Why Airflow for ML: Need to retrain model weekly with new data? Airflow schedules it. Complex pipeline with data collection → preprocessing → training → evaluation → deployment? Airflow orchestrates all steps. Handles failures, retries, monitoring.

**Core concepts:**

**DAG (Directed Acyclic Graph)**: Workflow definition. Collection of tasks with dependencies.

**Task**: Single unit of work. Python function, bash command, Docker container.

**Operator**: Template for task. PythonOperator, BashOperator, DockerOperator, KubernetesPodOperator.

**Scheduler**: Triggers DAG runs based on schedule.

**Installation:**
```bash
pip install apache-airflow

# Initialize database
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver and scheduler
airflow webserver --port 8080
airflow scheduler
```

**ML training pipeline DAG:**
```python
"""
Airflow DAG - ML Training Pipeline
-----------------------------------
Daily model retraining workflow
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'yolov8_training_pipeline',
    default_args=default_args,
    description='Daily YOLOv8 retraining pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False
)

def collect_new_data(**context):
    """Collect new training data from S3"""
    import boto3

    s3 = boto3.client('s3')
    # Download new images
    # Save to local directory
    print("New data collected")

def preprocess_data(**context):
    """Preprocess and augment data"""
    # Load images
    # Apply augmentation
    # Split train/val/test
    print("Data preprocessed")

def train_model(**context):
    """Train YOLOv8 model"""
    from ultralytics import YOLO
    import mlflow

    mlflow.set_tracking_uri("http://mlflow-server:5000")

    with mlflow.start_run():
        model = YOLO('yolov8n.pt')
        results = model.train(
            data='dataset.yaml',
            epochs=100,
            imgsz=640
        )

        # Log to MLflow
        mlflow.log_metrics({
            "mAP50": results.results_dict['metrics/mAP50(B)'],
            "mAP50-95": results.results_dict['metrics/mAP50-95(B)']
        })

        # Push run_id to XCom for downstream tasks
        context['ti'].xcom_push(key='mlflow_run_id', value=mlflow.active_run().info.run_id)

    print("Model trained")

def evaluate_model(**context):
    """Evaluate model on test set"""
    # Load model
    # Run inference on test set
    # Calculate metrics
    # Compare with production model

    mAP = 0.92

    # Push result to XCom
    context['ti'].xcom_push(key='test_mAP', value=mAP)

    if mAP < 0.85:
        raise ValueError(f"Model performance too low: {mAP}")

    print(f"Model evaluated: mAP={mAP}")

def deploy_model(**context):
    """Deploy model to production"""
    import mlflow
    from mlflow.tracking import MlflowClient

    # Get MLflow run ID from XCom
    run_id = context['ti'].xcom_pull(key='mlflow_run_id')
    test_mAP = context['ti'].xcom_pull(key='test_mAP')

    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "yolov8_detector")

    # Transition to production
    client.transition_model_version_stage(
        name="yolov8_detector",
        version=result.version,
        stage="Production"
    )

    print(f"Model v{result.version} deployed to production")

def send_notification(**context):
    """Send deployment notification"""
    # Send Slack/email notification
    print("Deployment notification sent")

# Define tasks
collect_data_task = PythonOperator(
    task_id='collect_new_data',
    python_callable=collect_new_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

# Define dependencies
collect_data_task >> preprocess_task >> train_task >> evaluate_task >> deploy_task >> notify_task
```

**Dynamic DAGs with sensors:**
```python
"""
Airflow Sensor - Wait for new data
-----------------------------------
"""
from airflow.sensors.filesystem import FileSensor

# Wait for new data file
wait_for_data = FileSensor(
    task_id='wait_for_new_data',
    filepath='/data/new_images/',
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Timeout after 1 hour
    dag=dag
)

wait_for_data >> collect_data_task
```

**When to use Airflow:**
- Need scheduled model retraining
- Complex multi-step ML workflows
- Data pipeline + ML pipeline orchestration
- Team already uses Airflow for data engineering
- Want monitoring, alerting, retry logic
- Require workflow versioning and auditing

---

## Kubeflow

**ML toolkit for Kubernetes:** Kubeflow makes ML workflows portable and scalable on Kubernetes. End-to-end ML platform from notebooks to production. Built by Google, now CNCF project. Perfect for teams already on Kubernetes.

Why Kubeflow: Run ML workloads on Kubernetes cluster. Portable across cloud providers. Scales training to 100s of GPUs. Built-in experiment tracking, hyperparameter tuning, pipelines, serving. Complete ML platform.

**Components:**

**Jupyter Notebooks**: Multi-user notebook servers on Kubernetes

**Kubeflow Pipelines**: ML workflow orchestration (like Airflow but Kubernetes-native)

**Katib**: Hyperparameter tuning and neural architecture search

**KServe**: Model serving on Kubernetes

**Training Operators**: Distributed training for TensorFlow, PyTorch, XGBoost

**Installation:**
```bash
# Install Kubeflow (requires Kubernetes cluster)
kubectl apply -k "github.com/kubeflow/manifests/example"

# Access UI
kubectl port-forward -n kubeflow svc/istio-ingressgateway 8080:80
```

**Kubeflow Pipeline example:**
```python
"""
Kubeflow Pipeline - Training Pipeline
--------------------------------------
"""
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

def download_data(data_path: str) -> str:
    """Download training data"""
    import boto3

    s3 = boto3.client('s3')
    # Download data
    return "/data/downloaded"

def preprocess_data(input_path: str) -> str:
    """Preprocess data"""
    # Preprocessing logic
    return "/data/processed"

def train_model(data_path: str, epochs: int, learning_rate: float) -> str:
    """Train model"""
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=f'{data_path}/dataset.yaml',
        epochs=epochs,
        lr0=learning_rate
    )

    return "/models/best.pt"

def evaluate_model(model_path: str) -> float:
    """Evaluate model"""
    # Evaluation logic
    mAP = 0.92
    return mAP

def deploy_model(model_path: str, mAP: float):
    """Deploy if mAP threshold met"""
    if mAP >= 0.85:
        # Deploy model
        print(f"Deploying model with mAP={mAP}")
    else:
        raise ValueError(f"Model performance too low: {mAP}")

# Create components
download_op = create_component_from_func(download_data)
preprocess_op = create_component_from_func(preprocess_data)
train_op = create_component_from_func(train_model, base_image='pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime')
evaluate_op = create_component_from_func(evaluate_model)
deploy_op = create_component_from_func(deploy_model)

# Define pipeline
@dsl.pipeline(
    name='YOLOv8 Training Pipeline',
    description='End-to-end YOLOv8 training and deployment'
)
def yolov8_training_pipeline(
    data_path: str = 's3://bucket/data',
    epochs: int = 100,
    learning_rate: float = 0.01
):
    """Pipeline definition"""

    # Download data
    download_task = download_op(data_path)

    # Preprocess
    preprocess_task = preprocess_op(download_task.output)

    # Train
    train_task = train_op(
        preprocess_task.output,
        epochs,
        learning_rate
    )

    # Evaluate
    evaluate_task = evaluate_op(train_task.output)

    # Deploy
    deploy_task = deploy_op(
        train_task.output,
        evaluate_task.output
    )

# Compile pipeline
kfp.compiler.Compiler().compile(
    yolov8_training_pipeline,
    'yolov8_pipeline.yaml'
)

# Run pipeline
client = kfp.Client(host='http://localhost:8080')
run = client.create_run_from_pipeline_func(
    yolov8_training_pipeline,
    arguments={
        'epochs': 100,
        'learning_rate': 0.01
    }
)
```

**Distributed training with PyTorch operator:**
```yaml
# pytorch-training.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: yolov8-distributed-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
            command: [python, train.py]
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
            command: [python, train.py]
            resources:
              limits:
                nvidia.com/gpu: 1
```

**Hyperparameter tuning with Katib:**
```yaml
# katib-experiment.yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: yolov8-hpo
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: mAP50
  algorithm:
    algorithmName: bayesianoptimization
  parallelTrialCount: 3
  maxTrialCount: 20
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.0001"
        max: "0.1"
    - name: batch_size
      parameterType: int
      feasibleSpace:
        min: "8"
        max: "64"
  trialTemplate:
    primaryContainerName: training-container
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training-container
                image: your-training-image
                command:
                  - python
                  - train.py
                  - --lr=${trialParameters.learningRate}
                  - --batch-size=${trialParameters.batchSize}
```

**When to use Kubeflow:**
- Running on Kubernetes
- Need distributed training at scale
- Want complete ML platform
- Team comfortable with Kubernetes
- Multi-cloud portability required
- Enterprise ML workflows

---

## DVC (Data Version Control)

**Git for data and models:** DVC versions large files (datasets, models) that Git can't handle. Works alongside Git. Track dataset changes, reproduce experiments, share data efficiently. Essential for ML reproducibility.

Why DVC critical: Git tracks code. DVC tracks data. Together they provide complete versioning. Can checkout any commit and reproduce exact experiment with correct code AND data versions. Share multi-GB datasets across team without bloating Git repository.

**Installation:**
```bash
pip install dvc
dvc init
```

**Track dataset:**
```bash
# Add dataset to DVC
dvc add data/train_images.zip

# DVC creates .dvc file (tracked by Git)
git add data/train_images.zip.dvc data/.gitignore
git commit -m "Add training dataset v1"

# Push data to remote storage (S3, Google Cloud, Azure, SSH)
dvc remote add -d storage s3://mybucket/dvc-storage
dvc push

# Team member pulls data
git pull
dvc pull
```

**Pipeline with DVC:**
```yaml
# dvc.yaml - Define ML pipeline
stages:
  prepare:
    cmd: python prepare_data.py
    deps:
      - data/raw
    params:
      - prepare.train_split
      - prepare.val_split
    outs:
      - data/prepared

  train:
    cmd: python train.py
    deps:
      - data/prepared
      - train.py
    params:
      - train.epochs
      - train.learning_rate
      - train.batch_size
    outs:
      - models/model.pt
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python evaluate.py
    deps:
      - data/prepared
      - models/model.pt
    metrics:
      - scores.json:
          cache: false
```

**Parameters file:**
```yaml
# params.yaml
prepare:
  train_split: 0.8
  val_split: 0.1

train:
  epochs: 100
  learning_rate: 0.01
  batch_size: 16
  img_size: 640
```

**Running pipeline:**
```bash
# Run entire pipeline
dvc repro

# DVC automatically:
# - Checks which stages changed
# - Only runs necessary stages
# - Caches outputs
# - Tracks metrics

# Compare experiments
dvc metrics show
dvc metrics diff
```

**Experiment tracking:**
```bash
# List experiments
dvc exp show

# Compare experiments
dvc exp diff

# Apply best experiment
dvc exp apply best-exp-id
```

**When to use DVC:**
- Need data versioning
- Want reproducible experiments
- Large datasets (> 100 MB)
- Team sharing data/models
- Git-like workflow for ML
- Cost-effective storage (use own S3/GCS)

---

## Metaflow

**Python library for data science workflows:** Metaflow by Netflix simplifies building and deploying data science projects. Focus on code, Metaflow handles infrastructure. Version code/data/artifacts automatically. Deploy to AWS seamlessly.

Why Metaflow: Extremely simple Python API. No YAML, no configuration files. Production-ready out of box. Netflix battle-tested at scale. AWS integration built-in. Great for data scientists who want to focus on models not infrastructure.

**Installation:**
```bash
pip install metaflow
```

**Basic flow:**
```python
"""
Metaflow - Training Flow
------------------------
"""
from metaflow import FlowSpec, step, Parameter

class YOLOv8TrainingFlow(FlowSpec):
    """
    YOLOv8 training workflow with Metaflow.

    Metaflow automatically:
    - Versions code
    - Versions data
    - Tracks artifacts
    - Handles retries
    - Enables branching
    """

    # Parameters
    epochs = Parameter('epochs', default=100)
    learning_rate = Parameter('learning_rate', default=0.01)
    batch_size = Parameter('batch_size', default=16)

    @step
    def start(self):
        """Initialize flow"""
        print("Starting YOLOv8 training flow")
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare dataset"""
        import os

        # Download/prepare data
        self.data_path = "/data/prepared"
        self.num_samples = 10000

        print(f"Data prepared: {self.num_samples} samples")
        self.next(self.train)

    @step
    def train(self):
        """Train model"""
        from ultralytics import YOLO

        print(f"Training with lr={self.learning_rate}, batch={self.batch_size}, epochs={self.epochs}")

        model = YOLO('yolov8n.pt')
        results = model.train(
            data=f'{self.data_path}/dataset.yaml',
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate
        )

        # Metaflow automatically versions these artifacts
        self.model_path = "runs/train/weights/best.pt"
        self.mAP50 = results.results_dict['metrics/mAP50(B)']
        self.mAP50_95 = results.results_dict['metrics/mAP50-95(B)']

        print(f"Training complete: mAP50={self.mAP50:.4f}")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate model"""
        # Load model and run evaluation
        self.test_mAP = 0.92
        self.test_precision = 0.89
        self.test_recall = 0.91

        print(f"Evaluation complete: test_mAP={self.test_mAP:.4f}")

        # Branch based on performance
        if self.test_mAP >= 0.85:
            self.deploy_approved = True
            self.next(self.deploy)
        else:
            self.deploy_approved = False
            self.next(self.end)

    @step
    def deploy(self):
        """Deploy model"""
        print(f"Deploying model with mAP={self.test_mAP}")

        # Deploy logic here
        self.deployment_url = "https://api.example.com/v1/detect"

        self.next(self.end)

    @step
    def end(self):
        """Flow complete"""
        if self.deploy_approved:
            print(f"✓ Model deployed: {self.deployment_url}")
        else:
            print(f"✗ Model not deployed (mAP too low: {self.test_mAP})")

if __name__ == '__main__':
    YOLOv8TrainingFlow()
```

**Running flow:**
```bash
# Run locally
python train_flow.py run --epochs 50 --learning_rate 0.001

# Run on AWS Batch
python train_flow.py run --with batch --epochs 100
```

**Inspecting results:**
```python
"""
Inspect Metaflow runs
----------------------
"""
from metaflow import Flow, Run

# Get latest run
run = Flow('YOLOv8TrainingFlow').latest_run

# Access artifacts
print(f"mAP50: {run.data.mAP50}")
print(f"Model path: {run.data.model_path}")

# Get specific run
run = Run('YOLOv8TrainingFlow/123')
model_path = run.data.model_path

# Compare runs
runs = Flow('YOLOv8TrainingFlow').runs()
for run in runs:
    print(f"{run.id}: mAP50={run.data.mAP50:.4f}")
```

**Parallel hyperparameter search:**
```python
"""
Metaflow - Hyperparameter search
---------------------------------
"""
from metaflow import FlowSpec, step

class HPOFlow(FlowSpec):

    @step
    def start(self):
        """Generate parameter combinations"""
        self.configs = [
            {'lr': 0.001, 'bs': 16},
            {'lr': 0.01, 'bs': 16},
            {'lr': 0.001, 'bs': 32},
            {'lr': 0.01, 'bs': 32},
        ]
        self.next(self.train, foreach='configs')

    @step
    def train(self):
        """Train with different configs - runs in parallel"""
        config = self.input
        # Train with config
        self.mAP = train_model(config['lr'], config['bs'])
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join parallel branches"""
        # Find best configuration
        best = max(inputs, key=lambda x: x.mAP)
        self.best_config = best.input
        self.best_mAP = best.mAP
        self.next(self.end)

    @step
    def end(self):
        print(f"Best config: {self.best_config}, mAP={self.best_mAP}")

if __name__ == '__main__':
    HPOFlow()
```

**When to use Metaflow:**
- Python-first workflow
- Want simplicity over features
- AWS-centric infrastructure
- Data science team (not DevOps heavy)
- Need automatic versioning
- Prefer code over configuration

---

## Pachyderm

**Data versioning and pipelines:** Pachyderm is Git for data plus Docker-based pipelines. Version data like code, build containerized pipelines. Automatic versioning, provenance tracking, data lineage. Kubernetes-native.

Why Pachyderm: Built for data science. Every data commit triggers pipeline. Complete data lineage from raw data to model. Roll back to any data version. Parallel processing automatically. Perfect for reproducibility.

**Installation:**
```bash
# Install pachctl CLI
brew tap pachyderm/tap && brew install pachyderm/tap/pachctl@2.7

# Deploy on Kubernetes
pachctl deploy local

# Or use cloud deployment
```

**Create repository:**
```bash
# Create data repo
pachctl create repo training_images

# Put data
pachctl put file training_images@master:/images -f ./images/

# List commits
pachctl list commit training_images

# Inspect data
pachctl list file training_images@master
```

**Pipeline specification:**
```json
{
  "pipeline": {
    "name": "yolov8_training"
  },
  "description": "Train YOLOv8 model",
  "input": {
    "pfs": {
      "repo": "training_images",
      "glob": "/*"
    }
  },
  "transform": {
    "cmd": ["python", "/app/train.py"],
    "image": "your-training-image:latest",
    "env": {
      "EPOCHS": "100",
      "LEARNING_RATE": "0.01"
    }
  },
  "resource_requests": {
    "memory": "8G",
    "cpu": 4,
    "gpu": {
      "type": "nvidia.com/gpu",
      "number": 1
    }
  }
}
```

```bash
# Create pipeline
pachctl create pipeline -f training_pipeline.json

# Pipeline automatically runs when new data added
pachctl put file training_images@master:/images/new_batch -f ./new_images/

# Monitor pipeline
pachctl list job

# Get outputs
pachctl get file yolov8_training@master:/model.pt -o ./model.pt
```

**Data provenance:**
```bash
# Track data lineage
pachctl inspect commit yolov8_training@master

# See which data version produced which model
pachctl list commit yolov8_training

# Reproduce experiment with exact data version
pachctl create pipeline -f training_pipeline.json --reprocess
```

**When to use Pachyderm:**
- Need complete data lineage
- Want Git-like versioning for data
- Containerized ML workflows
- Kubernetes infrastructure
- Complex data dependencies
- Compliance/audit requirements

---

## ClearML

**ML experiment management platform:** ClearML (formerly Allegro Trains) provides experiment tracking, orchestration, data management. Open source with enterprise features. AutoML capabilities, remote execution, model serving.

**Installation:**
```bash
pip install clearml

# Configure
clearml-init
```

**Basic usage:**
```python
"""
ClearML - Auto-tracking
-----------------------
"""
from clearml import Task

# Initialize task (automatically tracks everything)
task = Task.init(
    project_name='ObjectDetection',
    task_name='YOLOv8 Training'
)

# ClearML automatically logs:
# - Git info
# - Installed packages
# - Command line arguments
# - Console output
# - Tensorboard
# - Model checkpoints

# Train model (automatically tracked)
model = YOLO('yolov8n.pt')
results = model.train(data='dataset.yaml', epochs=100)

# Manually log additional info
task.upload_artifact('best_model', artifact_object='best.pt')
```

**When to use ClearML:**
- Want comprehensive tracking without code changes
- Need experiment comparison and collaboration
- Require remote execution
- Want integrated solution (tracking + orchestration + serving)
- Open source with enterprise option

---

## Neptune.ai

**Metadata store for ML:** Neptune tracks all ML metadata - experiments, models, datasets, code. Focused on team collaboration and large-scale experiment management. Strong visualization and comparison features. Commercial with generous free tier.

Why Neptune: Handles thousands of experiments easily. Superior experiment comparison UI. Great for research teams running many experiments. Lightweight integration. Good for organizations needing audit trails and governance.

**Installation:**
```bash
pip install neptune
```

**Basic tracking:**
```python
"""
Neptune - Experiment tracking
------------------------------
"""
import neptune

# Initialize run
run = neptune.init_run(
    project="workspace/object-detection",
    api_token="YOUR_API_TOKEN",
    tags=["yolov8", "baseline"]
)

# Log parameters
params = {
    "learning_rate": 0.01,
    "batch_size": 16,
    "epochs": 100,
    "model": "yolov8n"
}
run["parameters"] = params

# Training loop
for epoch in range(epochs):
    # Train
    train_loss = train_epoch()
    val_mAP = validate()

    # Log metrics
    run["train/loss"].append(train_loss)
    run["val/mAP50"].append(val_mAP)

# Log files
run["model/weights"].upload("best.pt")
run["predictions"].upload("predictions.png")

# Log metadata
run["sys/tags"].add(["production-ready"])
run["dataset/version"] = "v2.1"

# Stop run
run.stop()
```

**Advanced features:**
```python
"""
Neptune - Compare experiments
------------------------------
"""
import neptune

# Fetch project
project = neptune.init_project(
    project="workspace/object-detection",
    api_token="YOUR_API_TOKEN"
)

# Query runs
runs_table = project.fetch_runs_table(
    state="active",
    tag="yolov8"
).to_pandas()

# Filter best runs
best_runs = runs_table[runs_table["val/mAP50"] > 0.9]

# Download artifacts from best run
best_run_id = best_runs.iloc[0]["sys/id"]
best_run = neptune.init_run(
    with_id=best_run_id,
    project="workspace/object-detection"
)
best_run["model/weights"].download("best_model.pt")
```

**When to use Neptune:**
- Need powerful experiment comparison
- Running thousands of experiments
- Team collaboration critical
- Want hosted solution
- Require audit trails and governance
- Budget for commercial tool

---

## TensorBoard

**Visualization toolkit:** TensorBoard is TensorFlow's visualization tool but works with PyTorch too. Visualize training metrics, model graphs, embeddings, images. Free and open source. Standard for deep learning visualization.

Why TensorBoard: Simple, works everywhere. Real-time training visualization. Hyperparameter comparison. Embedding projector for high-dimensional data. No account needed, runs locally.

**Installation:**
```bash
pip install tensorboard
```

**PyTorch integration:**
```python
"""
TensorBoard with PyTorch
------------------------
"""
from torch.utils.tensorboard import SummaryWriter
import torch

# Create writer
writer = SummaryWriter('runs/yolov8_experiment_1')

# Log hyperparameters
writer.add_hparams(
    {'lr': 0.01, 'batch_size': 16},
    {'mAP50': 0, 'loss': 0}  # Will be filled later
)

# Training loop
for epoch in range(epochs):
    for i, (images, targets) in enumerate(train_loader):
        # Train
        loss = train_step(images, targets)

        # Log scalar
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/train', loss, global_step)

    # Validation
    val_loss, val_mAP = validate()
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('mAP50/val', val_mAP, epoch)

    # Log images with predictions
    writer.add_images('predictions', pred_images, epoch)

# Log model graph
dummy_input = torch.randn(1, 3, 640, 640)
writer.add_graph(model, dummy_input)

# Close writer
writer.close()
```

**Launch TensorBoard:**
```bash
# Start TensorBoard server
tensorboard --logdir=runs --port=6006

# Open browser to http://localhost:6006
```

**Compare multiple experiments:**
```bash
# Compare multiple runs
tensorboard --logdir=runs --port=6006

# Directory structure:
# runs/
#   experiment_1/
#   experiment_2/
#   experiment_3/
# TensorBoard automatically compares all experiments
```

**When to use TensorBoard:**
- Need quick visualization
- Training PyTorch/TensorFlow models
- Want local solution (no cloud account)
- Real-time monitoring during training
- Free and simple

---

## Prefect

**Modern workflow orchestration:** Prefect is next-generation Airflow. Python-native, easier to use, better error handling. Dynamic workflows, parametric runs. Hybrid execution model (code runs anywhere, orchestration in cloud).

Why Prefect over Airflow: Pure Python (no DAG syntax), easier debugging, better UI, parametric flows, hybrid architecture. Modern development experience. Growing adoption.

**Installation:**
```bash
pip install prefect

# Start Prefect server (or use cloud)
prefect server start
```

**Basic flow:**
```python
"""
Prefect - ML Training Flow
---------------------------
"""
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def download_data(data_url: str) -> str:
    """Download training data"""
    import boto3

    # Download data
    local_path = "/data/downloaded"
    return local_path

@task
def preprocess_data(data_path: str) -> str:
    """Preprocess dataset"""
    # Preprocessing logic
    processed_path = "/data/processed"
    return processed_path

@task(retries=3, retry_delay_seconds=60)
def train_model(data_path: str, learning_rate: float, epochs: int) -> dict:
    """Train YOLOv8 model"""
    from ultralytics import YOLO
    import mlflow

    mlflow.set_tracking_uri("http://mlflow-server:5000")

    with mlflow.start_run():
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=f'{data_path}/dataset.yaml',
            epochs=epochs,
            lr0=learning_rate
        )

        metrics = {
            'mAP50': results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': results.results_dict['metrics/mAP50-95(B)']
        }

        mlflow.log_metrics(metrics)

        return {
            'model_path': 'runs/train/weights/best.pt',
            'metrics': metrics,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }

@task
def evaluate_model(model_path: str) -> float:
    """Evaluate on test set"""
    # Evaluation logic
    test_mAP = 0.92
    return test_mAP

@task
def deploy_model(model_path: str, mlflow_run_id: str, test_mAP: float):
    """Deploy if performance threshold met"""
    if test_mAP >= 0.85:
        # Deploy logic
        print(f"Deploying model with mAP={test_mAP}")
        # Register in MLflow, update production, etc.
    else:
        raise ValueError(f"Model performance below threshold: {test_mAP}")

@flow(name="yolov8-training-pipeline")
def yolov8_training_pipeline(
    data_url: str = "s3://bucket/data",
    learning_rate: float = 0.01,
    epochs: int = 100
):
    """
    End-to-end YOLOv8 training pipeline.

    Prefect automatically:
    - Handles retries
    - Caches task results
    - Tracks state
    - Provides UI
    """
    # Download and preprocess
    data_path = download_data(data_url)
    processed_path = preprocess_data(data_path)

    # Train
    train_result = train_model(processed_path, learning_rate, epochs)

    # Evaluate
    test_mAP = evaluate_model(train_result['model_path'])

    # Deploy
    deploy_model(
        train_result['model_path'],
        train_result['mlflow_run_id'],
        test_mAP
    )

    return train_result

if __name__ == "__main__":
    # Run flow
    yolov8_training_pipeline(learning_rate=0.01, epochs=100)
```

**Scheduling:**
```python
"""
Schedule Prefect flow
----------------------
"""
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=yolov8_training_pipeline,
    name="daily-training",
    schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
    parameters={"learning_rate": 0.01, "epochs": 100},
    work_queue_name="ml-training"
)

deployment.apply()
```

**When to use Prefect:**
- Want modern Python workflow tool
- Need dynamic pipelines
- Prefer code over YAML/config
- Easier than Airflow
- Hybrid cloud/local execution
- Better developer experience

---

## ZenML

**MLOps framework:** ZenML provides abstractions for building portable ML pipelines. Framework-agnostic, runs anywhere (local, cloud, Kubernetes). Focuses on reproducibility and production readiness. Open source.

Why ZenML: Standardizes ML workflows. Write once, run anywhere. Built-in integrations with MLflow, W&B, Kubeflow, SageMaker. Easier than building MLOps from scratch. Good for teams building ML platform.

**Installation:**
```bash
pip install zenml

# Initialize
zenml init
```

**Basic pipeline:**
```python
"""
ZenML - Training Pipeline
--------------------------
"""
from zenml import pipeline, step
from zenml.config import DockerSettings

@step
def load_data() -> dict:
    """Load training data"""
    import pandas as pd

    # Load data
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

@step
def train_model(data: dict, learning_rate: float) -> str:
    """Train model"""
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        lr0=learning_rate
    )

    model_path = 'runs/train/weights/best.pt'
    return model_path

@step
def evaluate_model(model_path: str, test_data: dict) -> float:
    """Evaluate model"""
    # Evaluation logic
    mAP = 0.92
    return mAP

@step
def deploy_model(model_path: str, mAP: float):
    """Deploy model if threshold met"""
    if mAP >= 0.85:
        # Deploy
        print(f"Deploying model with mAP={mAP}")

@pipeline
def training_pipeline(learning_rate: float = 0.01):
    """End-to-end training pipeline"""
    data = load_data()
    model_path = train_model(data, learning_rate)
    mAP = evaluate_model(model_path, data['test'])
    deploy_model(model_path, mAP)

if __name__ == "__main__":
    # Run pipeline
    training_pipeline(learning_rate=0.01)
```

**Run on different stacks:**
```bash
# Run locally
zenml stack set local
python train_pipeline.py

# Run on Kubernetes
zenml stack set kubernetes
python train_pipeline.py

# Same code, different infrastructure
```

**When to use ZenML:**
- Building ML platform
- Want infrastructure portability
- Need standardized pipelines
- Team using multiple cloud providers
- Framework-agnostic approach

---

## BentoML

**Model serving framework:** BentoML packages models as production-ready API services. Supports all ML frameworks. Generates Docker images, Kubernetes configs automatically. Focus on model deployment and serving.

Why BentoML: Simplest way to deploy models as APIs. Automatic optimization, batching, adaptive scaling. Generate deployment artifacts with one command. Great for data scientists who need to deploy without DevOps.

**Installation:**
```bash
pip install bentoml
```

**Create service:**
```python
"""
BentoML - Model Serving
------------------------
"""
import bentoml
from bentoml.io import Image, JSON
import numpy as np
import cv2

# Save model to BentoML store
bentoml.onnx.save_model(
    "yolov8_detector",
    "model.onnx",
    signatures={"run": {"batchable": True}},
    metadata={"model": "yolov8n", "framework": "onnx"}
)

# Create service
@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 10}
)
class YOLOv8Detector:
    """YOLOv8 object detection service"""

    def __init__(self):
        # Load model
        self.model = bentoml.onnx.get("yolov8_detector:latest")
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.model.path)

    @bentoml.api
    def detect(self, image: Image) -> JSON:
        """
        Detect objects in image.

        Args:
            image: Input image

        Returns:
            Detections with bounding boxes
        """
        # Convert PIL image to numpy
        img = np.array(image)

        # Preprocess
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self.session.run(None, {'images': img})

        # Postprocess
        detections = self.postprocess(outputs)

        return {"detections": detections}

    def postprocess(self, outputs):
        """Extract detections from model output"""
        # NMS and filtering logic
        return []
```

**Build and deploy:**
```bash
# Build Bento
bentoml build

# Containerize
bentoml containerize yolov8_detector:latest

# Run locally
bentoml serve service:YOLOv8Detector

# Deploy to cloud
bentoml deploy yolov8_detector:latest --platform kubernetes

# Test API
curl -X POST http://localhost:3000/detect \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

**When to use BentoML:**
- Need to deploy models as APIs
- Want automatic optimization
- Prefer Python over DevOps
- Multiple ML frameworks
- Need batch inference support
- Kubernetes deployment

---

## Seldon Core

**ML deployment on Kubernetes:** Seldon Core deploys ML models on Kubernetes at scale. Supports all frameworks, A/B testing, canary rollouts, explainability. Production-grade model serving.

Why Seldon Core: Enterprise-ready serving. Advanced deployment patterns (A/B, multi-armed bandits, canary). Monitoring, explainability built-in. Best for Kubernetes-first organizations.

**Installation:**
```bash
# Install Seldon Core on Kubernetes
kubectl apply -f https://github.com/SeldonIO/seldon-core/releases/download/v1.15.0/seldon-core.yaml
```

**Model deployment:**
```yaml
# seldon-deployment.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: yolov8-detector
spec:
  predictors:
  - name: default
    replicas: 3
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: your-registry/yolov8-detector:latest
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              nvidia.com/gpu: "1"
    graph:
      name: classifier
      type: MODEL
      parameters:
      - name: model_uri
        value: "s3://models/yolov8/model.onnx"
      children: []
```

```bash
# Deploy model
kubectl apply -f seldon-deployment.yaml

# Test prediction
curl -X POST http://yolov8-detector/api/v1.0/predictions \
  -H 'Content-Type: application/json' \
  -d '{"data": {"ndarray": [[...]]}}'
```

**A/B testing:**
```yaml
# A/B test two model versions
spec:
  predictors:
  - name: model-v1
    replicas: 2
    traffic: 50
    graph:
      name: classifier
      modelUri: s3://models/v1/model.onnx
  - name: model-v2
    replicas: 2
    traffic: 50
    graph:
      name: classifier
      modelUri: s3://models/v2/model.onnx
```

**When to use Seldon Core:**
- Deploying on Kubernetes
- Need A/B testing, canary deployments
- Want advanced serving patterns
- Enterprise production requirements
- Multiple models, complex routing
- Monitoring and explainability critical

---

## Evidently AI

**ML monitoring and data drift detection:** Evidently monitors ML models in production. Detects data drift, model degradation, data quality issues. Generates reports and dashboards. Critical for maintaining model performance.

Why Evidently: Models degrade over time. Evidently detects when retraining needed. Monitors data distribution changes. Open source with enterprise option. Easy integration.

**Installation:**
```bash
pip install evidently
```

**Data drift detection:**
```python
"""
Evidently - Data Drift Detection
---------------------------------
"""
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd

# Reference data (training set)
reference_data = pd.read_csv('training_data.csv')

# Current production data
current_data = pd.read_csv('production_data.csv')

# Generate drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report
report.save_html('drift_report.html')

# Get drift score
drift_share = report.as_dict()['metrics'][0]['result']['drift_share']
print(f"Drift detected in {drift_share:.1%} of features")
```

**Model performance monitoring:**
```python
"""
Monitor model predictions
-------------------------
"""
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

# Predictions and ground truth
predictions_df = pd.DataFrame({
    'prediction': predictions,
    'actual': ground_truths,
    'confidence': confidences
})

# Generate performance report
report = Report(metrics=[ClassificationPreset()])

report.run(
    reference_data=train_predictions,
    current_data=predictions_df
)

report.save_html('performance_report.html')
```

**Real-time monitoring:**
```python
"""
Production monitoring pipeline
-------------------------------
"""
from evidently.test_suite import TestSuite
from evidently.tests import TestPrecisionScore, TestRecallScore

# Define tests
test_suite = TestSuite(tests=[
    TestPrecisionScore(gte=0.85),  # Precision >= 85%
    TestRecallScore(gte=0.80)      # Recall >= 80%
])

# Run tests on production data
test_suite.run(
    reference_data=None,
    current_data=predictions_df
)

# Get results
results = test_suite.as_dict()

# Alert if tests fail
if not results['summary']['all_passed']:
    send_alert("Model performance degraded!")
```

**When to use Evidently:**
- Monitor models in production
- Detect data drift
- Track model performance degradation
- Need to know when to retrain
- Compliance requirements for monitoring
- Open source monitoring solution

---

## Great Expectations

**Data validation framework:** Great Expectations validates data quality. Define expectations for data, test automatically. Prevents bad data from breaking models. Critical for data quality assurance.

Why Great Expectations: Bad data = bad models. Validate data before training and in production. Catch data issues early. Generate documentation automatically. Integrates with pipelines.

**Installation:**
```bash
pip install great_expectations
```

**Setup:**
```bash
# Initialize Great Expectations
great_expectations init

# Create data source
great_expectations datasource new
```

**Define expectations:**
```python
"""
Great Expectations - Data Validation
-------------------------------------
"""
import great_expectations as gx

# Create context
context = gx.get_context()

# Load data
df = context.sources.pandas_default.read_csv("training_data.csv")

# Define expectations
df.expect_table_row_count_to_be_between(min_value=1000, max_value=1000000)
df.expect_column_values_to_not_be_null("image_path")
df.expect_column_values_to_be_in_set("label", value_set=['car', 'person', 'bike'])
df.expect_column_values_to_be_between("bbox_x", min_value=0, max_value=1)
df.expect_column_values_to_be_between("bbox_y", min_value=0, max_value=1)

# Validate
validation_result = df.validate()

if not validation_result["success"]:
    print("Data validation failed!")
    for result in validation_result["results"]:
        if not result["success"]:
            print(f"Failed: {result['expectation_config']['expectation_type']}")
```

**In ML pipeline:**
```python
"""
Validate data in training pipeline
-----------------------------------
"""
def validate_training_data(data_path: str) -> bool:
    """Validate data before training"""
    import pandas as pd

    # Load data
    df = pd.read_csv(data_path)

    # Run validation
    suite = context.get_expectation_suite("training_data_suite")
    validation_result = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[df],
        run_id="training_validation"
    )

    if not validation_result["success"]:
        raise ValueError("Data validation failed - cannot train model")

    return True

# Use in pipeline
if validate_training_data("data/train.csv"):
    train_model()
```

**When to use Great Expectations:**
- Need data quality validation
- Prevent bad data from reaching models
- Compliance requirements
- Data pipeline testing
- Generate data documentation
- Catch data issues early

---

## Feast

**Feature store:** Feast manages ML features. Store, serve, and share features across team. Consistent features for training and serving. Solves training-serving skew problem.

Why Feast: Features computed differently in training vs production cause bugs. Feast ensures consistency. Share features across projects. Point-in-time correctness for historical features. Critical for large ML teams.

**Installation:**
```bash
pip install feast
```

**Define features:**
```python
"""
Feast - Feature definitions
----------------------------
"""
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

# Define entity
camera = Entity(
    name="camera_id",
    value_type=ValueType.STRING,
    description="Camera identifier"
)

# Define data source
camera_stats_source = FileSource(
    path="data/camera_stats.parquet",
    event_timestamp_column="timestamp"
)

# Define feature view
camera_stats_fv = FeatureView(
    name="camera_statistics",
    entities=["camera_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="avg_detections_per_hour", dtype=ValueType.FLOAT),
        Feature(name="avg_confidence", dtype=ValueType.FLOAT),
        Feature(name="total_frames_processed", dtype=ValueType.INT64),
    ],
    source=camera_stats_source
)
```

**feature_store.yaml:**
```yaml
project: object_detection
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
```

**Use features:**
```python
"""
Get features for training
--------------------------
"""
from feast import FeatureStore
import pandas as pd

# Initialize feature store
store = FeatureStore(repo_path=".")

# Get training data
entity_df = pd.DataFrame({
    "camera_id": ["cam_001", "cam_002", "cam_003"],
    "event_timestamp": [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01")
    ]
})

# Fetch features
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "camera_statistics:avg_detections_per_hour",
        "camera_statistics:avg_confidence"
    ]
).to_df()

# Use for training
train_model(training_data)

# Get features for online serving
features = store.get_online_features(
    features=[
        "camera_statistics:avg_detections_per_hour",
        "camera_statistics:avg_confidence"
    ],
    entity_rows=[{"camera_id": "cam_001"}]
).to_dict()
```

**When to use Feast:**
- Large ML team
- Multiple models using same features
- Training-serving skew issues
- Need feature reuse across projects
- Point-in-time correctness required
- Complex feature engineering

---

## Kedro

**Data pipeline framework:** Kedro structures data science code. Opinionated project template, data catalog, pipeline abstraction. Enforces best practices. Helps transition from notebooks to production.

Why Kedro: Notebook code is messy. Kedro provides structure. Separate data, code, configuration. Easy testing, versioning, deployment. Good for teams establishing ML engineering practices.

**Installation:**
```bash
pip install kedro
```

**Create project:**
```bash
# Create new Kedro project
kedro new --starter=pandas-iris

# Project structure:
# project/
#   conf/           # Configuration
#   data/           # Data (gitignored)
#   src/            # Source code
#     pipelines/    # Pipeline code
#   notebooks/      # Jupyter notebooks
```

**Define pipeline:**
```python
"""
Kedro - ML Pipeline
-------------------
"""
from kedro.pipeline import Pipeline, node

def preprocess_data(raw_data):
    """Preprocess function"""
    # Preprocessing logic
    return processed_data

def train_model(processed_data, parameters):
    """Training function"""
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=parameters['epochs'],
        lr0=parameters['learning_rate']
    )

    return results

def evaluate_model(results):
    """Evaluation function"""
    mAP = results.results_dict['metrics/mAP50(B)']
    return mAP

# Create pipeline
def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess_data,
            inputs="raw_data",
            outputs="processed_data",
            name="preprocess_node"
        ),
        node(
            func=train_model,
            inputs=["processed_data", "parameters"],
            outputs="training_results",
            name="training_node"
        ),
        node(
            func=evaluate_model,
            inputs="training_results",
            outputs="model_metrics",
            name="evaluation_node"
        )
    ])
```

**Data catalog (conf/catalog.yml):**
```yaml
raw_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/training_data.csv

processed_data:
  type: pickle.PickleDataSet
  filepath: data/02_intermediate/processed_data.pkl

training_results:
  type: pickle.PickleDataSet
  filepath: data/06_models/training_results.pkl
```

**Parameters (conf/parameters.yml):**
```yaml
epochs: 100
learning_rate: 0.01
batch_size: 16
```

**Run pipeline:**
```bash
# Run full pipeline
kedro run

# Run specific nodes
kedro run --nodes=preprocess_node,training_node

# Visualize pipeline
kedro viz
```

**When to use Kedro:**
- Transition from notebooks to production
- Need project structure
- Want enforced best practices
- Team lacks ML engineering experience
- Standardize across projects
- Testing and reproducibility important

---

## Comparison Table

| Tool | Best For | Complexity | Deployment | Cost | Cloud/Self-hosted |
|------|----------|------------|------------|------|-------------------|
| **MLflow** | General ML tracking, model registry | Low | Flexible | Free (OSS) | Both |
| **Comet** | Deep learning, collaboration | Low | Cloud | Free tier + paid | Cloud |
| **W&B** | DL research, visualization | Low | Cloud | Free tier + paid | Cloud |
| **Neptune.ai** | Experiment comparison, metadata store | Low | Cloud | Free tier + paid | Cloud |
| **TensorBoard** | PyTorch/TF visualization | Low | Self | Free (OSS) | Self |
| **Airflow** | Workflow orchestration, scheduling | Medium | Self/Cloud | Free (OSS) | Both |
| **Prefect** | Modern Python workflows | Low | Cloud/Self | Free + paid | Both |
| **Kubeflow** | Kubernetes-native ML platform | High | Kubernetes | Free (OSS) | Self (K8s) |
| **ZenML** | Portable ML pipelines | Medium | Flexible | Free (OSS) | Both |
| **DVC** | Data versioning, reproducibility | Low | Git-based | Free (OSS) | Self (Git) |
| **Metaflow** | Python workflows, AWS | Low | AWS | Free (OSS) | AWS |
| **Pachyderm** | Data versioning, pipelines | Medium | Kubernetes | Free + Enterprise | Both |
| **Kedro** | Structured data pipelines | Medium | Flexible | Free (OSS) | Self |
| **ClearML** | Auto-tracking, orchestration | Low | Flexible | Free + Enterprise | Both |
| **BentoML** | Model serving, deployment | Low | Flexible | Free (OSS) | Both |
| **Seldon Core** | K8s model deployment, A/B testing | High | Kubernetes | Free + Enterprise | Self (K8s) |
| **Evidently AI** | Model monitoring, data drift | Low | Flexible | Free (OSS) | Both |
| **Great Expectations** | Data validation, quality | Medium | Flexible | Free (OSS) | Both |
| **Feast** | Feature store | Medium | Flexible | Free (OSS) | Both |

---

## Choosing Right MLOps Stack

**Common combinations:**

**Startup stack (minimal):**
- MLflow for experiment tracking and model registry
- Git + DVC for code and data versioning
- BentoML for model serving
- Great Expectations for data validation
- Cost: $0 (all open source)

**Growing team stack:**
- MLflow or W&B for tracking
- Prefect or Airflow for pipelines
- DVC for data versioning
- Feast for feature store
- BentoML or Docker + Kubernetes for deployment
- Evidently AI for monitoring
- Cost: $0-500/month

**Enterprise stack:**
- W&B or Neptune.ai for tracking (team collaboration)
- Kubeflow or Airflow for orchestration
- Pachyderm for data lineage
- Seldon Core for K8s deployment with A/B testing
- Feast for centralized feature store
- Evidently AI for production monitoring
- Great Expectations for data quality
- Kubernetes for infrastructure
- Cost: $1000-10000/month depending on scale

**AWS-centric stack:**
- SageMaker (AWS managed ML platform)
- Metaflow for workflows
- S3 for data storage
- AWS Batch for training jobs
- Evidently AI for model monitoring
- Cost: Pay-per-use AWS pricing

**Research lab stack:**
- W&B or Neptune.ai for tracking and collaboration
- TensorBoard for visualization
- DVC for data versioning
- Jupyter notebooks
- Git for code
- Cost: $0-200/month (free tiers usually sufficient)

**Production-focused stack:**
- MLflow for experiment tracking and registry
- ZenML or Kedro for structured pipelines
- BentoML or Seldon Core for model serving
- Evidently AI for monitoring data drift
- Great Expectations for data validation
- Feast for feature management
- Cost: $0-1000/month depending on scale

---

## Implementation Roadmap

**Phase 1: Experiment tracking (Week 1)**
- Set up MLflow or Neptune.ai
- Add TensorBoard for visualization
- Add tracking to training scripts
- Team starts logging experiments
- Compare experiment results

**Phase 2: Model registry (Week 2-3)**
- Register models in MLflow
- Define staging/production stages
- Set up BentoML for model serving
- Implement model deployment from registry

**Phase 3: Data versioning & validation (Week 4)**
- Initialize DVC
- Version datasets
- Track data alongside code
- Add Great Expectations for data validation

**Phase 4: Pipeline automation (Month 2)**
- Set up Prefect or Airflow
- Consider Kedro for structured pipelines
- Automate retraining workflow
- Schedule periodic model updates

**Phase 5: Feature management (Month 2-3)**
- Set up Feast feature store
- Centralize feature definitions
- Ensure consistency across training/serving

**Phase 6: Monitoring (Month 3)**
- Add Evidently AI for model monitoring
- Track prediction distribution
- Monitor data drift
- Alert on performance degradation

**Phase 7: Advanced deployment (Month 4)**
- Move to Seldon Core if using K8s
- Implement A/B testing
- Set up canary deployments
- Gradual rollout mechanisms

**Phase 8: Continuous training (Month 4+)**
- Automated retraining triggers
- Use ZenML for portable pipelines
- Full CI/CD integration
- Automated model promotion

---

This guide covers essential MLOps tools. Start simple with MLflow + DVC, add complexity as needed. Focus on problems you actually have, not tools that sound cool. Every tool adds complexity - only adopt when benefit outweighs cost.
