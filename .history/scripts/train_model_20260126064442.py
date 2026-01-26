from src.pipeline.train import run_pipeline
from src.pipeline.evaluate import evaluate_model
from src.save_artifact import save_model, save_metrics, save_config
from src.config import MODEL_CONFIG_V2
from scripts.prepare_data import X_train, X_val, y_train, y_val


# model
model = run_pipeline(X_train, y_train)

# evaluation
metrics = evaluate_model(
    model, X_val, y_val, threshold=MODEL_CONFIG_V2['threshold'])


model_path = save_model(model, version='v2')
metrics_path = save_metrics(metrics, version='v2')
config_path = save_config(MODEL_CONFIG_V2, version='v2')

print(f'Model saved to: {model_path}')
print(f'Metrics saved to: {metrics_path}')
print(f'Config saved to: {config_path}')
