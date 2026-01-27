from src.pipeline.train import run_pipeline
from src.pipeline.evaluate import evaluate_model
from src.save_artifact import save_model, save_metrics, save_config
from src.config import MODEL_CONFIG_V2
from scripts.prepare_data import X_train, X_val, y_train, y_val, X_test, y_test


# model
model = run_pipeline(X_train, y_train)  # Train the model

# evaluation
metrics = evaluate_model(
    # Evaluate the model
    model, X_val, y_val, threshold=MODEL_CONFIG_V2['threshold'])

# final evaluation on the test set
test_metrics = evaluate_model(
    # Evaluate on test set
    model, X_test, y_test, threshold=MODEL_CONFIG_V2['threshold'])


model_path = save_model(model, version='v3')  # Save the model
metrics_path = save_metrics(test_metrics, version='v3')  # Save the metrics
config_path = save_config(MODEL_CONFIG_V2, version='v3')  # Save the config

print(f'Model saved to: {model_path}')
print(f'Metrics saved to: {metrics_path}')
print(f'Config saved to: {config_path}')
