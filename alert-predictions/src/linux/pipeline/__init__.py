# Pipeline module for log-forecast
"""
Batch processing pipeline for training and inference.

Scripts (in execution order):
    01_convert_log_to_csv: Convert syslog to CSV
    02_windowize: Create time-bucketed windows
    03_label_windows: Apply rule-based anomaly labels
    04_add_trends: Add rolling trend features
    05_make_future_labels: Create prediction targets
    06_train_baseline: Train logistic regression model
    07_infer_baseline: Run batch inference
    08_train_seq_gru: Train GRU sequence model (optional)
"""
