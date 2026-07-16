import logging

def log_mean_metrics_per_type(config, metricHandler, metric_dict):
    main_metric_config = config['evaluation']['main_metric']
    
    # log one line per endpoint type present (e.g. Binary AUC, Event C-index)
    for endpoint_type, mean_val in metricHandler.mean_metric_per_type(metric_dict).items():
        metric_name = main_metric_config[endpoint_type] if isinstance(main_metric_config, dict) else main_metric_config
        logging.info(f'   Mean {endpoint_type} {metric_name}: {mean_val}')