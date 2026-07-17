class EarlyStopTrial(Exception):
    """
    Raised by K_fold_cross_validation when an `early_stop_callback` decides that a trial
    should be stopped early (e.g. because intermediate fold results are clearly unpromising).
    This is intentionally framework-agnostic (i.e. does not depend on Optuna) so that
    src/training/k_fold_cross_validation.py does not need to know about Optuna at all.
    Callers (e.g. OptunaExperimentManager) are responsible for catching this and translating
    it into whatever their hyperparameter-tuning framework expects (e.g. optuna.exceptions.TrialPruned).
    """
    pass
