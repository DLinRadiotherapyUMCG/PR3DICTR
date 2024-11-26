import os
from time import sleep
import ray
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device

from ray.tune.progress_reporter import ProgressReporter
from ray.tune.experiment.trial import Trial
from ray.train.torch import TorchTrainer, get_device
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter, ProgressReporter
from ray.tune.progress_reporter import TuneReporterBase
from ray.experimental.tqdm_ray import safe_print

from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union

class TrialTerminationReporter(TuneReporterBase):
    """Command-line reporter

    Args:
        metric_columns: Names of metrics to
            include in progress table. If this is a dict, the keys should
            be metric names and the values should be the displayed names.
            If this is a list, the metric name is used directly.
        parameter_columns: Names of parameters to
            include in progress table. If this is a dict, the keys should
            be parameter names and the values should be the displayed names.
            If this is a list, the parameter name is used directly. If empty,
            defaults to all available parameters.
        max_progress_rows: Maximum number of rows to print
            in the progress table. The progress table describes the
            progress of each trial. Defaults to 20.
        max_error_rows: Maximum number of rows to print in the
            error table. The error table lists the error file, if any,
            corresponding to each trial. Defaults to 20.
        max_column_length: Maximum column length (in characters). Column
            headers and values longer than this will be abbreviated.
        max_report_frequency: Maximum report frequency in seconds.
            Defaults to 5s.
        infer_limit: Maximum number of metrics to automatically infer
            from tune results.
        print_intermediate_tables: Print intermediate result
            tables. If None (default), will be set to True for verbosity
            levels above 3, otherwise False. If True, intermediate tables
            will be printed with experiment progress. If False, tables
            will only be printed at then end of the tuning run for verbosity
            levels greater than 2.
        metric: Metric used to determine best current trial.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """

    def __init__(
        self,
        *,
        metric_columns: Optional[Union[List[str], Dict[str, str]]] = None,
        parameter_columns: Optional[Union[List[str], Dict[str, str]]] = None,
        total_samples: Optional[int] = None,
        max_progress_rows: int = 20,
        max_error_rows: int = 20,
        max_column_length: int = 20,
        max_report_frequency: int = 5,
        infer_limit: int = 3,
        print_intermediate_tables: Optional[bool] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        sort_by_metric: bool = False,
    ):
        super(TrialTerminationReporter, self).__init__(
            metric_columns=metric_columns,
            parameter_columns=parameter_columns,
            total_samples=total_samples,
            max_progress_rows=max_progress_rows,
            max_error_rows=max_error_rows,
            max_column_length=max_column_length,
            max_report_frequency=max_report_frequency,
            infer_limit=infer_limit,
            print_intermediate_tables=print_intermediate_tables,
            metric=metric,
            mode=mode,
            sort_by_metric=sort_by_metric,
            
        )
        self.num_terminated = 0

    def _print(self, msg: str):
        safe_print(msg)

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == (Trial.TERMINATED or Trial.ERROR)])
        #print(self._metric_columns)
        if done: 
            return done
        return self.num_terminated > old_num_terminated
    
    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        self._print(self._progress_str(trials, done, *sys_info))



class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated

from ray.tune.logger import LoggerCallback
class TestLoggerCallback(LoggerCallback):
    def on_trial_start(
        self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        self.log_trial_start(trial)
        print(f"Starting: {trial, info}")

    def on_trial_complete(self, iteration, trials, trial, **info):
        print(f"Trial complete: {trial, info}")
    
    def on_trial_error(
        self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        print(f"Trial errored: {trial, info}")


# Example usage
def train_func(config):
    
    for i in range(10):
        sleep(config['num_conv_layers'])
        #if i == 1:
        #    print(1/0)
        train.report({'mean_accuracy': config['lr'] * 0.1})

def trial_str_creator(trial):
    return f"{trial.trial_id}"

from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity



if __name__ == '__main__':
    os.environ["RAY_AIR_NEW_OUTPUT"] = '0'
    #set_verbosity(Verbosity.V0_MINIMAL)

    ray.shutdown()
    ray.init(log_to_driver=True)
    algo = OptunaSearch()  # use optuna
    # Concurrency limiter: prevents too many trials from running at the same time
    algo = ConcurrencyLimiter(algo, max_concurrent=3)

    
    param_space = {"lr": tune.choice([0, 0.001, 0.01, 0.1]),
                   "num_conv_layers": tune.randint(2, 6),
                   }
    tuner = tune.Tuner(
        tune.with_resources(train_func, {"cpu": 2, "gpu": 0}),
        run_config=train.RunConfig(
            progress_reporter=TrialTerminationReporter(),            
            #callbacks=[LoggerCallback()]
            ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=5,
            metric="mean_accuracy",
            mode="max",
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
        ),
        param_space=param_space
        )
                       #progress_reporter=ShowProgressOnTerminationReporter())
    results = tuner.fit()


"""

def objective(hp_param_config):
    print("Hello")

    for key in hp_param_config:
        cfg.key = hp_param_config[key]

    train.report({"mean_loss": random.random()})


def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
num_samples = 3

storage_path = "C:/Users/MacRaeDC/ray_results"
exp_name = "test_123"

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
        trial_name_creator=trial_str_creator,
    ),
    run_config=train.RunConfig(
        name=exp_name, storage_path=storage_path),

    param_space=param_space,
)
results = tuner.fit()

"""