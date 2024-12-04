


import wandb
import random
import copy
import time
import wandb.wandb_run


def main():
    random.seed(1)


    wandb.login(key="c7f0f65fac8b7178ad7c5859ba6114775b16e694") 

    for trial_n in range(3):
        wandb.init(project="test_WB_8", group="test_group_4", reinit=True)
        # define a metric we are interested in the minimum of
        #wandb.define_metric("loss", goal="minimize", summary="best")
        #wandb.define_metric("loss", summary="best")
        #wandb.define_metric("loss", summary="none")
        #wandb.define_metric("loss", summary="none")
        # define a metric we are interested in the maximum of
        #wandb.define_metric("acc")

        lowest_loss = float("inf")
        best_log_dict = None
        best_epoch_n = 0

        for i in range(20):
            loss = random.uniform(0, 1 / (i + 1))
            acc = random.uniform(1 / (i + 1), 1)

            
            log_dict = {
                'epoch': i,
                "loss":loss,
                "acc": acc,
            }
            #wandb.log(log_dict, commit=commit, step=i)
            wandb.log(log_dict, step=i)
            
            if loss < lowest_loss:
                #commit = True
                lowest_loss = loss
                best_epoch_n = i
                print(lowest_loss)
                best_log_dict = copy.deepcopy(log_dict)
            else:
                pass
                #commit = False

        # fix the summary
        print(best_log_dict)
        update_WandB_summary_table(best_log_dict)
        
        #wandb.run.summary.update(best_log_dict)
        #wandb.summary.update(best_log_dict)

        #wandb.run.summary.update(best_log_dict)
        #wandb.log(best_log_dict, commit=True, step=best_epoch_n)
        
        time.sleep(2)
        
        wandb.finish()

        time.sleep(2)

def update_WandB_summary_table(best_log_dict):
    """

    """
    # supress the WandB summary-maker 
    # (it somehow insists on using the most recent log, not the 'best', even if you define what that is)
    for key in best_log_dict.keys():
        wandb.define_metric(key, summary="none")

    # update the best log. This appears in the 'summary' tab in WandB (the table of results)
    wandb.run.summary.update(best_log_dict)


if __name__ == "__main__":
    wandb.Settings(quiet=True) 
    main()