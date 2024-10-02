# Imports

import argparse

import wandb
# from IPython.display import Image, display
import torch
import gc

import os
import time

# from transformer_lens.hook_points import HookedRootModule, HookPoint
# from transformer_lens.HookedTransformer import (
#     HookedTransformer,
# )

from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCExperiment import TLACDCExperiment

from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.text_entailment.utils import get_all_text_entailment_things

import argparse

torch.autograd.set_grad_enabled(False)


def get_all_task_things(task, device, num_examples, metric_name, model_name=None):

    if task == "greaterthan":
        things = get_all_greaterthan_things(
            num_examples=num_examples, metric_name=metric_name, device=device
        )
    elif task == "text-entailment":
        model_name = model_name
        things = get_all_text_entailment_things(
            model_name=model_name, num_examples=num_examples, metric_name=metric_name, device=device
        )
    else:
        raise ValueError(f"Unknown task {task}")
    return things
    
def main(args):
    
    # IN_COLAB = False
    # print("Running outside of colab")
    
    # if not os.path.exists("ims/"):
    #     os.mkdir("ims/")

    # warnings.filterwarnings("ignore")

    # setup task
    use_pos_embed = args.task.startswith("tracr")
    second_metric = None  # some tasks only have one metric
    things = get_all_task_things(args.task, args.device, args.nexamples, args.metric, args.model_name)

    # extract the things
    validation_metric = things.validation_metric # metric we use (e.g KL divergence)
    toks_int_values = things.validation_data # clean data x_i
    toks_int_values_other = things.validation_patch_data # corrupted data x_i'
    tl_model = things.tl_model # transformerlens model

    if args.reset_network:
        reset_network(args.task, args.device, tl_model)

    tl_model.reset_hooks()
    # Save some mem
    gc.collect()
    torch.cuda.empty_cache()

    # wandb setup
    exp_name = f"{ct()}{'_randomindices' if args.indices_mode=='random' else ''}_{args.threshold}{'_zero' if args.zero_ablation else ''}"
    if not args.using_wandb:
        WANDB_RUN_NAME = args.wandb_run_name if args.wandb_run_name else exp_name
        args.wandb_group_name = args.wandb_group_name if args.wandb_group_name else "example_group"
    else:
        if args.wandb_run_name is None:
            WANDB_RUN_NAME = f"{args.task}_{args.model_name.split('/')[-1].split('_')[-1] if 'andres-vs/' in args.model_name else args.model_name}_{args.threshold}"
            # WANDB_RUN_NAME = f"{args.task}_{args.model_name.split("/")[-1].split("_")[-1] if "andres-vs/" in args.model_name else args.model_name}_{args.threshold}"
    wandb_notes = "No notes generated"

    tl_model.reset_hooks()

    exp = TLACDCExperiment(
        model=tl_model,
        threshold=args.threshold,
        using_wandb=args.using_wandb,
        wandb_entity_name=args.wandb_entity_name,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_group_name=args.wandb_group_name,
        wandb_notes=wandb_notes,
        wandb_dir=args.wandb_dir,
        wandb_mode=args.wandb_mode,
        wandb_config=args,
        zero_ablation=args.zero_ablation,
        abs_value_threshold=args.abs_value_threshold,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=validation_metric,
        second_metric=second_metric,
        verbose=False,
        indices_mode=args.indices_mode,
        names_mode=args.names_mode,
        corrupted_cache_cpu=args.corrupted_cache_cpu,
        hook_verbose=False,
        online_cache_cpu=args.online_cache_cpu,
        add_sender_hooks=True,
        use_pos_embed=use_pos_embed,
        add_receiver_hooks=False,
        remove_redundant=True,
        show_full_index=use_pos_embed,
    )
    print("finished setting up experiment")
    start_time = time.time()
    for i in range(args.max_num_epochs):
        exp.step(testing=False)

        # show(
        #     exp.corr,
        #     f"ims/img_new_{i+1}.png",
        #     show_full_index=False,
        # )

        # if IN_COLAB or ipython is not None:
        #     display(Image(f"ims/img_new_{i+1}.png"))

        # print(i, "-" * 50)
        # print(exp.count_no_edges())
        print(f"Step {i+1} took {time.time()-start_time} seconds")
        start_time = time.time()

        if i == 0:
            exp.save_edges("edges.pkl")

        if exp.current_node is None or args.single_step:
            break

    exp.save_edges("final_edges.pkl")
    exp.save_subgraph("final_subgraph.pkl")

    if args.using_wandb:
        edges_fname = f"edges.pth"
        subgraph_fname = f"subgraph.pth"
        exp.save_edges(edges_fname)
        exp.save_subgraph(subgraph_fname)
        edges_artifact = wandb.Artifact(edges_fname, type="dataset")
        subgraph_artifact = wandb.Artifact(subgraph_fname, type="dataset")
        edges_artifact.add_file(edges_fname)
        subgraph_artifact.add_file(subgraph_fname)
        wandb.log_artifact(edges_artifact)
        wandb.log_artifact(subgraph_artifact)
        os.remove(edges_fname)
        os.remove(subgraph_fname)
        wandb.finish()

    wandb.finish()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACDC Experiment")
    parser.add_argument("--model_name", type=str, default="andres-vs/bert-base-uncased-finetuned_Att-Noneg-depth0", help="Name of the model to use")
    parser.add_argument("--nexamples", type=int, default=10, help="Number of examples")
    task_choices = ['greaterthan', 'text-entailment']
    parser.add_argument('--task', type=str, required=True, choices=task_choices, help=f'Choose a task from the available options: {task_choices}')
    parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
    parser.add_argument('--online_cache_cpu', type=str, required=False, default="False", help='Value for ONLINE_CACHE_CPU (the old name for the `online_cache`)')
    parser.add_argument('--corrupted_cache_cpu', type=str, required=False, default="False", help='Value for SECOND_CACHE_CPU (the old name for the `corrupted_cache`)')
    parser.add_argument('--zero_ablation', action='store_true', help='Use zero ablation')
    parser.add_argument('--using_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_entity_name', type=str, required=False, default="default", help='Value for WANDB_ENTITY_NAME')
    parser.add_argument('--wandb_group_name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
    parser.add_argument('--wandb_project_name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
    parser.add_argument('--wandb_run_name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
    parser.add_argument("--wandb_dir", type=str, default="/tmp/wandb")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument('--indices_mode', type=str, default="reverse", choices=["normal", "reverse"], help="Which mode to use for the indices")
    parser.add_argument('--names_mode', type=str, default="normal")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Which device to use")
    parser.add_argument('--reset_network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
    parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
    parser.add_argument('--torch_num_threads', type=int, default=0, help="How many threads to use for torch (0=all)")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--max_num_epochs",type=int, default=100_000)
    parser.add_argument('--single_step', action='store_true', help='Use single step, mostly for testing')
    parser.add_argument("--abs_value_threshold", action='store_true', help='Use the absolute value of the result to check threshold')

    args = parser.parse_args()
    print(args)
    main(args)
