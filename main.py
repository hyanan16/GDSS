import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol
from decode import decode_Sampler_mol
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config) 
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
    elif work_type_args.type == 'decode_valueF_train':
        wandb.init(
            entity='grelu',
            project="RNA-optimization",
            job_type='FA',
            name='decode_rewardF'
        )
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = decode_Sampler_mol(config)
        else:
            sampler = Sampler(config)
        reward_model_preds, selected_baseline_preds, baseline_preds = sampler.controlled_decode(sample_M=args.sample_M)
        hepg2_values_ours = reward_model_preds.cpu().numpy()
        hepg2_values_selected = selected_baseline_preds.cpu().numpy()
        hepg2_values_baseline = baseline_preds.cpu().numpy()
        import pandas as pd
        # Create a DataFrame for seaborn
        df = pd.DataFrame({
            f'{args.reward_name}': np.concatenate([hepg2_values_ours, hepg2_values_selected, hepg2_values_baseline]),
            'Type': ['Ours'] * len(hepg2_values_ours) + ['Filtered'] * len(hepg2_values_selected) + ['Baseline'] * len(hepg2_values_baseline)
        })

        # Plot using seaborn
        plt.figure(figsize=(15, 10))
        sns.violinplot(x='Type', y=f'{args.reward_name}', data=df, inner='box', scale='width')
        plt.title(f'Distribution of {args.reward_name} Values')
        plt.xlabel('Type')
        plt.ylabel(f'{args.reward_name}')

        # Save the plot
        plt.savefig(f"{args.reward_name}_distribution_comparison.png")

        # Upload to wandb
        wandb.log({f"{args.reward_name}_distribution_comparison": wandb.Image(
            f"{args.reward_name}_distribution_comparison.png")})

        wandb.finish()
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
