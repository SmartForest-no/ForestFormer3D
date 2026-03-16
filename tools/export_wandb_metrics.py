#!/usr/bin/env python3
import argparse
import os

import wandb


def main():
    parser = argparse.ArgumentParser(
        description='Export W&B run history to CSV.')
    parser.add_argument('--run-id', required=True, help='W&B run id')
    parser.add_argument(
        '--entity', default='wuhaili2002-cas', help='W&B entity')
    parser.add_argument(
        '--project', default='ForestFormer3D', help='W&B project')
    parser.add_argument(
        '--output', default='metrics.csv', help='Output CSV path')
    args = parser.parse_args()

    api_key = os.getenv('WANDB_API_KEY', '')
    if api_key:
        wandb.login(key=api_key, relogin=True)

    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.project}/{args.run_id}')
    metrics_dataframe = run.history()
    metrics_dataframe.to_csv(args.output, index=False)
    print(
        f'Exported metrics to {args.output} from '
        f'{args.entity}/{args.project}/{args.run_id}')


if __name__ == '__main__':
    main()
