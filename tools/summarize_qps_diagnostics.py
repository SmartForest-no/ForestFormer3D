import argparse
import csv
import json
import os
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize FF3D QPS diagnostic JSON files.')
    parser.add_argument(
        'diag_dir',
        help='Directory containing *_qps_diag.json files.')
    parser.add_argument(
        '--out-dir',
        help='Directory for summary outputs. Defaults to <diag_dir>/summary.')
    return parser.parse_args()


def load_scene_payloads(diag_dir):
    files = sorted(
        os.path.join(diag_dir, name)
        for name in os.listdir(diag_dir)
        if name.endswith('_qps_diag.json'))
    payloads = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            payloads.append(json.load(f))
    return payloads


def build_scene_row(payload):
    summary = payload.get('scene_summary', {})
    row = {
        'scene_name': payload.get('scene_name'),
        'fps_mode': payload.get('fps_mode'),
        'query_budget': payload.get('query_budget'),
        'num_recorded_regions': payload.get('num_recorded_regions', 0),
        'diagnostic_region_count': summary.get('diagnostic_region_count', 0),
    }
    for prefix, key in (
            ('avg_', 'region_average'),
            ('med_', 'region_median')):
        metrics = summary.get(key, {})
        for metric_name, value in metrics.items():
            row[f'{prefix}{metric_name}'] = value
    worst = summary.get('worst_region', {})
    row['worst_region_index'] = worst.get('region_index', -1)
    row['worst_query_instance_coverage'] = worst.get(
        'query_instance_coverage', 0.0)
    row['worst_query_small_tree_coverage'] = worst.get(
        'query_small_tree_coverage', 0.0)
    row['worst_decoder_external_instance_coverage'] = worst.get(
        'decoder_external_instance_coverage', 0.0)
    return row


def summarize_rows(rows):
    if not rows:
        return {
            'num_scenes': 0,
            'mean_of_scene_averages': {},
            'median_of_scene_averages': {},
            'worst_scene_by_small_tree_coverage': None,
        }

    metric_keys = sorted(
        key for key in rows[0].keys()
        if key.startswith('avg_'))
    mean_summary = {}
    median_summary = {}
    for key in metric_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        mean_summary[key] = float(mean(values))
        median_summary[key] = float(median(values))

    worst_scene = min(
        rows,
        key=lambda row: (
            float(row.get('avg_query_small_tree_coverage', 0.0)),
            float(row.get('avg_query_instance_coverage', 0.0)),
        ))

    return {
        'num_scenes': len(rows),
        'mean_of_scene_averages': mean_summary,
        'median_of_scene_averages': median_summary,
        'worst_scene_by_small_tree_coverage': {
            'scene_name': worst_scene.get('scene_name'),
            'avg_query_small_tree_coverage': worst_scene.get(
                'avg_query_small_tree_coverage', 0.0),
            'avg_query_instance_coverage': worst_scene.get(
                'avg_query_instance_coverage', 0.0),
            'worst_region_index': worst_scene.get('worst_region_index', -1),
        },
    }


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.diag_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)

    payloads = load_scene_payloads(args.diag_dir)
    rows = [build_scene_row(payload) for payload in payloads]
    dataset_summary = summarize_rows(rows)

    csv_path = os.path.join(out_dir, 'scene_summary.csv')
    json_path = os.path.join(out_dir, 'dataset_summary.json')

    write_csv(rows, csv_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=2)

    print(f'[QPS-DIAG] scenes={dataset_summary["num_scenes"]}')
    print(f'[QPS-DIAG] wrote {csv_path}')
    print(f'[QPS-DIAG] wrote {json_path}')
    worst = dataset_summary.get('worst_scene_by_small_tree_coverage')
    if worst:
        print(
            '[QPS-DIAG] worst_small_tree_scene='
            f'{worst["scene_name"]} '
            f'avg_query_small={worst["avg_query_small_tree_coverage"]:.3f} '
            f'avg_query_cov={worst["avg_query_instance_coverage"]:.3f}'
        )


if __name__ == '__main__':
    main()
