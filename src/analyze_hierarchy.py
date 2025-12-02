import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def load_cache(base_dir: Path) -> EcephysProjectCache:
    base_dir.mkdir(parents=True, exist_ok=True)
    return EcephysProjectCache.from_warehouse(manifest=str(base_dir / 'manifest.json'))

def map_areas(df: pd.DataFrame) -> pd.DataFrame:
    area_map = {
        'LGN': ['LGd'], 'LP': ['LP'], 'V1': ['VISp'], 'LM': ['VISl'],
        'AL': ['VISal'], 'RL': ['VISrl'], 'PM': ['VISpm'], 'AM': ['VISam']
    }
    inv_map = {a: k for k, vs in area_map.items() for a in vs}
    df['area'] = df['ecephys_structure_acronym'].map(inv_map)
    return df

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    q = (df['presence_ratio'] > 0.95) & (df['isi_violations'] < 0.5) & (df['amplitude_cutoff'] < 0.1)
    return df[q & df['area'].notnull()].copy()

def summarize(mq: pd.DataFrame, areas: list, mod_col: str, imgsel_col: str) -> pd.DataFrame:
    rows = []
    for a in areas:
        dfa = mq[mq['area'] == a]
        rows.append({
            'area': a,
            'n_units': int(len(dfa)),
            'median_modulation_index': float(dfa[mod_col].median()) if len(dfa) > 0 else np.nan,
            'median_image_selectivity': float(dfa[imgsel_col].median()) if len(dfa) > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def correlations(summ_df: pd.DataFrame) -> dict:
    ranks = {'LGN': 0, 'LP': 1, 'V1': 2, 'LM': 3, 'AL': 4, 'RL': 5, 'PM': 6, 'AM': 7}
    summ_df['rank'] = summ_df['area'].map(ranks)
    r_mod, p_mod = spearmanr(summ_df['rank'], summ_df['median_modulation_index'], nan_policy='omit')
    r_img, p_img = spearmanr(summ_df['rank'], summ_df['median_image_selectivity'], nan_policy='omit')
    return {
        'mod_idx_dg': {'r': float(r_mod), 'p': float(p_mod)},
        'image_selectivity_ns': {'r': float(r_img), 'p': float(p_img)}
    }

def visualizations(mq: pd.DataFrame, summ_df: pd.DataFrame, mod_col: str, imgsel_col: str, out_dir: Path):
    areas = ['LGN', 'LP', 'V1', 'LM', 'AL', 'RL', 'PM', 'AM']
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style='whitegrid')
    # Box + strip
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='area', y=mod_col, data=mq, order=areas)
    sns.stripplot(x='area', y=mod_col, data=mq.sample(min(2000, len(mq))), order=areas, color='black', alpha=0.2, size=2)
    plt.title('Drifting gratings modulation index (mod_idx_dg) by area')
    plt.tight_layout()
    plt.savefig(out_dir / 'mod_idx_dg_by_area.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='area', y=imgsel_col, data=mq, order=areas)
    sns.stripplot(x='area', y=imgsel_col, data=mq.sample(min(2000, len(mq))), order=areas, color='black', alpha=0.2, size=2)
    plt.title('Natural scenes image selectivity (image_selectivity_ns) by area')
    plt.tight_layout()
    plt.savefig(out_dir / 'image_selectivity_ns_by_area.png', dpi=200)
    plt.close()

    # Trends
    plt.figure(figsize=(8, 5))
    plt.plot(summ_df['rank'], summ_df['median_modulation_index'], '-o')
    plt.xticks(summ_df['rank'], summ_df['area'])
    plt.ylabel('Median mod_idx_dg')
    plt.title('Median drifting-gratings modulation vs anatomical rank')
    plt.tight_layout()
    plt.savefig(out_dir / 'median_mod_idx_dg_trend.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summ_df['rank'], summ_df['median_image_selectivity'], '-o', color='orange')
    plt.xticks(summ_df['rank'], summ_df['area'])
    plt.ylabel('Median image_selectivity_ns')
    plt.title('Median natural scenes selectivity vs anatomical rank')
    plt.tight_layout()
    plt.savefig(out_dir / 'median_image_selectivity_ns_trend.png', dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze functional hierarchy in Allen ecephys Neuropixels data')
    parser.add_argument('--base_dir', default='data/allen_ecephys', help='Cache base directory for AllenSDK warehouse manifest')
    parser.add_argument('--output_dir', default='results', help='Directory to save figures and summaries')
    parser.add_argument('--session_type', default='brain_observatory_1.1', choices=['brain_observatory_1.1', 'functional_connectivity'])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)

    cache = load_cache(base_dir)
    units = cache.get_units()
    metrics = cache.get_unit_analysis_metrics_by_session_type(args.session_type).reset_index()
    metrics = metrics.rename(columns={'ecephys_unit_id': 'unit_id'})

    units_st = units[units['session_type'] == args.session_type].copy()
    if 'unit_id' not in units_st.columns:
        units_st = units_st.reset_index().rename(columns={'id': 'unit_id'})

    m = units_st.merge(metrics[['unit_id', 'mod_idx_dg', 'image_selectivity_ns']], on='unit_id', how='inner')
    m = map_areas(m)
    mq = quality_filter(m)

    areas = ['LGN', 'LP', 'V1', 'LM', 'AL', 'RL', 'PM', 'AM']
    summ_df = summarize(mq, areas, 'mod_idx_dg', 'image_selectivity_ns')
    stats = correlations(summ_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    summ_df.to_csv(out_dir / 'area_summary.csv', index=False)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'summary': summ_df.to_dict(orient='records'), 'spearman': stats}, f, indent=2)

    visualizations(mq, summ_df, 'mod_idx_dg', 'image_selectivity_ns', out_dir)
    print('Analysis complete. Outputs saved to', out_dir)

if __name__ == '__main__':
    main()
