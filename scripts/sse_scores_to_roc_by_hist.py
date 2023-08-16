'''
Macro to use post-scripts/assess.py to convert the lsit of SSE scores to ROC curves for studies over a large data set
Requires input directory where bad_runs_sse_scores.csv is located (this is also the output directory) and list of bad
runs as labelled by data certification reports or similar (i.e. not algos!) (Runs not in the list of bad runs are 
considered good runs by default)
'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from statistics import mean

import json
import argparse
import awkward

from autodqm_ml.utils import setup_logger
from autodqm_ml.utils import expand_path
from autodqm_ml.plotting.plot_tools import make_original_vs_reconstructed_plot, make_sse_plot, plot_roc_curve
from autodqm_ml.evaluation.roc_tools import calc_roc_and_unc, print_eff_table
from autodqm_ml.constants import kANOMALOUS, kGOOD

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help = "output directory to place files in",
        type = str,
        required = False,
        default = "output"
    )
    parser.add_argument(
        "--input_file",
        help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
        type = str,
        required = True,
        default = None
    )
    parser.add_argument(
        "--histograms",
        help = "csv list of histograms to assess", 
        type = str,
        required = True,
        default = None
    )
    parser.add_argument(
        "--algorithms",
        help = "csv list of algorithm names to assess",
        type = str,
        required = True,
        default = None
    )
    parser.add_argument(
        "--bad_runs",
        help = "csv of runs that are labelled bad a priori by data certification reports, DPGs or similar",
        type = str,
        required = True,
        default = None
    )
    return parser.parse_args()

def count_hists_above(Fdf, Fthreshold_list):
    runs_list = Fdf['run_number']
    Ft_list = np.array([float(Fthreshold_list_item) for Fthreshold_list_item in Fthreshold_list])
    hist_bad_count = 0
    bad_hist_array = []
    print("Runs list = " + str(len(runs_list)))
    for run in runs_list:
        run_row = Fdf.loc[Fdf['run_number'] == run].drop(columns=['run_number'])
        run_row = run_row.iloc[0].values
        hist_bad_count = sum(hist_sse > hist_thresh for hist_sse, hist_thresh in zip(run_row, Ft_list))
        bad_hist_array.append(hist_bad_count)
    return sum(bad_hist_array)/len(runs_list)

def infer_algorithms(runs, histograms, algorithms):
    for histogram, info in histograms.items():
        for field in runs.fields:
            if field == histogram:
                histograms[histogram]["original"] = field
            elif histogram in field and "_score_" in field:
                algorithm = field.replace(histogram, "").replace("_score_", "")
                if algorithms is not None:
                    if algorithm not in algorithms:
                        continue

                if not algorithm in info["algorithms"].keys():
                    histograms[histogram]["algorithms"][algorithm] = { "score" : field }

                # Check if a reconstructed histogram also exists for algorithm
                reco = field.replace("score", "reco")
                if reco in runs.fields:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = reco
                else:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = None

    return histograms


def main(args):
    os.system("mkdir -p %s/" % args.output_dir)
    with open(args.output_dir + '/commands_sse_scores_to_roc.txt', 'w') as f:
        f.write(str(args))

    histograms = { x : {"algorithms" : {}} for x in args.histograms.split(",") }
    runs = awkward.from_parquet(args.input_file)

    if args.algorithms is not None:
        algorithms = args.algorithms.split(",")
    else:
        algorithms = None

    histograms = infer_algorithms(runs, histograms, algorithms)

    sse_df_ae = pd.DataFrame(runs.run_number)
    sse_df_ae.columns = ["run_number"]
    sse_df_ae['algo'] = 'ae'

    sse_df_pca = pd.DataFrame(runs.run_number)
    sse_df_pca.columns = ["run_number"]
    sse_df_pca['algo'] = 'pca'
    for h, info in histograms.items():
        for algorithm, algorithm_info in info["algorithms"].items():
            
            if any(x in algorithm.lower() for x in ["ae","autoencoder"]):
                sse_df_ae[h] = runs[algorithm_info["score"]]

            if any(x in algorithm.lower() for x in ["pca"]):
                sse_df_pca[h] = runs[algorithm_info["score"]]

    sse_df = pd.concat([sse_df_ae,sse_df_pca]).reset_index(drop=True)
    sse_df.to_csv(args.output_dir + "/runs_and_sse_scores.csv",index=False)
                
    # Saved csv of SSE scores, now to produce ROC curves
    sse_df = sse_df.loc[:,~sse_df.columns.duplicated()].copy()
  
    bad_runs = [int(run) for run in args.bad_runs.split(",")]

    df_pca = sse_df.loc[sse_df['algo'] == "pca"]
    df_ae = sse_df.loc[sse_df['algo'] == "ae"]

    for I in ["PCA", "AE"]:

      if I == "PCA": J = df_pca
      if I == "AE": J = df_ae

      hist_cols = [col for col in J.columns if 'Run summary' in col]
      hist_dict = {each_hist: "max" for each_hist in hist_cols}

      J = J.groupby('run_number')[hist_cols].agg(hist_dict).reset_index()

      J['run_good'] = "Y"
      print(bad_runs)
      print(J['run_number'])
      J.loc[J['run_number'].isin(bad_runs), 'run_good'] = "N"
      J = J.sort_values(['run_good']).reset_index()
      J = J[['run_number','run_good'] + [col for col in J.columns if (col != 'run_number')&(col != 'run_good')]]
      J_good = J.loc[J['run_good'] == "Y"].reset_index()
      J_bad = J.loc[J['run_good'] == "N"].reset_index()

      pct_99_good = []
      pct_95_good = []
      pct_90_good = []
      pct_80_good = []
      pct_70_good = []
      pct_60_good = []
      pct_40_good = []
      pct_20_good = []
      med_good = []

      pct_99_bad = []
      pct_95_bad = []
      pct_90_bad = []
      pct_80_bad = []
      pct_70_bad = []
      pct_60_bad = []
      pct_40_bad = []
      pct_20_bad = []
      med_bad = []

      med_good = J_good[hist_cols].median().values
      pct_99_good = J_good[hist_cols].quantile(q=0.99).values
      pct_95_good = J_good[hist_cols].quantile(q=0.95).values
      pct_90_good = J_good[hist_cols].quantile(q=0.90).values
      pct_80_good = J_good[hist_cols].quantile(q=0.80).values
      pct_70_good = J_good[hist_cols].quantile(q=0.70).values
      pct_60_good = J_good[hist_cols].quantile(q=0.60).values
      pct_40_good = J_good[hist_cols].quantile(q=0.40).values
      pct_20_good = J_good[hist_cols].quantile(q=0.20).values

      med_bad = J_bad[hist_cols].median().values
      pct_99_bad = J_bad[hist_cols].quantile(q=0.99).values
      pct_95_bad = J_bad[hist_cols].quantile(q=0.95).values
      pct_90_bad = J_bad[hist_cols].quantile(q=0.90).values
      pct_80_bad = J_bad[hist_cols].quantile(q=0.80).values
      pct_70_bad = J_bad[hist_cols].quantile(q=0.70).values
      pct_60_bad = J_bad[hist_cols].quantile(q=0.60).values
      pct_40_bad = J_bad[hist_cols].quantile(q=0.40).values
      pct_20_bad = J_bad[hist_cols].quantile(q=0.20).values

      null_set_good = med_good*0.0
      med_0p3_good = med_good*0.3
      med_0p6_good = med_good*0.6
      med_0p9_good = med_good*0.9
      med_1p2_good = med_good*1.2
      med_1p5_good = med_good*1.5
      med_1p8_good = med_good*1.8

      null_set_bad = med_bad*0.0
      med_0p3_bad = med_bad*0.3
      med_0p6_bad = med_bad*0.6
      med_0p9_bad = med_bad*0.9
      med_1p2_bad = med_bad*1.2
      med_1p5_bad = med_bad*1.5
      med_1p8_bad = med_bad*1.8

      J_good = J_good[['run_number'] + hist_cols]
      J_bad = J_bad[['run_number'] + hist_cols]

      p99g_hc = count_hists_above(J_good, pct_99_good)
      p95g_hc = count_hists_above(J_good, pct_95_good)
      p90g_hc = count_hists_above(J_good, pct_90_good)
      p80g_hc = count_hists_above(J_good, pct_80_good)
      p70g_hc = count_hists_above(J_good, pct_70_good)
      p60g_hc = count_hists_above(J_good, pct_60_good)
      p40g_hc = count_hists_above(J_good, pct_40_good)
      p20g_hc = count_hists_above(J_good, pct_20_good)
      nsg_hc = count_hists_above(J_good, null_set_good)
      m03g_hc = count_hists_above(J_good, med_0p3_good)
      m06g_hc = count_hists_above(J_good, med_0p6_good)
      m09g_hc = count_hists_above(J_good, med_0p9_good)
      m10g_hc = count_hists_above(J_good, med_good)
      m12g_hc = count_hists_above(J_good, med_1p2_good)
      m15g_hc = count_hists_above(J_good, med_1p5_good)
      m18g_hc = count_hists_above(J_good, med_1p8_good)

      p99b_hc = count_hists_above(J_bad, pct_99_bad)
      p95b_hc = count_hists_above(J_bad, pct_95_bad)
      p90b_hc = count_hists_above(J_bad, pct_90_bad)
      p80b_hc = count_hists_above(J_bad, pct_80_bad)
      p70b_hc = count_hists_above(J_bad, pct_70_bad)
      p60b_hc = count_hists_above(J_bad, pct_60_bad)
      p40b_hc = count_hists_above(J_bad, pct_40_bad)
      p20b_hc = count_hists_above(J_bad, pct_20_bad)
      nsb_hc = count_hists_above(J_bad, null_set_bad)
      m03b_hc = count_hists_above(J_bad, med_0p3_bad)
      m06b_hc = count_hists_above(J_bad, med_0p6_bad)
      m09b_hc = count_hists_above(J_bad, med_0p9_bad)
      m10b_hc = count_hists_above(J_bad, med_bad)
      m12b_hc = count_hists_above(J_bad, med_1p2_bad)
      m15b_hc = count_hists_above(J_bad, med_1p5_bad)
      m18b_hc = count_hists_above(J_bad, med_1p8_bad)

      mMF_ROC_good_Y = sorted([nsg_hc,m03g_hc,m06g_hc,m09g_hc,m10g_hc,m12g_hc,m15g_hc,m18g_hc,0.0])
      mMF_ROC_bad_X = sorted([nsb_hc,m03b_hc,m06b_hc,m09b_hc,m10b_hc,m12b_hc,m15b_hc,m18b_hc,0.0])

      pMF_ROC_good_Y = sorted([nsg_hc,p99g_hc,p95g_hc,p90g_hc,p80g_hc,p70g_hc,p60g_hc,p40g_hc,p20g_hc,0.0])
      pMF_ROC_bad_X = sorted([nsb_hc,p99b_hc,p95b_hc,p90b_hc,p80b_hc,p70b_hc,p60b_hc,p40b_hc,p20b_hc,0.0])

      fig, ax = plt.subplots(figsize=(6,6))

      ax.set_xlabel('Average # flagged histograms / good run')
      ax.set_ylabel('Average # flagged histograms / bad run')
      ax.axis(xmin=0,xmax=len(hist_cols),ymin=0,ymax=len(hist_cols))
      ax.plot(mMF_ROC_good_Y,mMF_ROC_bad_X, '-bo', mfc='r', mec='k', markersize=8, linewidth=1, label='SSE above median values')

      ax.plot(pMF_ROC_good_Y,pMF_ROC_bad_X, '-g^', mfc='orange', mec='k', markersize=8, linewidth=1, label='SSE above quantile values')
      ax.axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
      ax.annotate("Algorithm: " + I, xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')

      plt.legend(loc='lower right')

      plt.savefig(args.output_dir + "/MF_ROC_comparison_" + I + ".pdf")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
