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

import json
import argparse
import awkward

from autodqm_ml.utils import expand_path
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
  return parser.parse_args()

def count_number_of_hists_above_threshold(Fdf, Fthreshold_list):
  runs_list = Fdf['run_number']
  Ft_list = np.array([float(Fthreshold_list_item) for Fthreshold_list_item in Fthreshold_list])
  hist_bad_count = 0
  bad_hist_array = []
  for run in runs_list:
    run_row = Fdf.loc[Fdf['run_number'] == run].drop(columns=['run_number'])
    run_row = run_row.iloc[0].values
    hist_bad_count = sum(hist_sse > hist_thresh for hist_sse, hist_thresh in zip(run_row, Ft_list))
    bad_hist_array.append(hist_bad_count)
  return bad_hist_array

# returns mean number of runs with SSE above the given threshold
def count_mean_runs_above(Fdf, Fthreshold_list):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  mean_hists_flagged_per_run = sum(hists_flagged_per_run) / len(Fdf['run_number'])
  return mean_hists_flagged_per_run

# returns fraction of runs with SSE above the given threshold
def count_fraction_runs_above(Fdf, Fthreshold_list, N_bad_hists):
  hists_flagged_per_run = count_number_of_hists_above_threshold(Fdf, Fthreshold_list)
  count = len([i for i in hists_flagged_per_run if i > N_bad_hists])
  count_per_run = count / len(Fdf['run_number'])
  return count_per_run

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
  sse_df_ae['label'] = runs['label']

  sse_df_pca = pd.DataFrame(runs.run_number)
  sse_df_pca.columns = ["run_number"]
  sse_df_pca['algo'] = 'pca'
  sse_df_pca['label'] = runs['label']

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
  
  #bad_runs = [int(run) for run in args.bad_runs.split(",")]

  df_pca = sse_df.loc[sse_df['algo'] == "pca"]
  df_ae = sse_df.loc[sse_df['algo'] == "ae"]

  for I in ["PCA", "AE"]:

    if I == "PCA": J = df_pca
    if I == "AE": J = df_ae

    hist_cols = [col for col in J.columns if 'Run summary' in col]
    hist_dict = {each_hist: "max" for each_hist in hist_cols}

    J = J.groupby(['run_number','label'])[hist_cols].agg(hist_dict).reset_index()

    J = J.sort_values(['label']).reset_index()
    J = J[['run_number','label'] + [col for col in J.columns if (col != 'run_number')&(col != 'label')]]

    # new threshold cut-offs per Si's recommendations
    # 0th cut-off at 1st highest SSE + (1st - 2nd highest)*0.5   
    # 1st cut-off at mean<1st, 2nd> highest SSE
    # Nth cut-off at mean<Nth, N+1th> highest SSE
    cutoffs_across_hists = []
    for histogram in hist_cols:
      sse_ordered = sorted(J[histogram], reverse=True)
      cutoff_0 = sse_ordered[0] + 0.5*(sse_ordered[0] - sse_ordered[1])
      cutoff_thresholds = []
      cutoff_thresholds.append(cutoff_0)
      for ii in range(len(sse_ordered)-1):
        cutoff_ii = 0.5*(sse_ordered[ii]+sse_ordered[ii+1])
        cutoff_thresholds.append(cutoff_ii)
      cutoffs_across_hists.append(cutoff_thresholds)

    if len(cutoffs_across_hists[0]) < 7:
      print("There are only " + str(len(cutoffs_across_hists)) + " runs for study, which is a very small number. The script will need modifying to account for this.")
    else:
      cutoffs_across_hists = np.array(cutoffs_across_hists)

    pct_99 = []
    pct_95 = []
    pct_90 = []
    pct_80 = []
    pct_70 = []
    pct_60 = []
    pct_40 = []
    pct_20 = []
    med = []

    med = J[hist_cols].median().values
    pct_99 = J[hist_cols].quantile(q=0.99).values
    pct_95 = J[hist_cols].quantile(q=0.95).values
    pct_90 = J[hist_cols].quantile(q=0.90).values
    pct_80 = J[hist_cols].quantile(q=0.80).values
    pct_70 = J[hist_cols].quantile(q=0.70).values
    pct_60 = J[hist_cols].quantile(q=0.60).values
    pct_40 = J[hist_cols].quantile(q=0.40).values
    pct_20 = J[hist_cols].quantile(q=0.20).values

    null_set = med*0.0
    med_0p3 = med*0.3
    med_0p6 = med*0.6
    med_0p9 = med*0.9
    med_1p2 = med*1.2
    med_1p5 = med*1.5
    med_1p8 = med*1.8

    J_good = J.loc[J['label'] == 0].reset_index()
    J_bad = J.loc[J['label'] == 1].reset_index()
    J_good = J_good[['run_number'] + hist_cols]
    J_bad = J_bad[['run_number'] + hist_cols]

    #### number of bad histograms
    if len(hist_cols) < 5:
      print("There are too few histograms for a meaningful analysis, would consider at least 10 histograms")
      break
    N_bad_hists = [5,3,1]
    tFRF_ROC_good_X = []
    tFRF_ROC_bad_Y = []
    mFRF_ROC_good_X = []
    mFRF_ROC_bad_Y = []
    pFRF_ROC_good_X = []
    pFRF_ROC_bad_Y = []

    for nbh_ii in N_bad_hists:
      tFRF_ROC_good_X_init = [0.0]
      tFRF_ROC_bad_Y_init = [0.0]
      for cutoff_index in range(len(cutoffs_across_hists[:])):
        t_cutoff_index_g_FRF_rc = count_fraction_runs_above(J_good, cutoffs_across_hists[:,cutoff_index], nbh_ii)
        t_cutoff_index_b_FRF_rc = count_fraction_runs_above(J_bad, cutoffs_across_hists[:,cutoff_index], nbh_ii)
        tFRF_ROC_good_X_init.append(t_cutoff_index_g_FRF_rc)
        tFRF_ROC_bad_Y_init.append(t_cutoff_index_b_FRF_rc)

      tFRF_ROC_good_X_init = sorted(tFRF_ROC_good_X_init)
      tFRF_ROC_bad_Y_init = sorted(tFRF_ROC_bad_Y_init)

      tFRF_ROC_good_X.append(tFRF_ROC_good_X_init)
      tFRF_ROC_bad_Y.append(tFRF_ROC_bad_Y_init)

      p99g_FRF_rc = count_fraction_runs_above(J_good, pct_99, nbh_ii)
      p95g_FRF_rc = count_fraction_runs_above(J_good, pct_95, nbh_ii)
      p90g_FRF_rc = count_fraction_runs_above(J_good, pct_90, nbh_ii)
      p80g_FRF_rc = count_fraction_runs_above(J_good, pct_80, nbh_ii)
      p70g_FRF_rc = count_fraction_runs_above(J_good, pct_70, nbh_ii)
      p60g_FRF_rc = count_fraction_runs_above(J_good, pct_60, nbh_ii)
      p40g_FRF_rc = count_fraction_runs_above(J_good, pct_40, nbh_ii)
      p20g_FRF_rc = count_fraction_runs_above(J_good, pct_20, nbh_ii)
      nsg_FRF_rc = count_fraction_runs_above(J_good, null_set, nbh_ii)
      m03g_FRF_rc = count_fraction_runs_above(J_good, med_0p3, nbh_ii)
      m06g_FRF_rc = count_fraction_runs_above(J_good, med_0p6, nbh_ii)
      m09g_FRF_rc = count_fraction_runs_above(J_good, med_0p9, nbh_ii)
      m10g_FRF_rc = count_fraction_runs_above(J_good, med, nbh_ii)
      m12g_FRF_rc = count_fraction_runs_above(J_good, med_1p2, nbh_ii)
      m15g_FRF_rc = count_fraction_runs_above(J_good, med_1p5, nbh_ii)
      m18g_FRF_rc = count_fraction_runs_above(J_good, med_1p8, nbh_ii)

      p99b_FRF_rc = count_fraction_runs_above(J_bad, pct_99, nbh_ii)
      p95b_FRF_rc = count_fraction_runs_above(J_bad, pct_95, nbh_ii)
      p90b_FRF_rc = count_fraction_runs_above(J_bad, pct_90, nbh_ii)
      p80b_FRF_rc = count_fraction_runs_above(J_bad, pct_80, nbh_ii)
      p70b_FRF_rc = count_fraction_runs_above(J_bad, pct_70, nbh_ii)
      p60b_FRF_rc = count_fraction_runs_above(J_bad, pct_60, nbh_ii)
      p40b_FRF_rc = count_fraction_runs_above(J_bad, pct_40, nbh_ii)
      p20b_FRF_rc = count_fraction_runs_above(J_bad, pct_20, nbh_ii)
      nsb_FRF_rc = count_fraction_runs_above(J_bad, null_set, nbh_ii)
      m03b_FRF_rc = count_fraction_runs_above(J_bad, med_0p3, nbh_ii)
      m06b_FRF_rc = count_fraction_runs_above(J_bad, med_0p6, nbh_ii)
      m09b_FRF_rc = count_fraction_runs_above(J_bad, med_0p9, nbh_ii)
      m10b_FRF_rc = count_fraction_runs_above(J_bad, med, nbh_ii)
      m12b_FRF_rc = count_fraction_runs_above(J_bad, med_1p2, nbh_ii)
      m15b_FRF_rc = count_fraction_runs_above(J_bad, med_1p5, nbh_ii)
      m18b_FRF_rc = count_fraction_runs_above(J_bad, med_1p8, nbh_ii)

      mFRF_ROC_good_X_init = sorted([nsg_FRF_rc,m03g_FRF_rc,m06g_FRF_rc,m09g_FRF_rc,m10g_FRF_rc,m12g_FRF_rc,m15g_FRF_rc,m18g_FRF_rc,0.0])
      mFRF_ROC_bad_Y_init = sorted([nsb_FRF_rc,m03b_FRF_rc,m06b_FRF_rc,m09b_FRF_rc,m10b_FRF_rc,m12b_FRF_rc,m15b_FRF_rc,m18b_FRF_rc,0.0])

      pFRF_ROC_good_X_init = sorted([nsg_FRF_rc,p99g_FRF_rc,p95g_FRF_rc,p90g_FRF_rc,p80g_FRF_rc,p70g_FRF_rc,p60g_FRF_rc,p40g_FRF_rc,p20g_FRF_rc,0.0])
      pFRF_ROC_bad_Y_init = sorted([nsb_FRF_rc,p99b_FRF_rc,p95b_FRF_rc,p90b_FRF_rc,p80b_FRF_rc,p70b_FRF_rc,p60b_FRF_rc,p40b_FRF_rc,p20b_FRF_rc,0.0])

      mFRF_ROC_good_X.append(mFRF_ROC_good_X_init)
      mFRF_ROC_bad_Y.append(mFRF_ROC_bad_Y_init)
      pFRF_ROC_good_X.append(pFRF_ROC_good_X_init)
      pFRF_ROC_bad_Y.append(pFRF_ROC_bad_Y_init)

    t0g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,0])
    t1g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,1])
    t2g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,2])
    t3g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,3])
    t4g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,4])
    t5g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,5])
    t6g_MRF_rc = count_mean_runs_above(J_good, cutoffs_across_hists[:,6])
    t0b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,0])
    t1b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,1])
    t2b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,2])
    t3b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,3])
    t4b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,4])
    t5b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,5])
    t6b_MRF_rc = count_mean_runs_above(J_bad, cutoffs_across_hists[:,6])

    p99g_MRF_rc = count_mean_runs_above(J_good, pct_99)
    p95g_MRF_rc = count_mean_runs_above(J_good, pct_95)
    p90g_MRF_rc = count_mean_runs_above(J_good, pct_90)
    p80g_MRF_rc = count_mean_runs_above(J_good, pct_80)
    p70g_MRF_rc = count_mean_runs_above(J_good, pct_70)
    p60g_MRF_rc = count_mean_runs_above(J_good, pct_60)
    p40g_MRF_rc = count_mean_runs_above(J_good, pct_40)
    p20g_MRF_rc = count_mean_runs_above(J_good, pct_20)
    nsg_MRF_rc = count_mean_runs_above(J_good, null_set)
    m03g_MRF_rc = count_mean_runs_above(J_good, med_0p3)
    m06g_MRF_rc = count_mean_runs_above(J_good, med_0p6)
    m09g_MRF_rc = count_mean_runs_above(J_good, med_0p9)
    m10g_MRF_rc = count_mean_runs_above(J_good, med)
    m12g_MRF_rc = count_mean_runs_above(J_good, med_1p2)
    m15g_MRF_rc = count_mean_runs_above(J_good, med_1p5)
    m18g_MRF_rc = count_mean_runs_above(J_good, med_1p8)

    p99b_MRF_rc = count_mean_runs_above(J_bad, pct_99)
    p95b_MRF_rc = count_mean_runs_above(J_bad, pct_95)
    p90b_MRF_rc = count_mean_runs_above(J_bad, pct_90)
    p80b_MRF_rc = count_mean_runs_above(J_bad, pct_80)
    p70b_MRF_rc = count_mean_runs_above(J_bad, pct_70)
    p60b_MRF_rc = count_mean_runs_above(J_bad, pct_60)
    p40b_MRF_rc = count_mean_runs_above(J_bad, pct_40)
    p20b_MRF_rc = count_mean_runs_above(J_bad, pct_20)
    nsb_MRF_rc = count_mean_runs_above(J_bad, null_set)
    m03b_MRF_rc = count_mean_runs_above(J_bad, med_0p3)
    m06b_MRF_rc = count_mean_runs_above(J_bad, med_0p6)
    m09b_MRF_rc = count_mean_runs_above(J_bad, med_0p9)
    m10b_MRF_rc = count_mean_runs_above(J_bad, med)
    m12b_MRF_rc = count_mean_runs_above(J_bad, med_1p2)
    m15b_MRF_rc = count_mean_runs_above(J_bad, med_1p5)
    m18b_MRF_rc = count_mean_runs_above(J_bad, med_1p8)

    tMRF_ROC_good_X = sorted([t0g_MRF_rc,t1g_MRF_rc,t2g_MRF_rc,t3g_MRF_rc,t4g_MRF_rc,t5g_MRF_rc,t6g_MRF_rc,0.0])
    tMRF_ROC_bad_Y = sorted([t0b_MRF_rc,t1b_MRF_rc,t2b_MRF_rc,t3b_MRF_rc,t4b_MRF_rc,t5b_MRF_rc,t6b_MRF_rc,0.0])

    mMRF_ROC_good_X = sorted([nsg_MRF_rc,m03g_MRF_rc,m06g_MRF_rc,m09g_MRF_rc,m10g_MRF_rc,m12g_MRF_rc,m15g_MRF_rc,m18g_MRF_rc,0.0])
    mMRF_ROC_bad_Y = sorted([nsb_MRF_rc,m03b_MRF_rc,m06b_MRF_rc,m09b_MRF_rc,m10b_MRF_rc,m12b_MRF_rc,m15b_MRF_rc,m18b_MRF_rc,0.0])

    pMRF_ROC_good_X = sorted([nsg_MRF_rc,p99g_MRF_rc,p95g_MRF_rc,p90g_MRF_rc,p80g_MRF_rc,p70g_MRF_rc,p60g_MRF_rc,p40g_MRF_rc,p20g_MRF_rc,0.0])
    pMRF_ROC_bad_Y = sorted([nsb_MRF_rc,p99b_MRF_rc,p95b_MRF_rc,p90b_MRF_rc,p80b_MRF_rc,p70b_MRF_rc,p60b_MRF_rc,p40b_MRF_rc,p20b_MRF_rc,0.0])

    fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(12,12))
    AX_val = [axs[0,1], axs[1,0], axs[1,1]]

    for jj in range(len(N_bad_hists)):
      if N_bad_hists[jj] == 1:
        AX_val[jj].set_xlabel('Fraction of good runs with at least 1 histogram flagged')
        AX_val[jj].set_ylabel('Fraction of bad runs with at least 1 histogram flagged')
      else:
        AX_val[jj].set_xlabel('Fraction of good runs with at least '+str(N_bad_hists[jj])+' histograms flagged')
        AX_val[jj].set_ylabel('Fraction of bad runs with at least '+str(N_bad_hists[jj])+' histograms flagged')
      AX_val[jj].plot(mFRF_ROC_good_X[jj],mFRF_ROC_bad_Y[jj], '-bo', mfc='orange', mec='k', markersize=8, linewidth=1, label='Median of all SSE values')
      AX_val[jj].plot(pFRF_ROC_good_X[jj],pFRF_ROC_bad_Y[jj], '-r^', mfc='green', mec='k', markersize=8, linewidth=1, label='Quantile of all SSE values')
      AX_val[jj].plot(tFRF_ROC_good_X[jj],tFRF_ROC_bad_Y[jj], '-yD', mfc='purple', mec='k', markersize=8, linewidth=1, label='N,N+1 threshold SSE values')
      AX_val[jj].axis(xmin=0,xmax=1,ymin=0,ymax=1)
      AX_val[jj].axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
      AX_val[jj].annotate("Algorithm: " + I, xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
      AX_val[jj].legend(loc='lower right')

    axs[0,0].set_xlabel('Mean number of flagged good runs')
    axs[0,0].set_ylabel('Mean number of flagged bad runs')
    axs[0,0].plot(mMRF_ROC_good_X,mMRF_ROC_bad_Y, '-bo', mfc='orange', mec='k', markersize=8, linewidth=1, label='Median of all SSE values')
    axs[0,0].plot(pMRF_ROC_good_X,pMRF_ROC_bad_Y, '-r^', mfc='green', mec='k', markersize=8, linewidth=1, label='Quantile of all SSE values')
    axs[0,0].plot(tMRF_ROC_good_X,tMRF_ROC_bad_Y, '-yD', mfc='purple', mec='k', markersize=8, linewidth=1, label='N,N+1 threshold SSE values')
    axs[0,0].axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
    axs[0,0].annotate("Algorithm: " + I, xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
    axs[0,0].axis(xmin=0,ymin=0)
    axs[0,0].legend(loc='lower right')

    plt.savefig(args.output_dir + "/FRF_MRF_ROC_comparison_" + I + ".pdf")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
