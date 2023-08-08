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

def count_hists_above(Fdf, Fthreshold_list):
  runs_list = Fdf['run_number']
  hist_bad_count = 0
  bad_hist_array = []
  for run in runs_list:
    run_row = Fdf.loc[Fdf['run_number'] == run].drop(columns=['run_number'])
    run_row = run_row.iloc[0].values
    if [x for x in run_row] > [y for y in Fthreshold_list]:
      hist_bad_count = hist_bad_count + 1
    bad_hist_array.append(hist_bad_count)
  #df_new = pd.DataFrame({'run':runs_list, 'hists_bad':bad_hist_array})
  return mean(bad_hist_array)


def main(infile, bad_runs_string):

  #all_files = glob.glob(os.path.join("./", "scores*.csv"))
  #df = pd.concat(map(pd.read_csv, all_files), axis=1)

  df = pd.read_csv(infile + "/bad_runs_sse_scores.csv")
  df = df.loc[:,~df.columns.duplicated()].copy()
  
  bad_runs = [int(run) for run in bad_runs_string.split(",")]

  df_pca = df.loc[df['algo'] == "pca"]
  df_ae = df.loc[df['algo'] == "ae"]

  for I in ["PCA", "AE"]:

    if I == "PCA": J = df_pca
    if I == "AE": J = df_ae

    hist_cols = [col for col in J.columns if 'L1T//Run summary' in col]
    hist_dict = {each_hist: "max" for each_hist in hist_cols}

    J = J.groupby('run_number')[hist_cols].agg(hist_dict).reset_index()

    #J = J.drop_duplicates(subset='run_number', keep="first")

    J['run_good'] = "Y"
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
    pct_95_good	= J_good[hist_cols].quantile(q=0.95).values
    pct_90_good	= J_good[hist_cols].quantile(q=0.90).values
    pct_80_good  = J_good[hist_cols].quantile(q=0.80).values
    pct_70_good  = J_good[hist_cols].quantile(q=0.70).values
    pct_60_good  = J_good[hist_cols].quantile(q=0.60).values
    pct_40_good  = J_good[hist_cols].quantile(q=0.40).values
    pct_20_good  = J_good[hist_cols].quantile(q=0.20).values

    med_bad = J_bad[hist_cols].median().values
    pct_99_bad = J_bad[hist_cols].quantile(q=0.99).values
    pct_95_bad = J_bad[hist_cols].quantile(q=0.95).values
    pct_90_bad	= J_bad[hist_cols].quantile(q=0.90).values
    pct_80_bad  = J_bad[hist_cols].quantile(q=0.80).values
    pct_70_bad  = J_bad[hist_cols].quantile(q=0.70).values
    pct_60_bad  = J_bad[hist_cols].quantile(q=0.60).values
    pct_40_bad  = J_bad[hist_cols].quantile(q=0.40).values
    pct_20_bad  = J_bad[hist_cols].quantile(q=0.20).values

    null_set_good = med_good*0
    med_0p3_good = med_good*0.3
    med_0p6_good = med_good*0.6
    med_0p9_good = med_good*0.9
    med_1p2_good = med_good*1.2
    med_1p5_good = med_good*1.5
    med_1p8_good = med_good*1.8

    null_set_bad = med_bad*0
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

    # Required as the AE isn't very sensitive at different thresholds and will cause the make_interp_spline function throw and error
    #[[good_bad_arrays[i] += 0.00001 if (good_bad_arrays[i] == good_bad_arrays[i-1]) for i in range(1, len(good_bad_arrays))] for good_bad_arrays in [mMF_ROC_good_Y,mMF_ROC_bad_X,pMF_ROC_good_Y,pMF_ROC_bad_X]]
    for good_bad_arrays in [mMF_ROC_good_Y,mMF_ROC_bad_X,pMF_ROC_good_Y,pMF_ROC_bad_X]:
      for i in range(len(good_bad_arrays)):
        good_bad_arrays[i] = good_bad_arrays[i] + 0.00001*i


    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xlabel('Average # flagged histograms / bad run')
    ax.set_ylabel('Average # flagged histograms / good run')
    ax.axis(xmin=0,xmax=len(hist_cols),ymin=0,ymax=len(hist_cols))
    #ax.scatter(mMF_ROC_bad_X,mMF_ROC_good_Y, marker='x', c='k',s=60)
    ax.plot(mMF_ROC_bad_X,mMF_ROC_good_Y, '-bo', mfc='r', mec='k', markersize=8, linewidth=1, label='SSE above median values')

    #ax.plot(pMF_ROC_bad_X,pMF_ROC_good_Y, 'kx', markersize=10, linewidth=0)
    ax.plot(pMF_ROC_bad_X,pMF_ROC_good_Y, '-g^', mfc='orange', mec='k', markersize=8, linewidth=1, label='SSE above quantile values')
    ax.set_xlim([0, max(mMF_ROC_bad_X+pMF_ROC_bad_X)])
    ax.set_ylim([0, max(mMF_ROC_good_Y+pMF_ROC_good_Y)])
    ax.axline((0, 0), slope=(max(mMF_ROC_good_Y+pMF_ROC_good_Y)/max(mMF_ROC_bad_X+pMF_ROC_bad_X)), linestyle='--',linewidth=0.5,color='gray')
    ax.annotate("Algorithm: " + I, xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')

    plt.legend(loc='lower right')

    '''

    x1new = np.linspace(min(mMF_ROC_bad_X), max(mMF_ROC_bad_X), 30)
    gf1g = make_interp_spline(mMF_ROC_bad_X,mMF_ROC_good_Y,k=3)
    y1new = gf1g(x1new)
    #ax.plot(x1new, y1new)

    ax.set_xlim([0, max(mMF_ROC_bad_X)+2])
    ax.set_ylim([0, max(mMF_ROC_good_Y)+2])

    '''

    plt.savefig(infile + "/MF_ROC_comparison_" + I + ".pdf")

if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("-i","--infile", type=str, required=True, help="Input directory where bad_runs_sse_scores.csv file is located (also output directory)")
  parser.add_argument("-br","--bad_runs", type=str, required=True, help="List of bad runs as determined by data certification reports or similar bodies (enter as comma separated numbers e.g. 356000,356002,...)")
  args = parser.parse_args()

  main(infile=args.infile, args.bad_runs)

