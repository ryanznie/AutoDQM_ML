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


def main(infile):

  #all_files = glob.glob(os.path.join("./", "scores*.csv"))
  #df = pd.concat(map(pd.read_csv, all_files), axis=1)

  df = pd.read_csv(infile + "/bad_runs_sse_scores.csv")
  df = df.loc[:,~df.columns.duplicated()].copy()
  #bad_runs = [355989,355990,355991,355992,355993,355994,355995,355996,355997,356001,356002,356003,356046,356047,356048,356073,356162,356163,356164,356165,356170,356174,356175,356309,356321,356371,356375,356377,356378,356382,356383,356384,356385,356426,356427,356428,356431,356432,356436,356466,356467,356468,356469,356470,356471,356472,356473,356474,356475,356476,356478,356479,356481,356488,356489,356523,356524,356525,356526,356527,356528,356529,356530,356568,356576,356577,356581,356582,356613,356614,356709,356719,356720,356721,356722,356788,356789,356810,356825,356902,356906,356943,356944,356945,356950,356997,357059,357070,357076,357077,357078,357096,357098,357100]
  bad_runs = [355865,356071,356074,356321,356375,356466,356467,356469,356472,356473,356476,356478,356481,356488,356489,356577,356581,356709,356719,356720,356721,356722,356788,356789,356815,356943,356944,356945,356997,356998,357077,357078,357100,357101,357103,357105,357110]

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
    med_good = []

    pct_99_bad = []
    pct_95_bad = []
    pct_90_bad = []
    med_bad = []

    med_good = J_good[hist_cols].median().values
    pct_99_good = J_good[hist_cols].quantile(q=0.99).values
    pct_95_good	= J_good[hist_cols].quantile(q=0.95).values
    pct_90_good	= J_good[hist_cols].quantile(q=0.90).values

    med_bad = J_bad[hist_cols].median().values
    pct_99_bad = J_bad[hist_cols].quantile(q=0.99).values
    pct_95_bad = J_bad[hist_cols].quantile(q=0.95).values
    pct_90_bad	= J_bad[hist_cols].quantile(q=0.90).values

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

    pMF_ROC_good_Y = sorted([nsg_hc,p99g_hc,p95g_hc,p90g_hc,0.0])
    pMF_ROC_bad_X = sorted([nsb_hc,p99b_hc,p95b_hc,p90b_hc,0.0])

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
  parser.add_argument("-i","--infile", type=str, help="Input directory where bad_runs_sse_scores.csv file is located (also output directory)")
  args = parser.parse_args()

  main(infile=args.infile)

