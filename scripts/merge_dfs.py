import pandas as pd
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
    run_row = df.loc[df['run_number'] == run]
    if [x for x in run_row] > [y for y in Fthreshold_list]:
      hist_bad_count = hist_bad_count + 1
    bad_hist_array.append(hist_bad_count)
  #df_new = pd.DataFrame({'run':runs_list, 'hists_bad':bad_hist_array})
  return bad_hist_array.mean()

all_files = glob.glob(os.path.join("./", "scores*.csv"))

df = pd.concat(map(pd.read_csv, all_files), axis=1)
df = df.loc[:,~df.columns.duplicated()].copy()
df['run_good'] = "Y"
bad_runs = [355989,355990,355991,355992,355993,355994,355995,355996,355997,356001,356002,356003,356046,356047,356048,356073,356162,356163,356164,356165,356170,356174,356175,356309,356321,356371,356375,356377,356378,356382,356383,356384,356385,356426,356427,356428,356431,356432,356436,356466,356467,356468,356469,356470,356471,356472,356473,356474,356475,356476,356478,356479,356481,356488,356489,356523,356524,356525,356526,356527,356528,356529,356530,356568,356576,356577,356581,356582,356613,356614,356709,356719,356720,356721,356722,356788,356789,356810,356825,356902,356906,356943,356944,356945,356950,356997,357059,357070,357076,357077,357078,357096,357098,357100]

df = df.drop_duplicates(subset='run_number', keep="first")

df.loc[df['run_number'].isin(bad_runs), 'run_good'] = "N"
df = df.sort_values(['run_good'])
#col = df.pop('run_good')
#df = df.insert(1, col.name, col)
print(df)
df = df[['run_number','run_good'] + [col for col in df.columns if (col != 'run_number')&(col != 'run_good')]]

print("Dataframe successfully created")

df_pca = df.loc[df['algo'] == "pca"]
df_ae = df.loc[df['algo'] == "ae"]

for I in ["PCA", "AE"]:

  if I == "PCA": J = df_pca
  if I == "AE": J = df_ae

  J_good = J.loc[J['run_good'] == "Y"]
  J_bad = J.loc[J['run_good'] == "N"]

    pct_99_good = []
  pct_95_good = []
  pct_90_good = []
  med_good = []

  pct_99_bad = []
  pct_95_bad = []
  pct_90_bad = []
  med_bad = []

  hist_cols = [col for col in J.columns if 'L1T//Run summary' in col]
  for k in hist_cols:
    pct_99_good.append(J[k].quartile(q=0.99))
    pct_95_good.append(J[k].quartile(q=0.95))
    pct_90_good.append(J[k].quartile(q=0.90))
    med_good.append(J[k].median_good)

    pct_99_bad.append(J[k].quartile(q=0.99))
    pct_95_bad.append(J[k].quartile(q=0.95))
    pct_90_bad.append(J[k].quartile(q=0.90))
    med_bad.append(J[k].median_bad)

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

  p99g_hc = count_hists_above(J_good, pct_99_good)
  p95g_hc = count_hists_above(J_good, pct_95_good)
  p90g_hc = count_hists_above(J_good, pct_90_good)
  nsg_hc = count_hists_above(J_good, null_set_good)
  m03g_hc = = count_hists_above(J_good, med_0p3_good)
  m06g_hc = = count_hists_above(J_good, med_0p6_good)
  m09g_hc = = count_hists_above(J_good, med_0p9_good)
  m10g_hc = = count_hists_above(J_good, med_good)
  m12g_hc = = count_hists_above(J_good, med_1p2_good)
  m15g_hc = = count_hists_above(J_good, med_1p5_good)
  m18g_hc = = count_hists_above(J_good, med_1p8_good)

  p99b_hc = count_hists_above(J_bad, pct_99_bad)
  p95b_hc = count_hists_above(J_bad, pct_95_bad)
  p90b_hc = count_hists_above(J_bad, pct_90_bad)
  nsb_hc = count_hists_above(J_bad, null_set_bad)
  m03b_hc = = count_hists_above(J_bad, med_0p3_bad)
  m06b_hc = = count_hists_above(J_bad, med_0p6_bad)
  m09b_hc = = count_hists_above(J_bad, med_0p9_bad)
  m10b_hc = = count_hists_above(J_bad, med_bad)
  m12b_hc = = count_hists_above(J_bad, med_1p2_bad)
  m15b_hc = = count_hists_above(J_bad, med_1p5_bad)
  m18b_hc = = count_hists_above(J_bad, med_1p8_bad)

  mMF_ROC_good_Y = [nsg_hc,m03g_hc,m06g_hc,m09g_hc,m10g_hc,m12g_hc,m15g_hc,m18g_hc]
  mMF_ROC_bad_X = [nsb_hc,m03b_hc,m06b_hc,m09b_hc,m10b_hc,m12b_hc,m15b_hc,m18b_hc]

  pMF_ROC_good_Y = [nsg_hc,p99g_hc,p95g_hc,p90g_hc]
  pMF_ROC_bad_X = [nsb_hc,p99b_hc,p95b_hc,p90b_hc]

  fig = plt.figure(figsize=(15, 6))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(121)

  ax1.set_xlabel('Average # flagged histograms / bad run')
  ax1.set_ylabel('Average # flagged histograms / good run')
  ax1.axis(xmin=0,xmax=len(hist_cols),ymin=0,ymax=len(hist_cols))
  ax1.scatter(mMF_ROC_bad_X,mMF_ROC_good_Y, marker='.', c='k',s='10')
  
  x1new = np.linspace(mMF_ROC_bad_X.min(), mMF_ROC_bad_X.max(), 30)
  gf1g = make_interp_spline(mMF_ROC_bad_X,mMF_ROC_good_Y,k=3)
  y1new = gf1g(x1new)
  ax1.plot(x1new, y1new)

  ax2.set_xlabel('Average # flagged histograms / bad run')
  ax2.set_ylabel('Average # flagged histograms / good run')
  ax2.axis(xmin=0,xmax=len(hist_cols),ymin=0,ymax=len(hist_cols))
  ax2.scatter(pMF_ROC_bad_X,pMF_ROC_good_Y, marker='.', c='k',s='10')

  x2new = np.linspace(pMF_ROC_bad_X.min(), pMF_ROC_bad_X.max(), 30)
  gf2g = make_interp_spline(pMF_ROC_bad_X,pMF_ROC_good_Y,k=3)
  y2new = gf2g(x2new)
  ax2.plot(x2new, y2new)

  plt.savefig("./MF_ROC_comparison.pdf")
