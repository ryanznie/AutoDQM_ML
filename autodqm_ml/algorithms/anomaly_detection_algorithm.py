import os
import pandas
import numpy
import awkward
import json
import csv

from autodqm_ml import utils
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.constants import kANOMALOUS, kGOOD

import logging
logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ["run_number", "label"] # columns which should always be read from input df

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name = "default", **kwargs):
        self.name = name

        self.data_is_loaded = False

        # These arguments will be overwritten if provided in kwargs
        self.output_dir = "output"
        #self.tag = ""
        #self.algorithm = ""
        self.histograms = {}
        self.input_file = None
        self.remove_low_stat = True

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        

    def load_data(self, file = None, histograms = {}, train_frac = 0.5, remove_low_stat = True):
        """
        Loads data from pickle file into ML class. 

        :param file: file containing data to be extracted. File output of fetch_data.py
        :type file: str
        :param histograms: names of histograms to be loaded. Must match histogram names used in fetch_data.py. Dictionary in the form {<histogram name> : {"normalize" : <bool>}}.
        :type histograms: dict. Default histograms = {}
        :param train_frac: fraction of dataset to be kept as training data. Must be between 0 and 1. 
        :type train_frac: float. Default train_frac = 0.0
        :param remove_low_stat: removes runs containing histograms with low stats. Low stat threshold is 1000 events.
        :type remove_low_stat: bool. remove_low_stat = False
        """
        if self.data_is_loaded:
            return

        if file is not None:
            if self.input_file is not None:
                if not (file == self.input_file):
                    logger.warning("[AnomalyDetectionAlgorithm : load_data] Data file was previously set as '%s', but will be changed to '%s'." % (self.input_file, file)) 
                    self.input_file = file
            else:
                self.input_file = file

        if self.input_file is None:
            logger.exception("[AnomalyDetectionAlgorithm : load_data] No data file was provided to load_data and no data file was previously set for this instance, please specify the input data file.")
            raise ValueError()

        if not os.path.exists(self.input_file):
            self.input_file = utils.expand_path(self.input_file)

        if histograms:
            self.histograms = histograms
        self.histogram_name_map = {} # we replace "/" and spaces in input histogram names to play nicely with other packages, this map lets you go back and forth between them

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loading training data from file '%s'" % (self.input_file))

        # Load dataframe
        df = awkward.from_parquet(self.input_file)
       
        # Set helpful metadata
        for histogram, histogram_info in self.histograms.items():
            self.histograms[histogram]["name"] = histogram.replace("/", "").replace(" ","")
            self.histogram_name_map[self.histograms[histogram]["name"]] = histogram

            a = awkward.to_numpy(df[histogram][0])
            self.histograms[histogram]["shape"] = a.shape
            self.histograms[histogram]["n_dim"] = len(a.shape)
            self.histograms[histogram]["n_bins"] = 1
            for x in a.shape:
                self.histograms[histogram]["n_bins"] *= x 

        for histogram, histogram_info in self.histograms.items():
            # Normalize (if specified in histograms dict)
            if "normalize" in histogram_info.keys():
                if histogram_info["normalize"]:
                    sum = awkward.sum(df[histogram], axis = -1)
                    if histogram_info["n_dim"] == 2:
                        sum = awkward.sum(sum, axis = -1)

                    logger.debug("[anomaly_detection_algorithm : load_data] Scaling all entries in histogram '%s' by the sum of total entries." % histogram)
                    df[histogram] = df[histogram] * (1. / sum) 
        self.n_train = awkward.sum(df.label == 0)
        self.n_bad_runs = awkward.sum(df.label != 0)
        self.df = df
        self.n_histograms = len(list(self.histograms.keys()))

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loaded data for %d histograms with %d events in training set, excluding the %d bad runs." % (self.n_histograms, self.n_train, self.n_bad_runs))

        self.data_is_loaded = True


    def add_prediction(self, histogram, score, reconstructed_hist = None):
        """
        Add fields to the df containing the score for this algorithm (p-value/pull-value for statistical tests, sse for ML algorithms)
        and the reconstructed histograms (for ML algorithms only).
        """
        self.df[histogram + "_score_" + self.tag] = score
        if reconstructed_hist is not None:
            self.df[histogram + "_reco_" + self.tag] = reconstructed_hist


    def save(self, histograms = {}, tag = "", algorithm = ""):
        """

        """
        os.system("mkdir -p %s" % self.output_dir)

        self.output_file = "%s/%s_%s_runs_and_sse_scores.csv" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""), tag)

        output_parquet = "%s/%s.parquet" % (self.output_dir, self.input_file.split("/")[-1].replace(".parquet", ""))
        awkward.to_parquet(self.df, output_parquet)
        logger.info("[AnomalyDetectionAlgorithm : save] Saving output for plot assessment '%s'." % (output_parquet))

        columns_to_remove = list(histograms.keys())
        reco_columns = [hist_name + "_reco_" + tag for hist_name in columns_to_remove]
        score_columns = [hist_name + "_score_" + tag for hist_name in columns_to_remove]
        rename_columns_dict = {old_col : new_col for old_col, new_col in zip(score_columns, columns_to_remove)}
        columns_to_remove = columns_to_remove + reco_columns
        
        filtered_fields = {field: self.df[field] for field in self.df.fields if field not in columns_to_remove}
        self.df = awkward.zip(filtered_fields)

        #filtered_fields_for_assess_plots = {field: self.df[field] for field in self.df.fields if field not in score_columns}
        #new_df = awkward.zip(filtered_fields_for_assess_plots)

        for old_name, new_name in rename_columns_dict.items():
            self.df = awkward.with_field(self.df, self.df[old_name], new_name)

        if algorithm.lower() in ["ae","autoencoder"]:
            algo_name = "ae"
        elif algorithm.lower() == "pca":
            algo_name = "pca"

        if algo_name is not None:
            new_field = awkward.Array([algo_name] * len(self.df))
            self.df = awkward.with_field(self.df, new_field, "algo")

        list_of_dicts = awkward.to_list(self.df)
        fieldnames = list_of_dicts[0].keys()

        with open(self.output_file, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(list_of_dicts)

        self.config_file = "%s/%s_%s.json" % (self.output_dir, self.name, self.tag)
        logger.info("[AnomalyDetectionAlgorithm : save] Saving output for large data SSE assessment '%s'." % (self.output_file))
        config = {}
        for k,v in vars(self).items():
            if utils.is_json_serializable(v):
                config[k] = v

        logger.info("[AnomalyDetectionAlgorithm : save] Saving AnomalyDetectionAlgorithm config to file '%s'." % (self.config_file))
        with open(self.config_file, "w") as f_out:
            json.dump(config, f_out, sort_keys = True, indent = 4)
