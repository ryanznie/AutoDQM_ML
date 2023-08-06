from autodqm_ml.data_prep.data_fetcher import DataFetcher
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")
fetcher = DataFetcher(
        #tag = "l1t_calol2_example", # will identify output files
        output_dir = "l1t_calol2_example_020322",
        contents = "metadata/cl2_contents_example.json",
        datasets = "metadata/2017_2018_l1t_dataset_example.json",
        short = True
)

fetcher.run()

