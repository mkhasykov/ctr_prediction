import os
import json
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utls import check_present_data
import prepare_data as prd
from xxftrl import ftrl_proximal


DEFAULT_FOLDER_PATH = './data'


def setup_parser(parser):
    
    parser.add_argument(
            "-f", "--folder", default=DEFAULT_FOLDER_PATH, dest="folder",
            help="path to folder where parsed data should be saved",
    )
    parser.add_argument(
            "-o", "--output", required=True, dest="output",
            help="output file (json) path",
            )
    parser.add_argument(
            "-n", "--num_epoch", type=int, required=False, default=5, dest="num_epoch",
            help="number of train epochs. Default value is 5",
            )
    parser.add_argument(
            "-d", "--dim_hash", type=int, required=False, default=18, dest="d",
            help="parameter determines number of OHE features as 2**d. Default value is 18." + 
            "For sake of results adequacy, values less than 15 are prohibited.",
            )
    parser.add_argument(
            "-alpha", "--alpha", type=float, required=False, default=0.1, dest="alpha", 
            help="ftrl alpha parameter. Default value is .1",
            )
    parser.add_argument(
            "-beta", "--beta", type=float, required=False, default=1.0, dest="beta", 
            help="ftrl beta parameter. Default value is 1.",
            )
    parser.add_argument(
            "-l1", "--l1", type=float, required=False, default=0.0, dest="l1", 
            help="L1 regularization parameter. Default value is 0.",
            )
    parser.add_argument(
            "-l2", "--l2", type=float, required=False, default=0.0, dest="l2", 
            help="L2 regularization parameter. Default value is 0.",
            )
    
    
def process_data(input_args):
    start_date, end_date = check_present_data(input_args.folder)
    
    print('%s\tParsing ids of bids...' % (datetime.datetime.now()))
    print('%s\t\tParsing req_ids from ClickHouse...' % (datetime.datetime.now()))
    prd.get_reqs_ch(start_date, end_date, input_args.folder)
    print('%s\t\tParsing bid_ids from Yandex.Metrika...' % (datetime.datetime.now()))
    prd.parse_ya_metrika(start_date, end_date, input_args.folder)
    print('%s\t\tParsing bid_ids from Amplitude...' % (datetime.datetime.now()))
    prd.parse_amplitude(start_date, end_date, input_args.folder)
    print('%s\t\tParsing req_ids by bid_ids from Yandex.Metrika and Amplitude...' % (datetime.datetime.now()))
    prd.get_reqs_metrika_ampl(input_args.folder)
    prd.unite_reqs(input_args.folder)
    
    print('%s\tParsing features for bids...' % (datetime.datetime.now()))
    prd.parse_raw_features(input_args.folder)
    prd.parse_blacklist(input_args.folder)
    
    print('%s\tPreparing train dataset...' % (datetime.datetime.now()))
    prd.prepare_train_dataset(input_args.folder)
    print('%s\tTrain dataset is ready.' % (datetime.datetime.now()))
    

def build_model(input_args):
    
    print('%s\tModel learning started...' % (datetime.datetime.now()))
    D = 2**input_args.d # number of hash features / weights
    learner = ftrl_proximal(
            input_args.alpha, 
            input_args.beta, 
            input_args.l1, 
            input_args.l2,
            D, 
            interaction=False,
            )
    learner.fit(os.path.join(input_args.folder, 'train.csv'), input_args.num_epoch)
    weights = learner.output_weigts()
    
    # save weights
    with open(input_args.output, 'w') as fp:
        json.dump(weights, fp)
    
    print('%s\tModel updating finished.' % (datetime.datetime.now()))
    

def check_arguments(input_args):
    assert input_args.num_epoch > 0, "Allowed only positive number of train epochs"
    assert input_args.d >= 15, "Allowed hash feature space dimensionality not less than 2**15"
    assert input_args.alpha >= 0, "Allowed only non-negative ftrl alpha parameter"
    assert input_args.beta >= 0, "Allowed only non-negative ftrl beta parameter"
    assert input_args.l1 >= 0, "Allowed only non-negative l1 regularization parameter"
    assert input_args.l2 >= 0, "Allowed only non-negative l2 regularization parameter"


def main():
    """Python CLI interface for Building Model for CTR Prediction on Transit Page
    Usage: python main.py -f ./data_refactored -o output.json -n 7 -d 18 -alpha .1 -beta 1. -l1 5. -l2 5.
    
    For more information see:
    - https://git.snatchdev.com/kimberlite/backend/wikis/%D0%92%D0%B5%D1%80%D0%BE%D1%8F%D1%82%D0%BD%D0%BE%D1%81%D1%82%D1%8C-%D0%BA%D0%BE%D0%BD%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D0%B8-%D0%BD%D0%B0-%D1%82%D1%80%D0%B0%D0%BD%D0%B7%D0%B8%D1%82%D0%BA%D0%B5
    """
    
    parser = ArgumentParser(
        prog="transit-ctr-model-building",
        description="Build Model for Click Probability Prediction on the Transit Page: parse data, collect train dataset, train model, dump weights",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    
    args = parser.parse_args()
    check_arguments(args)
    process_data(args)
    build_model(args)
    

if __name__ == "__main__":
    main()
