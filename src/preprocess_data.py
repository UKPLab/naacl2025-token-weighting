import os
import argparse
from data_preprocessing import DataPreprocessor
from logger_setup import initialize_logger


def preprocess_data(args):
    # Initialize logger
    initialize_logger()

    # Prepare paths
    PATH_RAW_DATA = args.raw_data_path
    tokenizer_name = args.tokenizer_name
    PATH_CHUNKED_DATA = os.path.join(PATH_RAW_DATA, f"preprocessed_{tokenizer_name}")

    # Instantiate DataPreprocessor
    data_preprocessor = DataPreprocessor(PATH_RAW_DATA=PATH_RAW_DATA, PATH_CHUNKED_DATA=PATH_CHUNKED_DATA,
                                         tokenizer=tokenizer_name)

    # Process dataset
    dataset = data_preprocessor.fetch_data_from_file()

    # Save dataset locally
    data_preprocessor.save_locally_hf(dataset)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and preprocess data.", add_help=True)
    parser.add_argument("--raw-data-path", type=str, default="../data",
                        help="Path to the raw data directory.")
    parser.add_argument("--tokenizer-name", type=str, default="llama3",
                        help="Name of the tokenizer to use during preprocessing.")

    args = parser.parse_args()
    preprocess_data(args)
