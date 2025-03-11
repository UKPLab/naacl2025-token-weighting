import os
import argparse
from data_preprocessing import DataPreprocessor
from logger_setup import initialize_logger
from run_settings import get_settings


def main():
    # Initialize logger
    initialize_logger()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run data preprocessing with given parameters.", add_help=True)
    parser.add_argument("--tokenizer_name", type=str, default="llama3", help="Name of the tokenizer (e.g., 'llama3').")
    parser.add_argument("--no_devices", type=int, required=True, help="Number of GPUs used to score the data.")
    parser.add_argument("--path_raw_data", type=str, default="../data",
                        help="Path to the raw data (e.g., '/local/path/to/pg19').")

    # Parse arguments
    args = parser.parse_args()

    # Extract parameters
    tokenizer_name = args.tokenizer_name
    no_devices = args.no_devices
    PATH_RAW_DATA = args.path_raw_data

    # Construct paths based on arguments
    PATH_CHUNKED_DATA = os.path.join(PATH_RAW_DATA, f"preprocessed_{tokenizer_name}")
    PATH_FROZEN_CACHE = os.path.join(PATH_CHUNKED_DATA, f"precomputed_weights_3.0_8B")

    # Get settings for the specified mode
    settings = get_settings(mode=f"{tokenizer_name}_32k_dense_precompute_weights")
    data_preprocessor = DataPreprocessor(settings=settings)

    # Run data preprocessing
    data_preprocessor.frozen_collect_and_make_hf_dataset(
        precomputed_weights_path=PATH_FROZEN_CACHE,
        folder_non_frozen=PATH_CHUNKED_DATA,
        no_devices=no_devices
    )
    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
