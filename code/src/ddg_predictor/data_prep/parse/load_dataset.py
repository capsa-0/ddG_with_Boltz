import argparse
from loaders import Loader1


def load_dataset(dataset_type: str, raw_path: str, output_dir: str | None = None, **kwargs) -> object:
    """
    Args:
        dataset_type (str): Identifier for the dataset type to load (e.g., '1').
        raw_path (str): Path to the raw input file.
        output_dir (str | None, optional): Directory to save processed output. Defaults to None.
        **kwargs: Additional keyword arguments passed to the loader class.
    Returns:
        object: An instance of the loader class corresponding to the dataset type.
    """
    # Map dataset type to the corresponding loader class
    loader_map = {
        '1': Loader1,
    }
    
    if dataset_type not in loader_map:
        raise ValueError(f"Dataset type '{dataset_type}' not supported.")
    
    loader_class = loader_map[dataset_type]
    return loader_class(raw_path, output_dir, **kwargs)


def process_and_save(dataset_type: str, raw_path: str, output_dir: str | None = None, **kwargs) -> object:
    """
    Args:
        dataset_type (str): Identifier for the dataset type to load.
        raw_path (str): Path to the raw input file.
        output_dir (str | None, optional): Directory to save processed output. Defaults to None.
        **kwargs: Additional keyword arguments passed to the loader.
    Returns:
        object: Loader instance after processing and saving outputs.
    """
    # Load dataset using the appropriate loader
    loader = load_dataset(dataset_type, raw_path, output_dir, **kwargs)
    
    # Process and save results
    loader.process()
    loader.save_outputs()
    
    return loader


def main() -> None:
    """
    Main entry point.
    Parses CLI arguments, loads the dataset, processes it, and saves outputs.
    """
    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', required=True, help='Dataset type identifier')
    parser.add_argument('--raw_path', required=True, help='Path to raw input file')
    parser.add_argument('--output_dir', required=False, help='Directory for saving processed output')
    
    args = parser.parse_args()
    
    loader = process_and_save(
        dataset_type=args.dataset_type,
        raw_path=args.raw_path,
        output_dir=args.output_dir
    )
    # At this point, `loader` contains the processed dataset instance


if __name__ == '__main__':
    main()
