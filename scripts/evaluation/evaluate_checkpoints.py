#!/usr/bin/env python3
"""
Checkpoint Evaluation Script for MSSPlace Models

This script evaluates pre-trained MSSPlace model checkpoints on Oxford and NCLT
datasets to verify the results reported in our paper. It provides a clean interface
for testing different model variants with comprehensive logging and error handling.

Note: This script evaluates released checkpoints only. For full experimental
reproduction including training from scratch, refer to the training scripts
(not publicly released).

Key Features:
- Supports text-enabled datasets for all model variants
- Dynamic sensor setup configuration per model type
- Loguru-based logging with colored output
- Automatic sensor setup selection based on model name

Usage:
    python evaluate_checkpoints.py --dataset oxford --model mssplace-li
    python evaluate_checkpoints.py --dataset nclt --model mssplace-lis --batch-size 16
    python evaluate_checkpoints.py --dataset oxford --model mssplace-lit --verbose

Requirements:
    - PyTorch 2.1+
    - Python 3.10+
    - Hydra/OmegaConf for configuration management
    - Custom OPR (Open Place Recognition) library
    - loguru for enhanced logging
    - Custom datasets module with text support

Author: Generated from what_is_in_checkpoint.ipynb
Date: May 23, 2025
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from loguru import logger

# Import custom modules from the OPR library
from opr.testers.place_recognition.model import ModelTester, RetrievalResultsCollection

# Import text-enabled datasets from the installed mssplace package
from mssplace.datasets import NCLTDatasetWithText, OxfordDatasetWithText


# Configuration constants following the original notebook structure
DATASET_CHOICES = ["oxford", "nclt"]

MODEL_CHOICES = [
    "mssplace-li",
    "mssplace-lis",
    "mssplace-lit",
    "mssplace-list",
]

MODEL_CONFIG_NAMES = {
    "mssplace-li": "mssplace-li.yaml",
    "mssplace-lis": "mssplace-lis.yaml",
    "mssplace-lit": "mssplace-lit.yaml",
    "mssplace-list": "mssplace-list.yaml",
}

CHECKPOINT_NAMES = {
    "oxford": {
        "mssplace-li": "oxford_mssplace_li.pth",
        "mssplace-lis": "oxford_mssplace_lis.pth",
        "mssplace-lit": "oxford_mssplace_lit.pth",
        "mssplace-list": "oxford_mssplace_list.pth",
    },
    "nclt": {
        "mssplace-li": "nclt_mssplace_li.pth",
        "mssplace-lis": "nclt_mssplace_lis.pth",
        "mssplace-lit": "nclt_mssplace_lit.pth",
        "mssplace-list": "nclt_mssplace_list.pth",
    },
}

SENSOR_SETUPS = {
    "oxford": {
        "mssplace-li": [
            "pointcloud_lidar",
            "image_stereo_centre",
            "image_mono_left",
            "image_mono_rear",
            "image_mono_right"
        ],
        "mssplace-lis": [
            "pointcloud_lidar",
            "image_stereo_centre",
            "image_mono_left",
            "image_mono_rear",
            "image_mono_right",
            "mask_stereo_centre",
            "mask_mono_left",
            "mask_mono_rear",
            "mask_mono_right",
        ],
        "mssplace-lit": [
            "pointcloud_lidar",
            "image_stereo_centre",
            "image_mono_left",
            "image_mono_rear",
            "image_mono_right",
            "text_stereo_centre",
            "text_mono_left",
            "text_mono_rear",
            "text_mono_right",
        ],
        "mssplace-list": [
            "pointcloud_lidar",
            "image_stereo_centre",
            "image_mono_left",
            "image_mono_rear",
            "image_mono_right",
            "mask_stereo_centre",
            "mask_mono_left",
            "mask_mono_rear",
            "mask_mono_right",
            "text_stereo_centre",
            "text_mono_left",
            "text_mono_rear",
            "text_mono_right",
        ],
    },
    "nclt": {
        "mssplace-li": [
            "pointcloud_lidar",
            "image_Cam1",
            "image_Cam2",
            "image_Cam3",
            "image_Cam4",
            "image_Cam5"
        ],
        "mssplace-lis": [
            "pointcloud_lidar",
            "image_Cam1",
            "image_Cam2",
            "image_Cam3",
            "image_Cam4",
            "image_Cam5",
            "mask_Cam1",
            "mask_Cam2",
            "mask_Cam3",
            "mask_Cam4",
            "mask_Cam5"
        ],
        "mssplace-lit": [
            "pointcloud_lidar",
            "image_Cam1",
            "image_Cam2",
            "image_Cam3",
            "image_Cam4",
            "image_Cam5",
            "text_Cam1",
            "text_Cam2",
            "text_Cam3",
            "text_Cam4",
            "text_Cam5"
        ],
        "mssplace-list": [
            "pointcloud_lidar",
            "image_Cam1",
            "image_Cam2",
            "image_Cam3",
            "image_Cam4",
            "image_Cam5",
            "mask_Cam1",
            "mask_Cam2",
            "mask_Cam3",
            "mask_Cam4",
            "mask_Cam5",
            "text_Cam1",
            "text_Cam2",
            "text_Cam3",
            "text_Cam4",
            "text_Cam5"
        ],
    }
}


def setup_logging(verbose: bool = False) -> None:
    """
    Configure loguru logging for the script.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO
    """
    # Remove default logger
    logger.remove()

    # Configure loguru with appropriate level
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def validate_paths(datasets_dir: Path, checkpoint_dir: Path, config_dir: Path) -> None:
    """
    Validate that all required directories exist.

    Args:
        datasets_dir: Path to datasets directory
        checkpoint_dir: Path to checkpoints directory
        config_dir: Path to configs directory

    Raises:
        FileNotFoundError: If any required directory is missing
    """
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory does not exist: {datasets_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoint_dir}")
    if not config_dir.exists():
        raise FileNotFoundError(f"Configs directory does not exist: {config_dir}")


def get_dataset_path(dataset_name: str, datasets_dir: Path) -> Path:
    """
    Get the specific dataset path based on dataset name.

    Args:
        dataset_name: Name of the dataset ('oxford' or 'nclt')
        datasets_dir: Base datasets directory path

    Returns:
        Path to the specific dataset directory
    """
    if dataset_name == "oxford":
        return datasets_dir / "pnvlad_oxford_robotcar"
    elif dataset_name == "nclt":
        return datasets_dir / "NCLT_preprocessed"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> dict:
    """
    Load model checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to

    Returns:
        Dictionary containing model state dict

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    logger.info(f"Checkpoint loaded with {len(checkpoint.keys())} parameter groups")
    return checkpoint


def load_model_config(config_path: Path) -> OmegaConf:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        OmegaConf configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    logger.info(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    # Log config details for reproducibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    logger.debug(f"Model configuration: {config_dict}")

    return config


def create_dataset(dataset_name: str, data_dir: Path, sensor_setup: list[str]) -> torch.utils.data.Dataset:
    """
    Create dataset instance based on dataset name with specified sensor setup.

    Always uses text-enabled dataset classes (*DatasetWithText) to ensure
    compatibility with all model variants, including text-based models.

    Args:
        dataset_name: Name of dataset ('oxford' or 'nclt')
        data_dir: Path to dataset directory
        sensor_setup: List of sensors/modalities to load

    Returns:
        Dataset instance ready for testing (text-enabled)
    """
    logger.info(f"Creating {dataset_name} dataset from: {data_dir}")
    logger.debug(f"Sensor setup: {sensor_setup}")

    if dataset_name == "oxford":
        dataset = OxfordDatasetWithText(
            dataset_root=data_dir,
            subset="test",
            data_to_load=sensor_setup,
            pointcloud_quantization_size=0.01,
        )
    elif dataset_name == "nclt":
        dataset = NCLTDatasetWithText(
            dataset_root=data_dir,
            subset="test",
            data_to_load=sensor_setup,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Dataset created with {len(dataset)} samples")
    return dataset


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    distance_threshold: float = 25.0,
    memory_batch_size: int | None = None,
    verbose: bool = True,
) -> tuple[dict[str, float], RetrievalResultsCollection]:
    """
    Evaluate model performance using comprehensive ModelTester.

    This function leverages the advanced ModelTester class to provide detailed
    place recognition analysis beyond simple aggregate metrics. It supports
    memory-efficient evaluation and returns comprehensive results suitable
    for research analysis and reproducibility.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for test data
        device: Device to run evaluation on
        distance_threshold: Distance threshold for positive matches (meters)
        memory_batch_size: If specified, compute distance matrix in batches
            to reduce peak memory usage. Useful for large datasets.
        verbose: Whether to show detailed progress information

    Returns:
        Tuple containing:
        - dict: Aggregate metrics (recall_at_n, recall_at_one_percent, etc.)
        - RetrievalResultsCollection: Detailed per-query results for analysis

    Note:
        The memory_batch_size parameter trades computation speed for memory
        efficiency. For datasets with >10k samples, consider using batch
        sizes of 1000-5000 depending on available RAM.
    """
    logger.info("Starting comprehensive model evaluation with ModelTester...")

    # Initialize ModelTester with memory-efficient settings
    tester = ModelTester(
        model=model,
        dataloader=dataloader,
        dist_thresh=distance_threshold,
        at_n=25,  # Standard benchmark value
        device=device,
        verbose=verbose,
        batch_size=memory_batch_size,  # Enable memory-efficient computation
    )

    # Run comprehensive evaluation
    results_collection = tester.run()

    # Extract aggregate metrics for backward compatibility
    aggregate_metrics = results_collection.aggregate_metrics()

    # Convert to format expected by existing display logic
    recall_at_n_array = aggregate_metrics["recall_at_n"]
    recall_at_one_percent = aggregate_metrics["recall_at_one_percent"]

    # For top1_distance, use the aggregate value or compute if None
    top1_distance = aggregate_metrics.get("top1_distance")
    if top1_distance is None:
        # Fallback: compute mean embedding distance of correct top-1 matches
        top1_distances = []
        for result in results_collection.results:
            if result.queries_with_matches > 0 and result.top1_distance is not None:
                top1_distances.append(result.top1_distance)
        top1_distance = sum(top1_distances) / len(top1_distances) if top1_distances else 0.0

    # Create backward-compatible metrics dict
    backward_compatible_metrics = {
        "recall_at_n": recall_at_n_array,
        "recall_at_one_percent": recall_at_one_percent,
        "mean_top1_descriptor_distance": top1_distance,
    }

    logger.info("Comprehensive model evaluation completed")
    logger.info(f"Processed {results_collection.num_pairs} track pairs with "
                f"{results_collection.num_queries} total queries")

    return backward_compatible_metrics, results_collection


def save_evaluation_results(
    results_collection: RetrievalResultsCollection,
    dataset_name: str,
    model_name: str,
    results_dir: Path,
    additional_metadata: dict | None = None,
) -> Path:
    """
    Save detailed evaluation results to disk for later analysis.

    Creates a structured filename and saves both the raw results collection
    and additional metadata for research reproducibility.

    Args:
        results_collection: Detailed results from ModelTester
        dataset_name: Name of the evaluated dataset
        model_name: Name of the evaluated model
        results_dir: Directory to save results
        additional_metadata: Optional dict with extra information to save

    Returns:
        Path to the saved results file

    Note:
        Results are saved in JSON format with timestamp for uniqueness.
        The file includes both detailed per-query results and aggregate metrics.
    """
    from datetime import datetime

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{model_name}_results_{timestamp}.json"
    results_path = results_dir / filename

    logger.info(f"Saving detailed evaluation results to: {results_path}")

    # Save the results collection (includes built-in JSON serialization)
    results_collection.save(str(results_path))

    # If metadata provided, save it alongside
    if additional_metadata:
        metadata_path = results_dir / f"{dataset_name}_{model_name}_metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(additional_metadata, f, indent=2)
        logger.info(f"Saved evaluation metadata to: {metadata_path}")

    return results_path


def format_percentage(value: float) -> str:
    """
    Format a decimal value as a percentage with 2 decimal places (truncated, not rounded).

    Args:
        value: Decimal value between 0 and 1

    Returns:
        Formatted percentage string
    """
    # Truncate to 2 decimal places without rounding, as in original notebook
    integer_part = int(value * 100)
    decimal_part = int((value * 100) % 1 * 100)
    return f"{integer_part}.{decimal_part:02d}%"


def main() -> None:
    """
    Main function that orchestrates the checkpoint testing process.
    """
    parser = argparse.ArgumentParser(
        description="Test MSSPlace model checkpoints on Oxford and NCLT datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        required=True,
        help="Dataset to test on"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        required=True,
        help="Model variant to test"
    )

    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("/home/docker_mssplace/Datasets"),
        help="Path to datasets directory"
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "checkpoints",
        help="Path to checkpoints directory"
    )

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "configs" / "model",
        help="Path to model configs directory"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=25.0,
        help="Distance threshold for positive matches (meters)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--memory-batch-size",
        type=int,
        default=None,
        help="Batch size for memory-efficient distance computation (reduces memory usage)"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed evaluation results to JSON file for later analysis"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./evaluation_results"),
        help="Directory to save detailed evaluation results"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info(f"Starting checkpoint testing for {args.model} on {args.dataset}")

    try:
        # Validate all required paths exist
        validate_paths(args.datasets_dir, args.checkpoints_dir, args.configs_dir)

        # Get specific paths for this dataset/model combination
        checkpoint_name = CHECKPOINT_NAMES[args.dataset][args.model]
        config_name = MODEL_CONFIG_NAMES[args.model]
        sensor_setup = SENSOR_SETUPS[args.dataset][args.model]

        checkpoint_path = args.checkpoints_dir / checkpoint_name
        config_path = args.configs_dir / config_name
        data_dir = get_dataset_path(args.dataset, args.datasets_dir)

        # Load checkpoint and config
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        config = load_model_config(config_path)

        # Initialize model and load weights
        logger.info("Initializing model...")
        model = instantiate(config)
        model.load_state_dict(checkpoint, strict=True)

        num_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully with {num_parameters:,} parameters")

        # Create dataset and dataloader
        dataset = create_dataset(args.dataset, data_dir, sensor_setup)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

        # Evaluate model using comprehensive ModelTester
        metrics, results_collection = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=args.device,
            distance_threshold=args.distance_threshold,
            memory_batch_size=args.memory_batch_size,
            verbose=args.verbose,
        )

        # Extract metrics for display (backward compatibility)
        recall_at_n = metrics["recall_at_n"]
        recall_at_one_percent = metrics["recall_at_one_percent"]
        mean_top1_descriptor_distance = metrics["mean_top1_descriptor_distance"]

        # Optionally save detailed results for research analysis
        if args.save_results:
            evaluation_metadata = {
                "dataset": args.dataset,
                "model": args.model,
                "device": args.device,
                "distance_threshold": args.distance_threshold,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "memory_batch_size": args.memory_batch_size,
                "model_parameters": num_parameters,
                "dataset_size": len(dataset),
                "evaluation_timestamp": datetime.now().isoformat(),
            }

            results_path = save_evaluation_results(
                results_collection=results_collection,
                dataset_name=args.dataset,
                model_name=args.model,
                results_dir=args.results_dir,
                additional_metadata=evaluation_metadata,
            )

            logger.info(f"Detailed results saved for future analysis: {results_path}")

        # Display results with enhanced information
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        print(f"Distance threshold: {args.distance_threshold}m")
        if args.memory_batch_size:
            print(f"Memory batch size: {args.memory_batch_size} (memory-efficient mode)")
        print("-"*60)

        # Traditional metrics (backward compatibility)
        print("PLACE RECOGNITION METRICS:")
        print(f"  AR@1  = {format_percentage(recall_at_n[0])}")
        print(f"  AR@1% = {format_percentage(recall_at_one_percent)}")
        print(f"  Mean top-1 descriptor distance: {mean_top1_descriptor_distance:.6f}")

        # Enhanced insights from detailed analysis
        print("\nDETAILED ANALYSIS:")
        aggregate_metrics = results_collection.aggregate_metrics()
        print(f"  Total track pairs evaluated: {results_collection.num_pairs}")
        print(f"  Total query samples: {results_collection.num_queries}")
        print(f"  Queries with ground truth matches: {aggregate_metrics['queries_with_matches']}")
        print(f"  Overall accuracy (correct top-1): {format_percentage(aggregate_metrics['overall_accuracy'])}")

        # Additional recall metrics for research insight
        if len(recall_at_n) >= 5:
            print(f"  AR@5  = {format_percentage(recall_at_n[4])}")
        if len(recall_at_n) >= 10:
            print(f"  AR@10 = {format_percentage(recall_at_n[9])}")

        print("="*60)

        logger.info("Checkpoint testing completed successfully")

    except Exception as e:
        logger.error(f"Error during checkpoint testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
