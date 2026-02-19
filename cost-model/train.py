#!/usr/bin/env python3
"""
Cost Estimation Model Training - Car Repair Cost Prediction (v2)
=================================================================
Improved model that uses:
- Part_Name: Which car part is damaged (Front Bumper, Hood, etc.)
- Damage_Type: Type of damage (bumper_dent, door_scratch, etc.)
- Severity: Damage severity level (1-5)

Removed: Car_Segment (not meaningful for repair cost)

Usage:
    python train.py                                  # Train with defaults
    python train.py --data-path path/to/data.csv    # Custom data
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Constants
# =============================================================================

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "cost-model" / "dataset"
MODEL_DIR = ROOT / "cost-model" / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Training defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

# Random Forest defaults
DEFAULT_RF_ESTIMATORS = 200
DEFAULT_RF_MAX_DEPTH = 15
DEFAULT_RF_MIN_SAMPLES_SPLIT = 3
DEFAULT_RF_MIN_SAMPLES_LEAF = 1

# Gradient Boosting defaults
DEFAULT_GB_ESTIMATORS = 200
DEFAULT_GB_MAX_DEPTH = 6
DEFAULT_GB_LEARNING_RATE = 0.1
DEFAULT_GB_SUBSAMPLE = 0.8

# Cross-validation
CV_FOLDS = 5
CV_STD_MULTIPLIER = 2

# Display
SEPARATOR_WIDTH = 60
PERCENTAGE_MULTIPLIER = 100

# Severity
MAX_SEVERITY = 5

# Model version
MODEL_VERSION = 2


@dataclass
class TrainingConfig:
    """
    Training configuration for cost estimation models.

    Attributes:
        data_path: Path to the training data CSV file.
        random_state: Random seed for reproducibility.
        test_size: Proportion of data to use for testing.
        rf_estimators: Number of trees in Random Forest.
        rf_max_depth: Maximum depth of Random Forest trees.
        rf_min_samples_split: Minimum samples required to split RF node.
        rf_min_samples_leaf: Minimum samples required at RF leaf node.
        gb_estimators: Number of boosting stages for Gradient Boosting.
        gb_max_depth: Maximum depth of GB trees.
        gb_learning_rate: Learning rate for Gradient Boosting.
        gb_subsample: Subsample ratio for Gradient Boosting.
    """
    data_path: Path = DATA_DIR / "repair_costs_v2.csv"
    random_state: int = DEFAULT_RANDOM_STATE
    test_size: float = DEFAULT_TEST_SIZE
    rf_estimators: int = DEFAULT_RF_ESTIMATORS
    rf_max_depth: int = DEFAULT_RF_MAX_DEPTH
    rf_min_samples_split: int = DEFAULT_RF_MIN_SAMPLES_SPLIT
    rf_min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF
    gb_estimators: int = DEFAULT_GB_ESTIMATORS
    gb_max_depth: int = DEFAULT_GB_MAX_DEPTH
    gb_learning_rate: float = DEFAULT_GB_LEARNING_RATE
    gb_subsample: float = DEFAULT_GB_SUBSAMPLE


# Feature columns for the model
FEATURE_COLUMNS: List[str] = ["Part_Encoded", "Damage_Encoded", "Severity"]

# Required columns in training data
REQUIRED_COLUMNS = {"Part_Name", "Damage_Type", "Severity", "Estimated_Cost"}


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and validate the dataset.

    Args:
        data_path: Path to the CSV file containing training data.

    Returns:
        DataFrame with validated training data.

    Raises:
        ValueError: If required columns are missing from the dataset.
    """
    df = pd.read_csv(data_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def prepare_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, LabelEncoder]:
    """
    Prepare features with label encoding.

    Args:
        df: DataFrame with Part_Name, Damage_Type, and Severity columns.

    Returns:
        Tuple of (features_df, target_series, part_encoder, damage_encoder).
    """
    part_encoder = LabelEncoder()
    damage_encoder = LabelEncoder()

    df = df.copy()
    df["Part_Encoded"] = part_encoder.fit_transform(df["Part_Name"])
    df["Damage_Encoded"] = damage_encoder.fit_transform(df["Damage_Type"])

    X = df[FEATURE_COLUMNS]
    y = df["Estimated_Cost"]

    return X, y, part_encoder, damage_encoder


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str
) -> Dict[str, float]:
    """
    Evaluate model and print metrics.

    Args:
        model: Trained sklearn model with predict method.
        X: Feature DataFrame.
        y: Target Series.
        set_name: Name of the dataset (e.g., "Train", "Test").

    Returns:
        Dictionary with MAE, RMSE, R2, and MAPE metrics.
    """
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * PERCENTAGE_MULTIPLIER

    print(f"\n{set_name} Set:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train(config: TrainingConfig) -> Any:
    """
    Train cost estimation models.

    Trains both RandomForest and GradientBoosting regressors,
    evaluates their performance, and saves the best one.

    Args:
        config: Training configuration with all hyperparameters.

    Returns:
        The best performing model.
    """
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Cost Estimation Model Training (v2)")
    print(f"{'=' * SEPARATOR_WIDTH}")

    # Load data
    print(f"\nLoading data from: {config.data_path}")
    df = load_data(config.data_path)
    print(f"Total samples: {len(df)}")

    # Show data statistics
    print("\nData Statistics:")
    print(f"  Parts: {df['Part_Name'].nunique()} unique")
    print(f"  Damage Types: {df['Damage_Type'].nunique()} unique")
    print(f"  Severity Range: {df['Severity'].min()}-{df['Severity'].max()}")
    print(f"  Cost Range: ${df['Estimated_Cost'].min():,} - ${df['Estimated_Cost'].max():,}")
    print(f"  Mean Cost: ${df['Estimated_Cost'].mean():,.0f}")

    # Prepare features
    X, y, part_encoder, damage_encoder = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    print("\nDataset Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")

    # Train Random Forest
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Training Random Forest...")
    print(f"{'=' * SEPARATOR_WIDTH}")

    rf_model = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Print Random Forest model structure
    print("\nRandom Forest Structure:")
    print(f"  Type: Tree-based ensemble (NOT a neural network)")
    print(f"  Number of trees:      {rf_model.n_estimators}")
    print(f"  Max depth:            {rf_model.max_depth}")
    print(f"  Min samples split:    {rf_model.min_samples_split}")
    print(f"  Min samples leaf:     {rf_model.min_samples_leaf}")
    print(f"  Input features:       {rf_model.n_features_in_}")
    depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    nodes = [tree.tree_.node_count for tree in rf_model.estimators_]
    leaves = [tree.tree_.n_leaves for tree in rf_model.estimators_]
    print(f"  Avg tree depth:       {np.mean(depths):.1f}")
    print(f"  Avg nodes per tree:   {np.mean(nodes):.0f}")
    print(f"  Avg leaves per tree:  {np.mean(leaves):.0f}")
    print(f"  Total decision nodes: {sum(nodes):,}")

    rf_train = evaluate_model(rf_model, X_train, y_train, "Train")
    rf_test = evaluate_model(rf_model, X_test, y_test, "Test")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    print(f"\n  CV MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std() * CV_STD_MULTIPLIER:.2f})")

    # Train Gradient Boosting
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Training Gradient Boosting...")
    print(f"{'=' * SEPARATOR_WIDTH}")

    gb_model = GradientBoostingRegressor(
        n_estimators=config.gb_estimators,
        max_depth=config.gb_max_depth,
        learning_rate=config.gb_learning_rate,
        subsample=config.gb_subsample,
        random_state=config.random_state,
    )
    gb_model.fit(X_train, y_train)

    # Print Gradient Boosting model structure
    print("\nGradient Boosting Structure:")
    print(f"  Type: Tree-based boosting ensemble (NOT a neural network)")
    print(f"  Number of estimators: {gb_model.n_estimators}")
    print(f"  Max depth:            {gb_model.max_depth}")
    print(f"  Learning rate:        {gb_model.learning_rate}")
    print(f"  Subsample:            {gb_model.subsample}")
    print(f"  Input features:       {gb_model.n_features_in_}")
    gb_depths = [tree[0].tree_.max_depth for tree in gb_model.estimators_]
    gb_nodes = [tree[0].tree_.node_count for tree in gb_model.estimators_]
    gb_leaves = [tree[0].tree_.n_leaves for tree in gb_model.estimators_]
    print(f"  Avg tree depth:       {np.mean(gb_depths):.1f}")
    print(f"  Avg nodes per tree:   {np.mean(gb_nodes):.0f}")
    print(f"  Avg leaves per tree:  {np.mean(gb_leaves):.0f}")
    print(f"  Total decision nodes: {sum(gb_nodes):,}")

    gb_train = evaluate_model(gb_model, X_train, y_train, "Train")
    gb_test = evaluate_model(gb_model, X_test, y_test, "Test")

    # Cross-validation
    cv_scores = cross_val_score(gb_model, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    print(f"\n  CV MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std() * CV_STD_MULTIPLIER:.2f})")

    # Select best model based on test MAE
    if rf_test["mae"] < gb_test["mae"]:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_test
        other_model = gb_model
        other_name = "gradient_boosting"
    else:
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_metrics = gb_test
        other_model = rf_model
        other_name = "random_forest"

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"Best Model: {best_name}")
    print(f"   Test MAE:  ${best_metrics['mae']:.2f}")
    print(f"   Test MAPE: {best_metrics['mape']:.2f}%")
    print(f"   Test R2:   {best_metrics['r2']:.4f}")
    print(f"{'=' * SEPARATOR_WIDTH}")

    # Feature importance
    print("\nFeature Importance:")
    for name, importance in zip(FEATURE_COLUMNS, best_model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    # Save models and encoders
    joblib.dump(best_model, MODEL_DIR / "cost_estimator.pkl")
    joblib.dump(other_model, MODEL_DIR / f"cost_estimator_{other_name}.pkl")
    joblib.dump(part_encoder, MODEL_DIR / "part_encoder.pkl")
    joblib.dump(damage_encoder, MODEL_DIR / "damage_encoder.pkl")

    # Save metadata
    metadata = {
        "version": MODEL_VERSION,
        "best_model": best_name,
        "test_metrics": best_metrics,
        "parts": part_encoder.classes_.tolist(),
        "damage_types": damage_encoder.classes_.tolist(),
        "severity_range": [int(df["Severity"].min()), int(df["Severity"].max())],
        "feature_names": FEATURE_COLUMNS,
        "feature_importance": dict(zip(FEATURE_COLUMNS, [float(x) for x in best_model.feature_importances_])),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to: {MODEL_DIR}")

    # Show example predictions
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Example Predictions:")
    print(f"{'=' * SEPARATOR_WIDTH}")

    examples = [
        ("Front Bumper", "bumper_dent", 3),
        ("Front Door", "door_dent", 4),
        ("Hood", "hood_scratch", 2),
        ("Windshield", "glass_shatter", 5),
        ("Headlight", "head_lamp", 3),
    ]

    for part, damage_type, severity in examples:
        try:
            part_enc = part_encoder.transform([part])[0]
            damage_enc = damage_encoder.transform([damage_type])[0]
            cost = best_model.predict([[part_enc, damage_enc, severity]])[0]
            print(f"{part} - {damage_type} (Severity {severity}): ${cost:,.0f}")
        except ValueError as e:
            print(f"{part}: Not in training data - {e}")

    return best_model


def predict(part_name: str, damage_type: str, severity: int) -> float:
    """
    Predict repair cost for given inputs.

    Args:
        part_name: Name of the damaged car part.
        damage_type: Type of damage (e.g., bumper_dent).
        severity: Damage severity level (1-5).

    Returns:
        Estimated repair cost.
    """
    model = joblib.load(MODEL_DIR / "cost_estimator.pkl")
    part_encoder = joblib.load(MODEL_DIR / "part_encoder.pkl")
    damage_encoder = joblib.load(MODEL_DIR / "damage_encoder.pkl")

    part_enc = part_encoder.transform([part_name])[0]
    damage_enc = damage_encoder.transform([damage_type])[0]

    cost = model.predict([[part_enc, damage_enc, severity]])[0]

    print(f"Part: {part_name}")
    print(f"Damage Type: {damage_type}")
    print(f"Severity: {severity}/{MAX_SEVERITY}")
    print(f"Estimated Cost: ${cost:,.0f}")

    return float(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost Estimation Training v2")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR / "repair_costs_v2.csv",
                        help="Path to training data CSV")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test split ratio")

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    train(config)
