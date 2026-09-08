"""
Feature Validation Module

Validates CSV datasets and feature vectors against the canonical schema.
Detects missing features, extra columns, incompatible schemas, and forbidden
training columns before prediction or training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .schema import CANONICAL_FEATURES, FORBIDDEN_TRAINING_COLUMNS, validate_feature_list
from .aliases import normalize_column_name, normalize_columns


class FeatureValidationResult:
    """Result of feature validation against canonical schema."""
    
    def __init__(
        self,
        compatible: bool,
        expected_count: int,
        present_count: int,
        missing: List[str],
        extra: List[str],
        forbidden: List[str],
        renamed: Dict[str, str],
        warnings: List[str],
    ):
        self.compatible = compatible
        self.expected_count = expected_count
        self.present_count = present_count
        self.missing = missing
        self.extra = extra
        self.forbidden = forbidden
        self.renamed = renamed
        self.warnings = warnings
    
    def summary(self) -> str:
        """Human-readable validation summary."""
        lines = [
            "Dataset Compatibility",
            "-" * 40,
            f"Expected features: {self.expected_count}",
            f"Detected: {self.present_count}",
            f"Missing: {len(self.missing)}",
            f"Extra metadata columns: {len(self.extra)}",
        ]
        
        if self.renamed:
            lines.append(f"Renamed columns: {len(self.renamed)}")
        
        if self.compatible:
            lines.append("\n✅ COMPATIBLE")
        else:
            lines.append("\n❌ INCOMPATIBLE DATASET")
            if self.missing:
                lines.append("\nMissing features:")
                for f in self.missing[:20]:  # Show first 20
                    lines.append(f"  - {f}")
                if len(self.missing) > 20:
                    lines.append(f"  ... and {len(self.missing) - 20} more")
        
        if self.forbidden:
            lines.append("\n⚠️ Forbidden training columns detected:")
            for f in self.forbidden:
                lines.append(f"  - {f}")
        
        if self.warnings:
            lines.append("\n⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    strict: bool = True,
) -> FeatureValidationResult:
    """
    Validate a DataFrame against the canonical feature schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, missing features make the dataset incompatible.
                If False, missing features generate warnings but allow prediction.
    
    Returns:
        FeatureValidationResult with compatibility assessment
    """
    original_cols = list(df.columns)
    col_mapping = normalize_columns(original_cols)
    
    # Track renamed columns
    renamed = {}
    for orig, canonical in col_mapping.items():
        if orig != canonical:
            renamed[orig] = canonical
    
    # Check which canonical features are present after normalization
    normalized_set = set(col_mapping.values())
    canonical_set = set(CANONICAL_FEATURES)
    
    present = canonical_set & normalized_set
    missing = sorted(canonical_set - normalized_set)
    extra = sorted(normalized_set - canonical_set)
    forbidden = sorted(set(extra) & set(FORBIDDEN_TRAINING_COLUMNS))
    
    warnings = []
    
    # Check for non-numeric feature columns
    for orig_col, canon_col in col_mapping.items():
        if canon_col in canonical_set and orig_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[orig_col]):
                warnings.append(f"Feature '{orig_col}' is not numeric (dtype: {df[orig_col].dtype})")
    
    # Check for inf/nan
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
        if inf_cols:
            warnings.append(f"Infinite values in {len(inf_cols)} columns: {inf_cols[:5]}")
    
    compatible = len(missing) == 0 if strict else len(missing) < len(CANONICAL_FEATURES) * 0.5
    
    return FeatureValidationResult(
        compatible=compatible,
        expected_count=len(CANONICAL_FEATURES),
        present_count=len(present),
        missing=missing,
        extra=extra,
        forbidden=forbidden,
        renamed=renamed,
        warnings=warnings,
    )


def prepare_feature_matrix(
    df: pd.DataFrame,
    strict: bool = True,
) -> Tuple[pd.DataFrame, FeatureValidationResult]:
    """
    Prepare a DataFrame for model prediction by:
    1. Normalizing column names via aliases
    2. Validating against canonical schema
    3. Extracting features in canonical order
    4. Cleaning numeric values
    
    Args:
        df: Input DataFrame
        strict: If True, raises ValueError for incompatible schemas.
    
    Returns:
        Tuple of (feature_matrix_df, validation_result)
    
    Raises:
        ValueError: If strict=True and schema is incompatible
    """
    # Normalize column names
    df_norm = df.copy()
    col_mapping = normalize_columns(list(df_norm.columns))
    df_norm.columns = [col_mapping.get(c, c) for c in df_norm.columns]
    
    # Handle duplicate column names after normalization
    if df_norm.columns.duplicated().any():
        df_norm = df_norm.loc[:, ~df_norm.columns.duplicated(keep='first')]
    
    # Validate
    validation = validate_dataframe(df, strict=strict)
    
    if strict and not validation.compatible:
        raise ValueError(
            f"Incompatible dataset: {validation.missing_count} features missing.\n"
            f"Missing: {validation.missing}"
        )
    
    # Build feature matrix in canonical order
    feature_data = {}
    for feat in CANONICAL_FEATURES:
        if feat in df_norm.columns:
            feature_data[feat] = pd.to_numeric(df_norm[feat], errors="coerce").fillna(0.0)
        else:
            feature_data[feat] = 0.0
    
    X_df = pd.DataFrame(feature_data, index=df.index)
    
    # Replace inf values
    X_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    
    return X_df, validation
