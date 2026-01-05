"""
Sequence extraction from landmarks for pattern prediction.

Converts confirmed landmarks to integer sequences for use with find_patterns().
"""
import logging
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def extract_sequence(landmarks: List[Dict], landmark_type: str = 'low',
                   weekly_df: Optional[pd.DataFrame] = None) -> List[int]:
    """
    Convert confirmed landmarks to integer sequence for pattern finder.

    IMPORTANT: The sequence uses WEEK INDICES (time values), not prices.
    This follows the same method used for flood prediction and market event prediction.

    Args:
        landmarks: List of landmark dicts with 'date', 'weekly_index' keys
        landmark_type: 'low' or 'high' - which landmarks to extract
        weekly_df: Optional weekly DataFrame for calculating indices

    Returns:
        List of integers representing week numbers (e.g., [1827, 1849, 1887, ...])

    Example:
        >>> landmarks = [
        ...     {'date': '2020-01-01', 'weekly_index': 100, 'type': 'low'},
        ...     {'date': '2020-06-01', 'weekly_index': 122, 'type': 'low'},
        ...     {'date': '2021-01-01', 'weekly_index': 145, 'type': 'low'}
        ... ]
        >>> extract_sequence(landmarks, 'low')
        [100, 122, 145]
    """
    # Filter by type
    filtered = [lm for lm in landmarks if lm.get('type') == landmark_type]

    if not filtered:
        logger.warning(f"No {landmark_type} landmarks found")
        return []

    # Sort by date
    filtered.sort(key=lambda x: x.get('date', ''))

    # Extract week indices (time values), not prices
    # If weekly_index exists in landmark, use it; otherwise calculate from weekly_df
    sequence = []
    for lm in filtered:
        if 'weekly_index' in lm:
            sequence.append(lm['weekly_index'])
        elif weekly_df is not None:
            # Find the index in weekly DataFrame
            try:
                idx = weekly_df.index.get_loc(lm['date'])
                sequence.append(idx)
            except (KeyError, ValueError):
                logger.warning(f"Could not find weekly index for date {lm['date']}")
        else:
            logger.warning(f"No weekly_index available for landmark at {lm.get('date')}")

    logger.info(f"Extracted {len(sequence)} {landmark_type} landmarks as week indices: {sequence}")
    return sequence


def extract_sequences(landmarks: List[Dict]) -> Dict[str, List[int]]:
    """
    Extract both high and low sequences from landmarks.

    Args:
        landmarks: List of landmark dicts

    Returns:
        Dict with 'high' and 'low' keys containing integer sequences
    """
    return {
        'high': extract_sequence(landmarks, 'high'),
        'low': extract_sequence(landmarks, 'low')
    }


def validate_sequence(sequence: List[int], min_length: int = 3) -> bool:
    """
    Validate that a sequence is suitable for pattern prediction.

    Args:
        sequence: List of integers
        min_length: Minimum required sequence length

    Returns:
        True if sequence is valid
    """
    if not sequence:
        logger.warning("Empty sequence")
        return False

    if len(sequence) < min_length:
        logger.warning(f"Sequence too short: {len(sequence)} < {min_length}")
        return False

    # Check for valid positive integers
    if any(not isinstance(x, int) or x <= 0 for x in sequence):
        logger.warning("Sequence contains non-positive integers")
        return False

    logger.info(f"Sequence valid: {len(sequence)} elements")
    return True


def export_sequence_to_csv(sequence: List[int], filename: str,
                          index_label: str = 'index') -> None:
    """
    Export sequence to CSV for manual inspection.

    Args:
        sequence: List of integers
        filename: Output CSV filename
        index_label: Column name for the index
    """
    import pandas as pd

    df = pd.DataFrame({index_label: sequence})
    df.to_csv(filename, index=False)
    logger.info(f"Exported sequence to {filename}")


def test_sequence_extractor():
    """Test the sequence extractor."""
    print("=== Testing Sequence Extractor ===\n")

    # Sample landmarks
    sample_landmarks = [
        {'date': '2020-01-15', 'price': 1827.5, 'type': 'low', 'confirmed': True},
        {'date': '2020-06-10', 'price': 1849.3, 'type': 'low', 'confirmed': True},
        {'date': '2021-01-20', 'price': 1887.2, 'type': 'low', 'confirmed': True},
        {'date': '2021-06-15', 'price': 1909.8, 'type': 'low', 'confirmed': True},
        {'date': '2022-01-18', 'price': 1931.4, 'type': 'low', 'confirmed': True},
        {'date': '2022-08-22', 'price': 1969.6, 'type': 'low', 'confirmed': True},
        {'date': '2023-02-14', 'price': 1991.1, 'type': 'low', 'confirmed': True},
        {'date': '2023-09-18', 'price': 2007.3, 'type': 'low', 'confirmed': True},
    ]

    # Extract low sequence
    low_sequence = extract_sequence(sample_landmarks, 'low')
    print(f"Low sequence: {low_sequence}")

    # Validate
    is_valid = validate_sequence(low_sequence)
    print(f"Valid: {is_valid}")

    # Export
    export_sequence_to_csv(low_sequence, 'test_sequence.csv')
    print("Exported to test_sequence.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_sequence_extractor()
