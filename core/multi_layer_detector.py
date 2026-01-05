"""
Multi-layer landmark detection system with configurable parameters.

This module implements a 4-layer filtering system:
1. Layer 1: Low-threshold ZigZag detection (5-10%)
2. Layer 2: Statistical confirmation (frequency>2, deviation>15%)
3. Layer 3: Trend filtering (major reversals only)
4. Layer 4: Time interval filtering (8-12 weeks configurable)

All parameters are configurable and can be adjusted in real-time.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LayerFilterParams:
    """Configurable parameters for multi-layer filtering system."""

    # Layer 1: ZigZag Detection
    layer1_threshold: float = 0.08  # 8% for initial detection (lower catches more)

    # Layer 2: Statistical Confirmation
    layer2_min_frequency: int = 2  # Minimum occurrences to be valid
    layer2_min_deviation_pct: float = 0.15  # 15% minimum price deviation

    # Layer 3: Trend Filtering
    layer3_min_trend_strength: float = 0.5  # Minimum strength (0.0-1.0)
    layer3_trend_reversal_threshold: float = 0.20  # 20% for major trend reversal
    layer3_trend_window: int = 20  # Window for trend calculation (weeks)

    # Layer 4: Time Interval Filtering
    layer4_min_weeks_same_type: int = 10  # Minimum weeks between same-type landmarks
    layer4_min_weeks_alternating: int = 6  # Minimum weeks for alternating high/low

    # Visualization
    show_layer_outputs: bool = False  # Show intermediate layer results in plots


class MultiLayerDetector:
    """
    Multi-layer landmark detection system.

    Each layer progressively filters noise and refines the landmark candidates.
    """

    def __init__(self, params: Optional[LayerFilterParams] = None):
        """
        Initialize the multi-layer detector.

        Args:
            params: Filter parameters (uses defaults if not provided)
        """
        self.params = params or LayerFilterParams()
        self.layer_stats = {
            'layer1_count': 0,
            'layer2_count': 0,
            'layer3_count': 0,
            'layer4_count': 0,
            'final_count': 0
        }

    def detect(self, prices: pd.Series, debug: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Apply multi-layer filtering to detect landmarks.

        Args:
            prices: Price series (weekly recommended)
            debug: Print detailed debug information

        Returns:
            Tuple of (final_landmarks, layer_stats)
            - final_landmarks: List of landmark dicts with keys:
                - 'index': Position in price series
                - 'price': Price at landmark
                - 'type': 'high' or 'low'
                - 'layer': Which layer confirmed this landmark (1-4)
                - 'confidence': Overall confidence score (0.0-1.0)
            - layer_stats: Dictionary with counts from each layer
        """
        if len(prices) < 10:
            logger.warning("Insufficient data for multi-layer detection")
            return [], self.layer_stats

        if debug:
            print(f"\n{'='*60}")
            print(f"MULTI-LAYER LANDMARK DETECTION")
            print(f"{'='*60}")
            print(f"Data points: {len(prices)}")
            print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
            print(f"\nParameters:")
            print(f"  Layer 1 threshold: {self.params.layer1_threshold*100:.1f}%")
            print(f"  Layer 2 min frequency: {self.params.layer2_min_frequency}")
            print(f"  Layer 2 min deviation: {self.params.layer2_min_deviation_pct*100:.1f}%")
            print(f"  Layer 3 trend strength: {self.params.layer3_min_trend_strength}")
            print(f"  Layer 3 reversal threshold: {self.params.layer3_trend_reversal_threshold*100:.1f}%")
            print(f"  Layer 4 same-type interval: {self.params.layer4_min_weeks_same_type} weeks")
            print(f"  Layer 4 alternating interval: {self.params.layer4_min_weeks_alternating} weeks")
            print(f"{'='*60}\n")

        # Layer 1: Low-threshold ZigZag detection
        layer1_landmarks = self._layer1_zigzag_detection(prices, debug=debug)
        self.layer_stats['layer1_count'] = len(layer1_landmarks)

        if not layer1_landmarks:
            if debug:
                print("No landmarks detected in Layer 1")
            return [], self.layer_stats

        # Layer 2: Statistical confirmation
        layer2_landmarks = self._layer2_statistical_confirmation(
            layer1_landmarks, prices, debug=debug
        )
        self.layer_stats['layer2_count'] = len(layer2_landmarks)

        if not layer2_landmarks:
            if debug:
                print("No landmarks passed Layer 2 statistical confirmation")
            return [], self.layer_stats

        # Layer 3: Trend filtering
        layer3_landmarks = self._layer3_trend_filtering(
            layer2_landmarks, prices, debug=debug
        )
        self.layer_stats['layer3_count'] = len(layer3_landmarks)

        if not layer3_landmarks:
            if debug:
                print("No landmarks passed Layer 3 trend filtering")
            return [], self.layer_stats

        # Layer 4: Time interval filtering
        final_landmarks = self._layer4_interval_filtering(
            layer3_landmarks, prices, debug=debug
        )
        self.layer_stats['layer4_count'] = len(final_landmarks)
        self.layer_stats['final_count'] = len(final_landmarks)

        if debug:
            print(f"\n{'='*60}")
            print(f"FILTERING SUMMARY")
            print(f"{'='*60}")
            print(f"Layer 1 (ZigZag): {self.layer_stats['layer1_count']} candidates")
            print(f"Layer 2 (Statistical): {self.layer_stats['layer2_count']} passed")
            print(f"Layer 3 (Trend): {self.layer_stats['layer3_count']} passed")
            print(f"Layer 4 (Interval): {self.layer_stats['layer4_count']} final")
            print(f"Reduction: {(1 - len(final_landmarks)/len(layer1_landmarks))*100:.1f}%")
            print(f"{'='*60}\n")

        return final_landmarks, self.layer_stats

    def _layer1_zigzag_detection(self, prices: pd.Series, debug: bool = False) -> List[Dict]:
        """
        Layer 1: Low-threshold ZigZag detection.

        Uses a lower threshold (5-10%) to catch all potential reversals.
        This layer is permissive to avoid missing important landmarks.

        Args:
            prices: Price series
            debug: Print debug information

        Returns:
            List of landmark candidates
        """
        threshold = self.params.layer1_threshold
        landmarks = []
        trend = None
        last_pivot = {'price': prices.iloc[0], 'index': 0}
        current_extreme = last_pivot.copy()

        for i in range(1, len(prices)):
            price = prices.iloc[i]

            if last_pivot['price'] > 0:
                pct_change = (price - last_pivot['price']) / last_pivot['price']
            else:
                continue

            if trend is None:
                if abs(pct_change) >= threshold:
                    trend = 'up' if pct_change > 0 else 'down'
                    last_pivot = {'price': price, 'index': i}
                    current_extreme = {'price': price, 'index': i}

            elif trend == 'up':
                if price > current_extreme['price']:
                    current_extreme = {'price': price, 'index': i}

                reversal_pct = (current_extreme['price'] - price) / current_extreme['price']
                if reversal_pct >= threshold:
                    landmarks.append({
                        'index': current_extreme['index'],
                        'price': current_extreme['price'],
                        'type': 'high',
                        'layer': 1,
                        'zigzag_pct': reversal_pct,
                        'confidence': 0.25  # Base confidence for layer 1
                    })
                    last_pivot = current_extreme.copy()
                    current_extreme = {'price': price, 'index': i}
                    trend = 'down'

            elif trend == 'down':
                if price < current_extreme['price']:
                    current_extreme = {'price': price, 'index': i}

                reversal_pct = (price - current_extreme['price']) / current_extreme['price']
                if reversal_pct >= threshold:
                    landmarks.append({
                        'index': current_extreme['index'],
                        'price': current_extreme['price'],
                        'type': 'low',
                        'layer': 1,
                        'zigzag_pct': reversal_pct,
                        'confidence': 0.25
                    })
                    last_pivot = current_extreme.copy()
                    current_extreme = {'price': price, 'index': i}
                    trend = 'up'

        # Add final point
        if landmarks and current_extreme['index'] > landmarks[-1]['index']:
            landmarks.append({
                'index': current_extreme['index'],
                'price': current_extreme['price'],
                'type': 'low' if trend == 'up' else 'high',
                'layer': 1,
                'zigzag_pct': 0,
                'confidence': 0.25
            })

        if debug:
            print(f"\nLayer 1 - ZigZag Detection ({threshold*100:.1f}% threshold):")
            print(f"  Detected {len(landmarks)} candidate landmarks")
            for i, lm in enumerate(landmarks[:5]):  # Show first 5
                print(f"    {i+1}. {lm['type'].upper()} at week {lm['index']}: "
                      f"price={lm['price']:.2f}, zigzag={lm['zigzag_pct']*100:.1f}%")
            if len(landmarks) > 5:
                print(f"    ... and {len(landmarks)-5} more")

        return landmarks

    def _layer2_statistical_confirmation(self, landmarks: List[Dict],
                                        prices: pd.Series,
                                        debug: bool = False) -> List[Dict]:
        """
        Layer 2: Statistical confirmation.

        Filter landmarks based on:
        1. Price deviation from local context (must be significant)
        2. Check if similar price levels have been landmarks before (frequency)

        Args:
            landmarks: Candidates from Layer 1
            prices: Price series for context
            debug: Print debug information

        Returns:
            Filtered list of landmarks
        """
        confirmed = []

        for lm in landmarks:
            idx = lm['index']
            lm_type = lm['type']
            price = lm['price']

            # Get local context (10 weeks before and after)
            start_idx = max(0, idx - 10)
            end_idx = min(len(prices), idx + 11)
            local_prices = prices.iloc[start_idx:end_idx]

            if len(local_prices) == 0:
                continue

            local_min = local_prices.min()
            local_max = local_prices.max()
            local_range = local_max - local_min

            if local_range <= 0:
                continue

            # Calculate price deviation
            if lm_type == 'high':
                deviation = (price - local_min) / local_min
            else:
                deviation = (local_max - price) / local_max

            # Check minimum deviation requirement
            if deviation < self.params.layer2_min_deviation_pct:
                continue

            # Check frequency (how many similar landmarks exist)
            similar_count = 0
            for other_lm in landmarks:
                if (other_lm['type'] == lm_type and
                    other_lm['index'] != idx and
                    abs(other_lm['price'] - price) / price < 0.05):  # Within 5%
                    similar_count += 1

            # Frequency check: either no similar (unique) or multiple (pattern)
            # We filter out single-occurrence weak signals
            if similar_count > 0 and similar_count < self.params.layer2_min_frequency:
                continue

            # Update confidence based on layer 2 checks
            lm_copy = lm.copy()
            lm_copy['layer'] = 2
            lm_copy['deviation'] = deviation
            lm_copy['frequency'] = similar_count + 1

            # Boost confidence for strong signals
            if deviation >= self.params.layer2_min_deviation_pct * 2:  # 2x requirement
                lm_copy['confidence'] = 0.6
            elif similar_count >= self.params.layer2_min_frequency:
                lm_copy['confidence'] = 0.5
            else:
                lm_copy['confidence'] = 0.35

            confirmed.append(lm_copy)

        if debug:
            print(f"\nLayer 2 - Statistical Confirmation:")
            print(f"  Parameters: min_dev={self.params.layer2_min_deviation_pct*100:.1f}%, "
                  f"min_freq={self.params.layer2_min_frequency}")
            print(f"  Passed: {len(confirmed)}/{len(landmarks)}")
            print(f"  Filtered out: {len(landmarks) - len(confirmed)}")

        return confirmed

    def _layer3_trend_filtering(self, landmarks: List[Dict],
                               prices: pd.Series,
                               debug: bool = False) -> List[Dict]:
        """
        Layer 3: Trend filtering - keep only major trend reversals.

        Strategy:
        1. Calculate local trend around each landmark
        2. In strong trends, only keep significant reversals
        3. Filter out minor counter-trend fluctuations

        Args:
            landmarks: Candidates from Layer 2
            prices: Price series
            debug: Print debug information

        Returns:
            Filtered list of landmarks
        """
        if len(landmarks) <= 2:
            return [lm.copy() for lm in landmarks]

        filtered = []

        for i, lm in enumerate(landmarks):
            idx = lm['index']
            lm_type = lm['type']

            # Always keep first and last landmarks
            if i == 0 or i == len(landmarks) - 1:
                lm_copy = lm.copy()
                lm_copy['layer'] = 3
                lm_copy['confidence'] = min(lm['confidence'] + 0.1, 1.0)
                filtered.append(lm_copy)
                continue

            # Get local trend context
            window = self.params.layer3_trend_window
            start_idx = max(0, idx - window)
            end_idx = min(len(prices), idx + window)
            local_prices = prices.iloc[start_idx:end_idx]

            # Calculate trend
            trend = self._calculate_trend(local_prices)

            # Check if this landmark is a significant trend reversal
            should_keep = True

            if trend['strength'] >= self.params.layer3_min_trend_strength:
                # Strong trend detected - check if landmark reverses it
                expected_price = trend['intercept'] + trend['slope'] * (idx - start_idx)

                if expected_price > 0:
                    deviation = (lm['price'] - expected_price) / expected_price
                else:
                    deviation = 0

                # In strong downtrend, filter out highs that don't significantly deviate
                if trend['direction'] == 'down' and lm_type == 'high':
                    if deviation < self.params.layer3_trend_reversal_threshold * 0.5:
                        should_keep = False
                        if debug:
                            print(f"    Filtered: HIGH at week {idx} not significant "
                                  f"above downtrend (dev={deviation*100:.1f}%)")

                # In strong uptrend, filter out lows that don't significantly deviate
                elif trend['direction'] == 'up' and lm_type == 'low':
                    if deviation > -self.params.layer3_trend_reversal_threshold * 0.5:
                        should_keep = False
                        if debug:
                            print(f"    Filtered: LOW at week {idx} not significant "
                                  f"below uptrend (dev={deviation*100:.1f}%)")

            if should_keep:
                lm_copy = lm.copy()
                lm_copy['layer'] = 3
                lm_copy['trend_direction'] = trend['direction']
                lm_copy['trend_strength'] = trend['strength']
                # Boost confidence for landmarks that survive trend filtering
                lm_copy['confidence'] = min(lm['confidence'] + 0.15, 1.0)
                filtered.append(lm_copy)

        if debug:
            print(f"\nLayer 3 - Trend Filtering:")
            print(f"  Parameters: min_strength={self.params.layer3_min_trend_strength}, "
                  f"reversal_th={self.params.layer3_trend_reversal_threshold*100:.1f}%")
            print(f"  Passed: {len(filtered)}/{len(landmarks)}")
            print(f"  Filtered out: {len(landmarks) - len(filtered)}")

        return filtered

    def _calculate_trend(self, prices: pd.Series) -> Dict:
        """Calculate trend direction and strength."""
        if len(prices) < 5:
            return {'direction': 'sideways', 'strength': 0.0, 'slope': 0, 'intercept': 0}

        x = np.arange(len(prices))
        try:
            slope, intercept = np.polyfit(x, prices.values, 1)
            slope_pct = slope / prices.mean() * 100 if prices.mean() > 0 else 0

            # Determine trend direction
            if abs(slope_pct) < 0.05:
                direction = 'sideways'
                strength = 0.0
            elif slope_pct > 0:
                direction = 'up'
                strength = min(abs(slope_pct) / 0.5, 1.0)  # Normalize to 0-1
            else:
                direction = 'down'
                strength = min(abs(slope_pct) / 0.5, 1.0)

            return {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'intercept': intercept
            }
        except:
            return {'direction': 'sideways', 'strength': 0.0, 'slope': 0, 'intercept': 0}

    def _layer4_interval_filtering(self, landmarks: List[Dict],
                                  prices: pd.Series,
                                  debug: bool = False) -> List[Dict]:
        """
        Layer 4: Time interval filtering.

        Enforce minimum time intervals between landmarks:
        - Same-type landmarks: longer interval (default 10 weeks)
        - Alternating high/low: shorter interval (default 6 weeks)

        This prevents multiple landmarks too close together.

        Args:
            landmarks: Candidates from Layer 3
            prices: Price series (for significance scoring if needed)
            debug: Print debug information

        Returns:
            Final filtered list of landmarks
        """
        if len(landmarks) <= 1:
            return [lm.copy() for lm in landmarks]

        # Sort by index
        sorted_landmarks = sorted(landmarks, key=lambda x: x['index'])
        filtered = [sorted_landmarks[0]]

        for lm in sorted_landmarks[1:]:
            last_lm = filtered[-1]
            weeks_between = lm['index'] - last_lm['index']

            # Check if types alternate
            alternating = last_lm['type'] != lm['type']

            # Apply appropriate interval
            min_interval = (self.params.layer4_min_weeks_alternating if alternating
                          else self.params.layer4_min_weeks_same_type)

            if weeks_between >= min_interval:
                # Keep this landmark
                lm_copy = lm.copy()
                lm_copy['layer'] = 4
                lm_copy['confidence'] = min(lm['confidence'] + 0.1, 1.0)
                filtered.append(lm_copy)
            else:
                # Too close - keep the more significant one
                if lm['confidence'] > last_lm['confidence']:
                    filtered[-1] = lm.copy()
                    filtered[-1]['layer'] = 4
                    if debug:
                        print(f"    Replaced: {lm['type'].upper()} at week {lm['index']} "
                              f"more significant (conf: {lm['confidence']:.2f} vs {last_lm['confidence']:.2f})")
                else:
                    if debug:
                        print(f"    Filtered: {lm['type'].upper()} at week {lm['index']} "
                              f"too close to last ({weeks_between} < {min_interval} weeks)")

        if debug:
            print(f"\nLayer 4 - Interval Filtering:")
            print(f"  Parameters: same_type={self.params.layer4_min_weeks_same_type}w, "
                  f"alternating={self.params.layer4_min_weeks_alternating}w")
            print(f"  Passed: {len(filtered)}/{len(landmarks)}")
            print(f"  Filtered out: {len(landmarks) - len(filtered)}")

        return filtered


def detect_with_params(prices: pd.Series,
                      layer1_threshold: float = 0.08,
                      layer2_min_freq: int = 2,
                      layer2_min_dev: float = 0.15,
                      layer3_trend_str: float = 0.5,
                      layer4_same_type: int = 10,
                      layer4_alt: int = 6,
                      debug: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Convenience function to detect landmarks with custom parameters.

    Args:
        prices: Price series
        layer1_threshold: ZigZag threshold for Layer 1
        layer2_min_freq: Minimum frequency for Layer 2
        layer2_min_dev: Minimum deviation for Layer 2
        layer3_trend_str: Minimum trend strength for Layer 3
        layer4_same_type: Minimum weeks for same-type landmarks
        layer4_alt: Minimum weeks for alternating landmarks
        debug: Print debug information

    Returns:
        Tuple of (landmarks, stats)
    """
    params = LayerFilterParams(
        layer1_threshold=layer1_threshold,
        layer2_min_frequency=layer2_min_freq,
        layer2_min_deviation_pct=layer2_min_dev,
        layer3_min_trend_strength=layer3_trend_str,
        layer4_min_weeks_same_type=layer4_same_type,
        layer4_min_weeks_alternating=layer4_alt
    )

    detector = MultiLayerDetector(params)
    return detector.detect(prices, debug=debug)


def main():
    """Test the multi-layer detector with sample data."""
    import matplotlib.pyplot as plt

    # Generate synthetic price data
    np.random.seed(42)
    n = 200
    prices = pd.Series([100.0], dtype=float)

    for i in range(1, n):
        if i < 50:
            change = np.random.normal(-0.01, 0.02)  # Downtrend
        elif i < 120:
            change = np.random.normal(0.015, 0.025)  # Uptrend with noise
        elif i < 160:
            change = np.random.normal(-0.012, 0.02)  # Downtrend
        else:
            change = np.random.normal(0.008, 0.02)  # Uptrend

        new_price = prices.iloc[-1] * (1 + change)
        prices = pd.concat([prices, pd.Series([new_price])], ignore_index=True)

    # Test with default parameters
    print("="*60)
    print("TESTING MULTI-LAYER DETECTOR WITH DEFAULT PARAMETERS")
    print("="*60)

    detector = MultiLayerDetector()
    landmarks, stats = detector.detect(prices, debug=True)

    print(f"\nFinal landmarks found: {len(landmarks)}")
    for lm in landmarks:
        print(f"  {lm['type'].upper()} at week {lm['index']}: "
              f"price={lm['price']:.2f}, layer={lm['layer']}, conf={lm['confidence']:.2f}")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Price with landmarks
    ax = axes[0]
    ax.plot(prices.index, prices.values, 'b-', linewidth=1, alpha=0.7, label='Price')

    colors = {'high': 'red', 'low': 'green'}
    for lm in landmarks:
        color = colors[lm['type']]
        ax.scatter(lm['index'], lm['price'], c=color, s=100,
                  edgecolors='black', linewidths=2, zorder=5,
                  label=f"{lm['type'].upper()} (Layer {lm['layer']})")
        ax.annotate(f"L{lm['layer']}", (lm['index'], lm['price']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title('Multi-Layer Filtered Landmarks', fontsize=12, fontweight='bold')
    ax.set_xlabel('Week')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Layer statistics
    ax = axes[1]
    layers = ['Layer 1\nZigZag', 'Layer 2\nStatistical', 'Layer 3\nTrend', 'Layer 4\nInterval']
    counts = [
        stats['layer1_count'],
        stats['layer2_count'],
        stats['layer3_count'],
        stats['layer4_count']
    ]

    bars = ax.bar(layers, counts, color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'])
    ax.set_ylabel('Number of Landmarks')
    ax.set_title('Filtering Progress Through Layers', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/apple/study/predict/multi_layer_test.png', dpi=100)
    print(f"\nVisualization saved to multi_layer_test.png")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
