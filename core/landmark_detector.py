"""
Landmark detection using ZigZag algorithm with multi-filter confirmation.

Identifies major turning points (landmark highs and lows) in stock price data.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def adaptive_threshold(prices: pd.Series, initial_threshold: float = 0.05,
                      max_landmarks_per_year: int = 8) -> float:
    """
    Calculate adaptive threshold based on price volatility and desired landmark density.

    Args:
        prices: Series of prices
        initial_threshold: Starting threshold (default 5%)
        max_landmarks_per_year: Maximum landmarks per year (default 8 = ~1 per 6 weeks)

    Returns:
        Adjusted threshold
    """
    # Calculate approximate years in data
    weeks = len(prices)
    years = weeks / 52
    target_landmarks = int(max_landmarks_per_year * years)

    # Test with initial threshold
    initial_landmarks = len(_zigzag_detect_raw(prices, initial_threshold))

    # If too many landmarks, increase threshold
    if initial_landmarks > target_landmarks:
        # Calculate price volatility (standard deviation of returns)
        returns = prices.pct_change().dropna()
        volatility = returns.std()

        # Adjust threshold based on volatility
        # Higher volatility = higher threshold needed
        if volatility > 0.08:  # Very volatile
            adjusted = max(initial_threshold * 2.5, 0.15)
        elif volatility > 0.06:  # Moderately volatile
            adjusted = max(initial_threshold * 2.0, 0.10)
        elif volatility > 0.04:  # Normal volatility
            adjusted = max(initial_threshold * 1.5, 0.08)
        else:
            adjusted = initial_threshold

        # Still too many? Increase further
        test_landmarks = len(_zigzag_detect_raw(prices, adjusted))
        while test_landmarks > target_landmarks and adjusted < 0.50:
            adjusted += 0.05
            test_landmarks = len(_zigzag_detect_raw(prices, adjusted))

        logger.info(f"Adaptive threshold: {initial_threshold*100:.1f}% -> {adjusted*100:.1f}% "
                   f"(landmarks: {initial_landmarks} -> {test_landmarks}, target: {target_landmarks})")
        return adjusted
    else:
        return initial_threshold


def _zigzag_detect_raw(prices: pd.Series, threshold: float) -> List[Dict]:
    """Raw ZigZag detection without adaptive thresholding (internal function)."""
    if len(prices) < 3:
        return []

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
                    'zigzag_pct': reversal_pct
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
                    'zigzag_pct': reversal_pct
                })
                last_pivot = current_extreme.copy()
                current_extreme = {'price': price, 'index': i}
                trend = 'up'

    return landmarks


def _calculate_landmark_score(lm: Dict, prices: pd.Series) -> float:
    """
    Calculate significance score for a landmark.

    Combines zigzag_pct with price magnitude significance.

    Args:
        lm: Landmark dict
        prices: Full price series for context

    Returns:
        Significance score (higher = more significant)
    """
    # Base score from zigzag_pct (already in percentage)
    score = lm.get('zigzag_pct', 0) * 100

    # Add price significance bonus
    idx = lm['index']
    price = lm['price']

    # Get local context (20 weeks before and after)
    start_idx = max(0, idx - 20)
    end_idx = min(len(prices), idx + 21)
    local_prices = prices.iloc[start_idx:end_idx]

    if len(local_prices) > 0:
        local_min = local_prices.min()
        local_max = local_prices.max()
        local_range = local_max - local_min

        if local_range > 0 and local_min > 0:
            # Calculate how extreme this price is within local range
            if lm['type'] == 'high':
                # For highs: percentage above local minimum
                price_significance = ((price - local_min) / local_min) * 100
            else:
                # For lows: percentage below local maximum
                price_significance = ((local_max - price) / local_max) * 100

            # Add to score (weighted 30%)
            score += price_significance * 0.3

    return score


def filter_landmarks_by_interval(landmarks: List[Dict],
                                 min_weeks_between: int = 8,
                                 prices: Optional[pd.Series] = None) -> List[Dict]:
    """
    Filter landmarks to enforce minimum interval between consecutive landmarks.

    This prevents multiple landmarks being detected too close together.

    When landmarks conflict (within min_weeks_between), the one with higher
    significance score is kept. The score combines zigzag_pct with price
    magnitude significance.

    SPECIAL RULE: Alternating high/low pairs can be closer together.
    If the last landmark and current landmark are different types (high vs low),
    the minimum interval is reduced to 4 weeks to preserve important swing points.

    Args:
        landmarks: List of landmark dicts
        min_weeks_between: Minimum weeks between same-type landmarks (default 8 weeks)
        prices: Price series for calculating significance (optional, but recommended)

    Returns:
        Filtered list of landmarks
    """
    if len(landmarks) <= 1:
        return landmarks

    filtered = [landmarks[0]]  # Always keep the first landmark

    for lm in landmarks[1:]:
        last_lm = filtered[-1]
        weeks_between = lm['index'] - last_lm['index']

        # Check if types alternate (high -> low or low -> high)
        alternating = last_lm['type'] != lm['type']

        # Use shorter interval for alternating high/low pairs
        min_interval = 4 if alternating else min_weeks_between

        # Only keep if enough time has passed
        if weeks_between >= min_interval:
            filtered.append(lm)
        else:
            # Choose the more significant one
            # If prices available, use comprehensive scoring
            if prices is not None:
                score_current = _calculate_landmark_score(lm, prices)
                score_last = _calculate_landmark_score(last_lm, prices)

                if score_current > score_last:
                    filtered[-1] = lm
            else:
                # Fallback to zigzag_pct only
                if lm.get('zigzag_pct', 0) > last_lm.get('zigzag_pct', 0):
                    filtered[-1] = lm

    logger.info(f"Interval filter: {len(landmarks)} -> {len(filtered)} landmarks "
               f"(min {min_weeks_between} weeks, 4 for alternating types)")
    return filtered


def detect_trend(prices: pd.Series, window: int = 20) -> Dict[str, any]:
    """
    Detect the overall trend direction and strength.

    Uses moving average and linear regression to determine trend.

    Args:
        prices: Series of prices
        window: Moving average window (default 20 weeks)

    Returns:
        Dict with trend direction ('up', 'down', 'sideways') and strength
    """
    if len(prices) < window * 2:
        return {'direction': 'sideways', 'strength': 0.0}

    # Calculate moving average
    ma = prices.rolling(window=window).mean()

    # Price vs MA relationship
    above_ma = (prices > ma).sum()
    below_ma = (prices < ma).sum()
    total = above_ma + below_ma

    if total == 0:
        return {'direction': 'sideways', 'strength': 0.0}

    # Calculate trend strength based on MA relationship
    above_ratio = above_ma / total
    below_ratio = below_ma / total

    # Also check linear regression slope
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices.values, 1)

    # Normalize slope by price
    slope_pct = slope / prices.mean() * 100  # % change per week

    # Combine MA and slope signals
    if above_ratio > 0.65 and slope_pct > 0.1:
        direction = 'up'
        strength = min((above_ratio - 0.5) * 2 + abs(slope_pct) * 10, 1.0)
    elif below_ratio > 0.65 and slope_pct < -0.1:
        direction = 'down'
        strength = min((below_ratio - 0.5) * 2 + abs(slope_pct) * 10, 1.0)
    else:
        direction = 'sideways'
        strength = 0.0

    return {'direction': direction, 'strength': strength, 'slope_pct': slope_pct}


def filter_by_trend(landmarks: List[Dict], prices: pd.Series,
                    trend_window: int = 20,
                    min_trend_strength: float = 0.4) -> List[Dict]:
    """
    Filter landmarks based on trend context.

    In strong trends, only keep major trend-reversal points and filter out
    minor counter-trend fluctuations.

    Strategy:
    1. Detect trend direction and strength using sliding window
    2. Use trend line deviation to determine if counter-trend landmark is significant
    3. In strong downtrend: filter out highs that don't significantly deviate above trend
    4. In strong uptrend: filter out lows that don't significantly deviate below trend

    Args:
        landmarks: List of landmark dicts
        prices: Price series
        trend_window: Window for trend detection (default 20 weeks)
        min_trend_strength: Minimum strength to consider trend "strong" (default 0.4)

    Returns:
        Filtered list of landmarks
    """
    if len(landmarks) <= 2:
        return landmarks

    filtered = []

    for i, lm in enumerate(landmarks):
        idx = lm['index']
        lm_type = lm['type']

        # Get local price context for trend detection
        start_idx = max(0, idx - trend_window)
        end_idx = min(len(prices), idx + trend_window)
        local_prices = prices.iloc[start_idx:end_idx]

        # Detect trend at this point
        trend = detect_trend(local_prices, window=min(10, len(local_prices) // 2))

        # Decision logic
        should_keep = True

        if trend['strength'] >= min_trend_strength:
            # Calculate expected price from trend line
            x = np.arange(len(local_prices))
            slope, intercept = np.polyfit(x, local_prices.values, 1)

            # Position of this landmark in the local window
            local_pos = idx - start_idx
            expected_price = slope * local_pos + intercept
            actual_price = lm['price']

            # Calculate deviation from trend line
            if expected_price > 0:
                deviation = (actual_price - expected_price) / expected_price
            else:
                deviation = 0

            if trend['direction'] == 'down' and lm_type == 'high':
                # In strong downtrend, filter out highs that don't significantly
                # deviate ABOVE the trend line
                # A high in a downtrend should be WELL above the trend line to be significant
                # (e.g., a genuine trend reversal, not just a bounce)
                if deviation < 0.25:  # Less than 25% above trend line
                    should_keep = False
                    logger.debug(f"Filtering high at Week {idx} (price={lm['price']:.2f}, "
                               f"deviation={deviation*100:.1f}%) - not significant above downtrend line")
                else:
                    logger.debug(f"Keeping high at Week {idx} (price={lm['price']:.2f}, "
                               f"deviation={deviation*100:.1f}%) - significantly above downtrend line")

            elif trend['direction'] == 'up' and lm_type == 'low':
                # In strong uptrend, filter out lows that don't significantly
                # deviate BELOW the trend line
                if deviation > -0.25:  # Less than 25% below trend line
                    should_keep = False
                    logger.debug(f"Filtering low at Week {idx} (price={lm['price']:.2f}, "
                               f"deviation={deviation*100:.1f}%) - not significant below uptrend line")
                else:
                    logger.debug(f"Keeping low at Week {idx} (price={lm['price']:.2f}, "
                               f"deviation={deviation*100:.1f}%) - significantly below uptrend line")

        # Always keep the first and last landmarks
        if i == 0 or i == len(landmarks) - 1:
            should_keep = True

        if should_keep:
            filtered.append(lm)

    logger.info(f"Trend filter: {len(landmarks)} -> {len(filtered)} landmarks")
    return filtered


def simplify_landmarks(landmarks: List[Dict], prices: pd.Series = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Simplify landmarks to ensure STRICT alternating high/low pattern for ALL landmarks.

    The sequence must alternate: high -> low -> high -> low -> ...
    No two consecutive landmarks can be the same type, regardless of level.

    Rules:
    1. All landmarks (primary + secondary) must strictly alternate high/low
    2. When same-type landmarks are consecutive, keep only the most extreme one
    3. The most extreme one becomes 'primary', filtered-out ones become 'secondary'
    4. For consecutive highs: keep the highest price
    5. For consecutive lows: keep the lowest price

    Args:
        landmarks: List of landmark dicts

    Returns:
        Tuple of (primary_landmarks, secondary_landmarks)
        - primary_landmarks: Major landmarks with alternating pattern
        - secondary_landmarks: Filtered-out landmarks marked as 'secondary' type
    """
    print(f"*** simplify_landmarks called with {len(landmarks)} landmarks, prices is None: {prices is None} ***")

    if len(landmarks) <= 2:
        return landmarks, []

    # Sort by index
    sorted_landmarks = sorted(landmarks, key=lambda x: x['index'])

    # Build final sequence with strict alternating pattern
    final_sequence = []
    primary = []
    secondary = []

    i = 0
    while i < len(sorted_landmarks):
        current = sorted_landmarks[i]
        current_type = current['type']

        # Determine what type we expect next
        if final_sequence:
            last_type = final_sequence[-1]['type']
            expected_type = 'low' if last_type == 'high' else 'high'
        else:
            expected_type = None

        # Check if current type matches expected
        if expected_type is None or current_type == expected_type:
            # This landmark fits the pattern
            final_sequence.append({
                **current,
                'level': 'primary'
            })
            i += 1
        else:
            # Current type doesn't match expected - need to resolve conflict
            # Look ahead to find all consecutive same-type landmarks
            conflict_group = [current]
            j = i + 1
            while j < len(sorted_landmarks) and sorted_landmarks[j]['type'] == current_type:
                conflict_group.append(sorted_landmarks[j])
                j += 1

            # Determine the range of this conflict group
            conflict_start_idx = conflict_group[0]['index']
            conflict_end_idx = conflict_group[-1]['index']

            print(f"*** Conflict detected: {len(conflict_group)} consecutive {current_type}s at weeks {conflict_start_idx}-{conflict_end_idx} ***")

            # Check if the last item in final_sequence has the same type as this conflict group
            # This means we have a chain of same-type landmarks that needs to be broken
            if final_sequence and final_sequence[-1]['type'] == current_type:
                print(f"*** Last item in final_sequence is also {current_type} - merging with conflict group ***")
                # Remove last item from final_sequence and add it to conflict_group
                prev_lm = final_sequence.pop()
                conflict_group.insert(0, prev_lm)
                # Update conflict range
                conflict_start_idx = conflict_group[0]['index']
                print(f"*** Expanded conflict group to {len(conflict_group)} {current_type}s from weeks {conflict_start_idx}-{conflict_end_idx} ***")
            # These represent trend reversals that should be PRESERVED and INSERTED into the sequence
            opposite_type = 'high' if current_type == 'low' else 'low'
            intermediate_landmarks = []
            for lm in sorted_landmarks:
                if (lm['type'] == opposite_type and
                    conflict_start_idx < lm['index'] < conflict_end_idx):
                    intermediate_landmarks.append(lm)

            print(f"*** Found {len(intermediate_landmarks)} intermediate {opposite_type} landmarks in range ***")

            if intermediate_landmarks:
                print(f"*** Using existing intermediate landmarks for alternation ***")
                # Found intermediate landmarks - these break the monotony and should be preserved
                # Strategy: Find the sequence that best represents the price movement
                # For consecutive LOWs with intermediate HIGHs:
                #   - Keep the first LOW (start of downtrend)
                #   - Keep the highest HIGH (peak of retracement)
                #   - Keep the last LOW (end of downtrend, most extreme)

                # Keep the most extreme one from the conflict group as primary
                if current_type == 'high':
                    most_extreme = max(conflict_group, key=lambda x: x['price'])
                else:
                    most_extreme = min(conflict_group, key=lambda x: x['price'])

                # Add the first of conflict group to sequence (it fit the pattern before)
                final_sequence.append({
                    **conflict_group[0],
                    'level': 'primary'
                })

                # Add the best intermediate landmark to maintain alternation
                if opposite_type == 'high':
                    best_intermediate = max(intermediate_landmarks, key=lambda x: x['price'])
                else:
                    best_intermediate = min(intermediate_landmarks, key=lambda x: x['price'])

                final_sequence.append({
                    **best_intermediate,
                    'level': 'secondary'  # Mark as secondary since it's a minor retracement
                })

                # Add the most extreme from conflict group (continuation of trend)
                if most_extreme['index'] != conflict_group[0]['index']:
                    final_sequence.append({
                        **most_extreme,
                        'level': 'primary'
                    })

                # Mark others as secondary
                for lm in conflict_group:
                    if lm['index'] != conflict_group[0]['index'] and lm['index'] != most_extreme['index']:
                        secondary.append({
                            **lm,
                            'level': 'secondary',
                            'original_type': lm['type']
                        })

                # Mark other intermediates as secondary
                for lm in intermediate_landmarks:
                    if lm['index'] != best_intermediate['index']:
                        secondary.append({
                            **lm,
                            'level': 'secondary',
                            'original_type': lm['type']
                        })

                logger.debug(f"Split {current_type} group at weeks {conflict_start_idx}-{conflict_end_idx} "
                           f"with intermediate {opposite_type} at week {best_intermediate['index']}")

            else:
                # No intermediate landmarks found in zigzag output
                # Try to find potential turning points in the price data
                print(f"*** No intermediate landmarks - trying synthetic landmark creation ***")
                logger.debug(f"No intermediate landmarks for {current_type} group at weeks "
                           f"{conflict_start_idx}-{conflict_end_idx}, prices is None: {prices is None}, "
                           f"conflict_group size: {len(conflict_group)}")
                if prices is not None and len(conflict_group) >= 2:
                    # Look for price extremes between conflict landmarks
                    conflict_start_idx = conflict_group[0]['index']
                    conflict_end_idx = conflict_group[-1]['index']

                    # Extract price data in this range
                    if conflict_end_idx < len(prices):
                        price_slice = prices.iloc[conflict_start_idx:conflict_end_idx+1]

                        # Find potential opposite-type landmark
                        opposite_type = 'high' if current_type == 'low' else 'low'

                        if opposite_type == 'high':
                            # For consecutive LOWs, find the highest price
                            max_price = price_slice.max()
                            # Get the integer position of max price in the slice
                            max_pos_in_slice = price_slice.argmax()
                            start_price = conflict_group[0]['price']
                            end_price = conflict_group[-1]['price']

                            # Check if the retracement is significant (at least 15% of the move)
                            price_move = abs(end_price - start_price)
                            retracement = max_price - min(start_price, end_price)

                            print(f"*** Synthetic HIGH check: weeks {conflict_start_idx}-{conflict_end_idx}, "
                                       f"prices {start_price:.2f}->{max_price:.2f}->{end_price:.2f}, "
                                       f"move={price_move:.2f}, retracement={retracement:.2f} ({retracement/price_move*100:.1f}%) ***")

                            if retracement >= price_move * 0.15 and max_price > start_price and max_price > end_price:
                                # Found a significant retracement - insert as secondary landmark
                                actual_idx = conflict_start_idx + max_pos_in_slice
                                print(f"*** INSERTING synthetic {opposite_type} at week {actual_idx}, price {max_price:.2f} ***")

                                final_sequence.append({
                                    **conflict_group[0],
                                    'level': 'primary'
                                })

                                final_sequence.append({
                                    'index': actual_idx,
                                    'price': float(max_price),
                                    'type': opposite_type,
                                    'level': 'primary',  # Mark as PRIMARY to maintain strict alternation
                                    'zigzag_pct': 0
                                })

                                final_sequence.append({
                                    **conflict_group[-1],
                                    'level': 'primary'
                                })

                                print(f"*** Added 3 landmarks: {conflict_group[0]['type']}-{opposite_type}-{conflict_group[-1]['type']} ***")

                                # NOTE: We do NOT add middle conflict_group members to secondary list here
                                # because the synthetic landmark is sufficient to maintain alternation.
                                # Adding them would create Ls -> L violations.

                                logger.debug(f"Inserted synthetic {opposite_type} at week {actual_idx} "
                                           f"(price {max_price:.2f}) between {current_type}s "
                                           f"at weeks {conflict_start_idx}-{conflict_end_idx}")

                                i = j
                                continue

                        else:  # opposite_type == 'low', for consecutive HIGHs
                            # For consecutive HIGHs, find the lowest price
                            min_price = price_slice.min()
                            # Get the integer position of min price in the slice
                            min_pos_in_slice = price_slice.argmin()
                            start_price = conflict_group[0]['price']
                            end_price = conflict_group[-1]['price']

                            # Check if the retracement is significant (at least 15% of the move)
                            price_move = abs(end_price - start_price)
                            retracement = max(start_price, end_price) - min_price

                            if retracement >= price_move * 0.15 and min_price < start_price and min_price < end_price:
                                # Found a significant retracement - insert as secondary landmark
                                actual_idx = conflict_start_idx + min_pos_in_slice

                                final_sequence.append({
                                    **conflict_group[0],
                                    'level': 'primary'
                                })

                                final_sequence.append({
                                    'index': actual_idx,
                                    'price': float(min_price),
                                    'type': opposite_type,
                                    'level': 'primary',  # Mark as PRIMARY to maintain strict alternation
                                    'zigzag_pct': 0
                                })

                                final_sequence.append({
                                    **conflict_group[-1],
                                    'level': 'primary'
                                })

                                # NOTE: We do NOT add middle conflict_group members to secondary list here
                                # because the synthetic landmark is sufficient to maintain alternation.
                                # Adding them would create Hs -> H violations.

                                logger.debug(f"Inserted synthetic {opposite_type} at week {actual_idx} "
                                           f"(price {min_price:.2f}) between {current_type}s "
                                           f"at weeks {conflict_start_idx}-{conflict_end_idx}")

                                i = j
                                continue

                # No significant retracement found or price data not available
                # Just keep the most extreme one and mark others as secondary
                if current_type == 'high':
                    most_extreme = max(conflict_group, key=lambda x: x['price'])
                else:
                    most_extreme = min(conflict_group, key=lambda x: x['price'])

                # Add the most extreme to final sequence
                final_sequence.append({
                    **most_extreme,
                    'level': 'primary'
                })

                # Mark others as secondary
                for lm in conflict_group:
                    if lm['index'] != most_extreme['index']:
                        secondary.append({
                            **lm,
                            'level': 'secondary',
                            'original_type': lm['type']
                        })

                logger.debug(f"No intermediate landmarks found for {current_type} group "
                           f"at weeks {conflict_start_idx}-{conflict_end_idx}, "
                           f"kept most extreme at week {most_extreme['index']}")

            i = j

    # Separate primary from final sequence
    primary = [lm for lm in final_sequence if lm['level'] == 'primary']

    # Extract secondary from final sequence (includes synthetic landmarks)
    final_secondary = [lm for lm in final_sequence if lm['level'] == 'secondary']

    # Merge with any secondary landmarks that were collected separately
    # (for landmarks that are outside the main sequence)
    seen_indices = {lm['index'] for lm in final_secondary}
    for lm in secondary:
        if lm['index'] not in seen_indices:
            final_secondary.append(lm)

    print(f"*** Before trend reclassification: {len(primary)} primary, {len(final_secondary)} secondary ***")

    # ===== NEW: Trend-based simplification =====
    # Identify major trends and merge internal landmarks
    #
    # User's requirement: "趋势只存在于高点与低点之间"
    # Translation: Trends only exist between HIGH and LOW points
    # - A major rising trend: LOW -> ... -> HIGH with >=20% cumulative increase
    # - A major declining trend: HIGH -> ... -> LOW with >=20% cumulative decrease
    #
    # Within a major trend, keep only the START and END points (mark rest as secondary)
    # This maintains H-L-H-L alternation in the primary sequence
    if prices is not None and len(primary) >= 3:
        TREND_THRESHOLD = 0.20  # 20% price change defines a major trend
        newly_secondary = []
        skip_indices = set()

        # Scan through primary landmarks to find major trends
        i = 0
        while i < len(primary) - 1:
            start_idx = i
            start_lm = primary[start_idx]

            # Look ahead to find matching opposite-type landmarks
            for j in range(start_idx + 1, len(primary)):
                end_lm = primary[j]

                # Only consider opposite-type endpoints (H-L or L-H)
                if end_lm['type'] != start_lm['type']:
                    price_change = abs(end_lm['price'] - start_lm['price']) / start_lm['price']

                    if price_change >= TREND_THRESHOLD:
                        trend_dir = "RISING" if start_lm['type'] == 'low' else "DECLINING"
                        print(f"*** Major {trend_dir} trend: weeks {start_lm['index']}->{end_lm['index']}, "
                              f"price {start_lm['price']:.2f}->{end_lm['price']:.2f} ({price_change*100:.1f}%) ***")

                        # Mark ALL intermediate points (between start_idx and j) as secondary
                        for k in range(start_idx + 1, j):
                            if k not in skip_indices:
                                newly_secondary.append({
                                    **primary[k],
                                    'level': 'secondary',
                                    'original_type': primary[k]['type']
                                })
                                skip_indices.add(k)
                                print(f"*** Deleting week {primary[k]['index']} {primary[k]['type'].upper()} (trend internal) ***")

                        # Move i to j to continue from the end point
                        i = j
                        break
            else:
                # No major trend found starting from i, move to next
                i += 1

        # Rebuild primary list excluding newly-secondary landmarks
        newly_secondary_indices = {lm['index'] for lm in newly_secondary}
        reclassified_primary = [lm for lm in primary if lm['index'] not in newly_secondary_indices]

        primary = reclassified_primary
        final_secondary.extend(newly_secondary)

        print(f"*** Trend reclassification: {len(newly_secondary)} landmarks deleted (moved to secondary) ***")

    print(f"*** simplify_landmarks returning: {len(primary)} primary, {len(final_secondary)} secondary ***")
    print(f"*** Primary weeks: {[lm['index'] for lm in primary]} ***")
    print(f"*** Secondary weeks: {[lm['index'] for lm in final_secondary]} ***")

    logger.info(f"Simplify landmarks: {len(sorted_landmarks)} -> {len(primary)} primary, {len(final_secondary)} secondary")
    logger.info(f"Final sequence strictly alternates: {[lm['type'] for lm in final_sequence]}")

    return primary, final_secondary


def zigzag_detector(prices: pd.Series, threshold: float = 0.20,
                    adaptive: bool = True, use_trend_filter: bool = True,
                    return_secondary: bool = False) -> List[Dict]:
    """
    Identify trend reversals using ZigZag algorithm with adaptive noise filtering.

    The ZigZag algorithm filters out minor price movements and only
    identifies reversals that exceed a percentage threshold.

    Args:
        prices: Series of prices (typically closing prices)
        threshold: Minimum % change for reversal (e.g., 0.20 = 20%)
        adaptive: Use adaptive thresholding to reduce noise (default True)
        use_trend_filter: Filter out counter-trend noise in strong trends (default True)
        return_secondary: Also return secondary/minor landmarks (default False)

    Returns:
        List of landmark dicts with keys:
        - 'index': Position in the price series
        - 'price': Price at the landmark
        - 'type': 'high' or 'low'
        - 'level': 'primary' or 'secondary'
        - 'zigzag_pct': Percentage change that confirmed the reversal
    """
    if len(prices) < 3:
        return []

    # Apply adaptive threshold if enabled
    if adaptive:
        actual_threshold = adaptive_threshold(prices, threshold)
    else:
        actual_threshold = threshold

    landmarks = []
    trend = None  # 'up' or 'down'
    last_pivot = {'price': prices.iloc[0], 'index': 0}
    current_extreme = last_pivot.copy()

    for i in range(1, len(prices)):
        price = prices.iloc[i]

        # Calculate percentage change from last pivot
        if last_pivot['price'] > 0:
            pct_change = (price - last_pivot['price']) / last_pivot['price']
        else:
            continue

        if trend is None:
            # Initialize trend on first threshold breach
            if abs(pct_change) >= actual_threshold:
                trend = 'up' if pct_change > 0 else 'down'
                last_pivot = {'price': price, 'index': i}
                current_extreme = {'price': price, 'index': i}

        elif trend == 'up':
            # Looking for new high or reversal to down
            if price > current_extreme['price']:
                current_extreme = {'price': price, 'index': i}

            # Check for reversal (price dropped > threshold from high)
            reversal_pct = (current_extreme['price'] - price) / current_extreme['price']
            if reversal_pct >= actual_threshold:
                # Found a high landmark
                landmarks.append({
                    'index': current_extreme['index'],
                    'price': current_extreme['price'],
                    'type': 'high',
                    'zigzag_pct': reversal_pct
                })
                last_pivot = current_extreme.copy()
                current_extreme = {'price': price, 'index': i}
                trend = 'down'

        elif trend == 'down':
            # Looking for new low or reversal to up
            if price < current_extreme['price']:
                current_extreme = {'price': price, 'index': i}

            # Check for reversal (price rose > threshold from low)
            reversal_pct = (price - current_extreme['price']) / current_extreme['price']
            if reversal_pct >= actual_threshold:
                # Found a low landmark
                landmarks.append({
                    'index': current_extreme['index'],
                    'price': current_extreme['price'],
                    'type': 'low',
                    'zigzag_pct': reversal_pct
                })
                last_pivot = current_extreme.copy()
                current_extreme = {'price': price, 'index': i}
                trend = 'up'

    # Add final point if meaningful
    if landmarks and current_extreme['index'] > landmarks[-1]['index']:
        landmarks.append({
            'index': current_extreme['index'],
            'price': current_extreme['price'],
            'type': 'low' if trend == 'up' else 'high',
            'zigzag_pct': 0
        })

    # Apply interval filtering to remove noise
    if adaptive:
        landmarks = filter_landmarks_by_interval(landmarks, min_weeks_between=8, prices=prices)

    # Apply trend filtering to remove counter-trend noise
    if adaptive and use_trend_filter and len(landmarks) > 3:
        landmarks = filter_by_trend(landmarks, prices)

    # Simplify to ensure alternating pattern and create primary/secondary levels
    primary, secondary = simplify_landmarks(landmarks, prices)

    # Return based on return_secondary flag
    if return_secondary:
        # Combine primary and secondary landmarks
        # simplify_landmarks() already ensures primary list strictly alternates
        # Secondary landmarks are marked with their original_type for reference
        result = []

        for lm in primary:
            result.append({
                'index': lm['index'],
                'price': lm['price'],
                'type': lm['type'],
                'level': 'primary',
                'zigzag_pct': lm.get('zigzag_pct', 0)
            })

        for lm in secondary:
            # For synthetic landmarks, use 'type' directly; for filtered landmarks, use 'original_type'
            # Ensure we get a string type value
            if 'original_type' in lm:
                landmark_type = lm['original_type']
            elif 'type' in lm:
                landmark_type = lm['type']
            else:
                landmark_type = 'unknown'  # Should not happen

            result.append({
                'index': int(lm['index']),  # Convert numpy types to int
                'price': float(lm['price']),  # Convert numpy types to float
                'type': landmark_type,
                'original_type': landmark_type,
                'level': 'secondary',
                'zigzag_pct': float(lm.get('zigzag_pct', 0))  # Convert numpy types
            })

        # Sort by index
        result.sort(key=lambda x: x['index'])

        primary_count = len([r for r in result if r['level'] == 'primary'])
        secondary_count = len([r for r in result if r['level'] == 'secondary'])

        logger.info(f"ZigZag detected {primary_count} primary + {secondary_count} secondary landmarks "
                   f"with {actual_threshold*100:.1f}% threshold")

        # Verify primary landmarks strictly alternate
        primary_types = [r['type'] for r in result if r['level'] == 'primary']
        for i in range(len(primary_types) - 1):
            if primary_types[i] == primary_types[i+1]:
                logger.warning(f"Non-alternating primary pattern at position {i}: {primary_types[i]} -> {primary_types[i+1]}")

        return result
    else:
        # Only return primary landmarks (for backward compatibility)
        for lm in primary:
            lm.pop('level', None)  # Remove level key
        logger.info(f"ZigZag detected {len(primary)} landmarks with {actual_threshold*100:.1f}% threshold")
        return primary


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of prices
        period: RSI period (default: 14)

    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12,
                   slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator.

    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def confirm_landmarks(landmarks: List[Dict], df: pd.DataFrame,
                     volume_threshold: float = 2.0,
                     rsi_oversold: float = 30,
                     rsi_overbought: float = 70) -> List[Dict]:
    """
    Apply multi-filter confirmation to landmarks.

    Filters:
    1. Volume Surge: Volume > threshold * average(20)
    2. RSI Analysis: Oversold/overbought conditions
    3. RSI Divergence: Price direction differs from RSI direction
    4. MACD Histogram: Positive for lows, negative for highs

    Args:
        landmarks: List of landmark dicts from zigzag_detector
        df: DataFrame with price and volume data
        volume_threshold: Multiplier for average volume (default: 2.0x)
        rsi_oversold: RSI threshold for oversold condition (default: 30)
        rsi_overbought: RSI threshold for overbought condition (default: 70)

    Returns:
        List of confirmed landmarks with additional keys:
        - 'confidence': Confidence score (0.0 to 1.0)
        - 'confirmation_reasons': List of reasons for confirmation
        - 'confirmed': Boolean indicating if passed minimum confidence
    """
    if landmarks.empty if isinstance(landmarks, pd.DataFrame) else not landmarks:
        return []

    confirmed = []

    # Calculate indicators
    prices = df['收盘'] if '收盘' in df.columns else df['close']
    volumes = df['成交量'] if '成交量' in df.columns else df.get('volume', pd.Series())

    rsi = calculate_rsi(prices)
    macd_line, signal_line, histogram = calculate_macd(prices)

    # Volume moving average
    vol_ma = volumes.rolling(window=20).mean()

    for lm in landmarks:
        idx = lm['index']
        if idx >= len(df):
            continue

        confirmation_score = 0
        reasons = []

        # Filter 1: Volume Surge
        if idx < len(volumes) and not pd.isna(volumes.iloc[idx]) and not pd.isna(vol_ma.iloc[idx]):
            if volumes.iloc[idx] > volume_threshold * vol_ma.iloc[idx]:
                confirmation_score += 1
                reasons.append('volume_surge')

        # Filter 2: RSI Oversold/Overbought
        if idx < len(rsi) and not pd.isna(rsi.iloc[idx]):
            rsi_val = rsi.iloc[idx]
            if lm['type'] == 'low' and rsi_val < rsi_oversold:
                confirmation_score += 1
                reasons.append('rsi_oversold')
            elif lm['type'] == 'high' and rsi_val > rsi_overbought:
                confirmation_score += 1
                reasons.append('rsi_overbought')

        # Filter 3: RSI Divergence (stronger signal)
        if idx >= 10 and idx < len(rsi):
            lookback = 10
            prev_landmark_idx = max(0, idx - lookback)

            # Find if there was a similar landmark in recent history
            prev_rsi = rsi.iloc[prev_landmark_idx:idx].dropna()
            prev_price = prices.iloc[prev_landmark_idx:idx].dropna()

            if len(prev_rsi) > 0 and len(prev_price) > 0:
                prev_rsi_val = prev_rsi.iloc[-1]
                prev_price_val = prev_price.iloc[-1]

                if lm['type'] == 'low':
                    # Bullish divergence: price lower, RSI higher
                    if (prices.iloc[idx] < prev_price_val and
                        rsi.iloc[idx] > prev_rsi_val):
                        confirmation_score += 2
                        reasons.append('rsi_bullish_divergence')
                elif lm['type'] == 'high':
                    # Bearish divergence: price higher, RSI lower
                    if (prices.iloc[idx] > prev_price_val and
                        rsi.iloc[idx] < prev_rsi_val):
                        confirmation_score += 2
                        reasons.append('rsi_bearish_divergence')

        # Filter 4: MACD Histogram
        if idx < len(histogram) and not pd.isna(histogram.iloc[idx]):
            if lm['type'] == 'low' and histogram.iloc[idx] > 0:
                confirmation_score += 1
                reasons.append('macd_bullish')
            elif lm['type'] == 'high' and histogram.iloc[idx] < 0:
                confirmation_score += 1
                reasons.append('macd_bearish')

        # Calculate confidence (max score is around 6-7 depending on divergence)
        max_score = 7
        confidence = min(confirmation_score / max_score, 1.0)

        # Minimum confidence threshold (0.4 = at least 2-3 signals)
        lm['confidence'] = confidence
        lm['confirmation_reasons'] = reasons
        lm['confirmed'] = confidence >= 0.4

        if lm['confirmed']:
            confirmed.append(lm)

    logger.info(f"Confirmed {len(confirmed)}/{len(landmarks)} landmarks "
               f"({100*len(confirmed)/len(landmarks) if landmarks else 0:.1f}%)")

    return confirmed


def multi_timeframe_analysis(weekly_df: pd.DataFrame,
                            daily_df: pd.DataFrame,
                            threshold: float = 0.20) -> List[Dict]:
    """
    Perform multi-timeframe landmark detection.

    Strategy:
    1. Detect landmarks on weekly data (major trend changes)
    2. Confirm with weekly volume/momentum
    3. Map weekly landmarks to daily timeframe for precision

    Args:
        weekly_df: Weekly K-line DataFrame
        daily_df: Daily K-line DataFrame
        threshold: ZigZag threshold (default: 20%)

    Returns:
        List of refined landmarks with daily precision
    """
    if weekly_df.empty or daily_df.empty:
        logger.warning("Empty data provided for multi-timeframe analysis")
        return []

    # Step 1: Detect landmarks on weekly
    weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']
    weekly_landmarks = zigzag_detector(weekly_prices, threshold=threshold)

    if not weekly_landmarks:
        logger.info("No weekly landmarks detected")
        return []

    # Step 2: Confirm with weekly indicators
    weekly_confirmed = confirm_landmarks(weekly_landmarks, weekly_df)

    if not weekly_confirmed:
        logger.info("No weekly landmarks passed confirmation")
        return []

    # Step 3: Map to daily timeframe
    daily_refined = []

    for wl in weekly_confirmed:
        weekly_date = weekly_df.index[wl['index']]

        # Find corresponding daily window (±1 week)
        daily_window = daily_df[
            (daily_df.index >= weekly_date - pd.Timedelta(days=7)) &
            (daily_df.index <= weekly_date + pd.Timedelta(days=7))
        ]

        if daily_window.empty:
            continue

        # Find exact daily low/high in window
        if wl['type'] == 'low':
            daily_idx = daily_window['最低'].idxmin() if '最低' in daily_window.columns else daily_window['low'].idxmin()
            daily_price = daily_window['最低'].min() if '最低' in daily_window.columns else daily_window['low'].min()
        else:
            daily_idx = daily_window['最高'].idxmax() if '最高' in daily_window.columns else daily_window['high'].idxmax()
            daily_price = daily_window['最高'].max() if '最高' in daily_window.columns else daily_window['high'].max()

        daily_refined.append({
            'date': daily_idx,
            'price': daily_price,
            'type': wl['type'],
            'weekly_date': weekly_date,
            'weekly_index': wl['index'],  # Week index in weekly DataFrame
            'confidence': wl['confidence'],
            'zigzag_pct': wl['zigzag_pct'],
            'confirmation_reasons': wl['confirmation_reasons'],
            'confirmed': wl['confirmed']
        })

    logger.info(f"Multi-timeframe analysis: {len(daily_refined)} refined landmarks")
    return daily_refined


def test_landmark_detector():
    """Test the landmark detector with sample data."""
    print("=== Testing Landmark Detector ===\n")

    # Create synthetic price data with known landmarks
    np.random.seed(42)

    # Generate price series with major swings
    n = 200
    prices = pd.Series([100.0], dtype=float)

    for i in range(1, n):
        # Add trend and noise
        if i < 50:
            change = np.random.normal(-0.01, 0.02)  # Downtrend
        elif i < 100:
            change = np.random.normal(0.015, 0.02)  # Uptrend
        elif i < 150:
            change = np.random.normal(-0.015, 0.02)  # Downtrend
        else:
            change = np.random.normal(0.01, 0.02)  # Uptrend

        new_price = prices.iloc[-1] * (1 + change)
        prices = pd.concat([prices, pd.Series([new_price])], ignore_index=True)

    # Test ZigZag
    landmarks = zigzag_detector(prices, threshold=0.15)
    print(f"Detected {len(landmarks)} landmarks")

    for lm in landmarks:
        print(f"  {lm['type'].upper()} at index {lm['index']}: "
              f"price={lm['price']:.2f}, pct={lm['zigzag_pct']*100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_landmark_detector()
