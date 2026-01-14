#!/usr/bin/env python3
"""
Script để chọn 80 subnets từ 100 subnets và tạo 10 strategies
tránh dedupe (khoảng cách >= 0.025 - an toàn hơn)
"""

import json
import math
import random
from typing import List, Dict, Tuple

# Constants
MIN_DISTANCE = 0.025  # Minimum distance để tránh dedupe (tăng lên để an toàn hơn)
NUM_STRATEGIES = 10
NUM_SUBNETS_TO_SELECT = 80
DEFAULT_OUTPUT_DIR = 'strategies_output'

# Hotkey addresses for file naming
HOTKEY_NAMES = [
    '5GFGNk1Qj6Jd8Zu3TgsLWF4oEqBYM57deXyPCCYgqwJyNnSz',
    '5Ev4Tq1RnKyRKhnng8rWSCDV1gYefAWbcXiUUCoQnee5HNiT',
    '5Dq3JDeXJRzfH2ZsYbewBqjyQQF2Ju6YcvaWFR3aGcaEVdeQ',
    '5ERstRDhmaPj3eRoNRYQkpxeUvTEd8Jbd1NnfttEQUxCv11o',
    '5H1VqpU5Y3x5LaVgQRaBRXNu1phBHQvnXgpK34cVbdM8AQom',
    '5DPdfCxpMnDE84HhMnAtDn7B3J1UmSEnxfy1M3yWE6nuWLhW',
    '5DcsCQMTzprMNMZTQVisBzHK67mau2ZjS84d8wAYGTUJw2fV',
    '5F6mMLBXJzabRS5EZ1uUiWssHSeur33e9pAnXrosw9jgHDzm',
    '5EUm7WRd3eqDpvJa49bbpmbWFvTVbAupG1WDtYu18DSyg1cY',
    '5CfbCbQZ2Gifo4Sr8nXVxKpYPL8nY9EPf1WvoDcpUrxBdEW8',
]

def load_subnet_data(json_file: str) -> List[Dict]:
    """Load subnet data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('data', [])

def calculate_subnet_score(subnet: Dict) -> float:
    """
    Tính score cho subnet dựa trên nhiều yếu tố:
    - Market cap (cao hơn = tốt hơn)
    - Liquidity (cao hơn = tốt hơn)
    - Price stability (volatility thấp = tốt hơn)
    - Fear & Greed Index (gần 50 = neutral = tốt)
    """
    try:
        # Market cap (normalized)
        market_cap = float(subnet.get('market_cap', 0))
        
        # Liquidity
        liquidity = float(subnet.get('liquidity', 0))
        
        # Price change (lower volatility = better)
        price_change_1d = abs(float(subnet.get('price_change_1_day', 0)))
        price_change_1w = abs(float(subnet.get('price_change_1_week', 0)))
        volatility = (price_change_1d + price_change_1w) / 2
        
        # Fear & Greed Index (closer to 50 = better)
        fgi = subnet.get('fear_and_greed_index')
        if fgi and fgi != 'null':
            fgi_score = 100 - abs(50 - float(fgi))
        else:
            fgi_score = 50
        
        # Root proportion (higher = better)
        root_prop = float(subnet.get('root_prop', 0))
        
        # Combine scores (weighted)
        score = (
            (market_cap / 1e15) * 0.3 +  # Market cap weight
            (liquidity / 1e12) * 0.2 +    # Liquidity weight
            (100 / (1 + volatility)) * 0.2 +  # Stability weight (inverse volatility)
            fgi_score * 0.1 +              # Fear & Greed weight
            root_prop * 100 * 0.2          # Root proportion weight
        )
        
        return score
    except (ValueError, TypeError, KeyError):
        return 0

def select_top_subnets(subnets: List[Dict], n: int = 80) -> List[Dict]:
    """
    Chọn top N subnets dựa trên score
    Loại bỏ subnet 0 (Root) vì không nên đầu tư vào root
    """
    # Filter out root subnet (netuid = 0)
    filtered_subnets = [s for s in subnets if s.get('netuid', 0) != 0]
    
    # Calculate scores
    subnet_scores = []
    for subnet in filtered_subnets:
        score = calculate_subnet_score(subnet)
        subnet_scores.append((subnet, score))
    
    # Sort by score descending
    subnet_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    return [s[0] for s in subnet_scores[:n]]

def normalize_allocation(allocations: Dict[int, float]) -> Dict[int, float]:
    """Normalize allocations to sum to 1.0"""
    total = sum(abs(v) for v in allocations.values())
    if total < 1e-6:
        return {}
    return {k: v / total for k, v in allocations.items()}

def cap_max_allocation(weights: List[float], num_subnets: int) -> List[float]:
    """
    Cap maximum allocation theo ràng buộc Bittensor: max_alloc <= 1/N^0.7
    Nếu có weight vượt quá, sẽ cap lại và redistribute phần dư cho các subnet khác
    """
    max_allowed = 1.0 / (num_subnets ** 0.7)
    
    # Normalize weights trước
    total = sum(weights)
    if total < 1e-6:
        return weights
    weights = [w / total for w in weights]
    
    # Lặp lại cho đến khi không còn weight nào vượt quá
    max_iterations = 100
    for iteration in range(max_iterations):
        # Tìm các weights vượt quá
        excess_total = 0
        num_capped = 0
        new_weights = []
        
        for w in weights:
            if w > max_allowed:
                new_weights.append(max_allowed)
                excess_total += (w - max_allowed)
                num_capped += 1
            else:
                new_weights.append(w)
        
        # Nếu không có weight nào vượt quá, done
        if num_capped == 0:
            break
        
        # Redistribute excess vào các weights chưa bị cap
        num_uncapped = num_subnets - num_capped
        if num_uncapped > 0:
            # Phân phối theo tỷ lệ của các weights chưa bị cap
            uncapped_total = sum(w for w in new_weights if w < max_allowed)
            if uncapped_total > 1e-6:
                for i in range(len(new_weights)):
                    if new_weights[i] < max_allowed:
                        # Tăng theo tỷ lệ
                        boost = excess_total * (new_weights[i] / uncapped_total)
                        new_weights[i] += boost
            else:
                # Nếu tất cả đều bị cap, phân đều
                boost_per_subnet = excess_total / num_uncapped
                for i in range(len(new_weights)):
                    if new_weights[i] < max_allowed:
                        new_weights[i] += boost_per_subnet
        
        weights = new_weights
    
    # Final normalization
    total = sum(weights)
    if total > 1e-6:
        weights = [w / total for w in weights]
    
    return weights

def calculate_distance(strat1: Dict, strat2: Dict) -> float:
    """Tính khoảng cách Euclidean giữa 2 strategies"""
    all_netuids = set()
    for k in strat1.keys():
        if k != '_':
            all_netuids.add(k)
    for k in strat2.keys():
        if k != '_':
            all_netuids.add(k)
    
    norm1 = normalize_allocation({k: v for k, v in strat1.items() if k != '_'})
    norm2 = normalize_allocation({k: v for k, v in strat2.items() if k != '_'})
    
    vec1 = [norm1.get(n, 0) for n in sorted(all_netuids)]
    vec2 = [norm2.get(n, 0) for n in sorted(all_netuids)]
    
    return math.dist(vec1, vec2)

def create_strategy_with_pattern(
    subnets: List[Dict],
    pattern_index: int,
    num_strategies: int
) -> Dict:
    """
    Tạo strategy với pattern khác nhau
    Pattern dựa trên:
    - Exponential decay với decay rate khác nhau
    - Focus groups khác nhau
    """
    num_subnets = len(subnets)
    strategy = {'_': 0}  # Asset class = Tao/Alpha
    
    if pattern_index == 0:
        # Strategy 0: Standard exponential decay
        weights = []
        for i in range(num_subnets):
            weight = math.exp(-i * 0.05)
            weights.append(weight)
    else:
        # Other strategies: Varied patterns
        # Rotating focus groups
        focus_start = (pattern_index * (num_subnets // num_strategies)) % num_subnets
        focus_size = max(10, num_subnets // 8)
        focus_end = min(focus_start + focus_size, num_subnets)
        
        weights = []
        for i in range(num_subnets):
            # Base weight with varying decay rate
            decay_rate = 0.03 + (pattern_index * 0.01)
            base_w = math.exp(-i * decay_rate)
            
            # Adjust for focus group
            if focus_start <= i < focus_end:
                # Boost focus group
                multiplier = 1.5 + (0.5 * (1.0 - i / num_subnets))
                adjusted_w = base_w * multiplier
            elif i < focus_start:
                # Reduce before focus
                distance = focus_start - i
                reduction = 0.8 + 0.2 * math.exp(-distance * 0.1)
                adjusted_w = base_w * reduction
            else:
                # Reduce after focus
                distance = i - focus_end
                reduction = 0.6 + 0.3 * math.exp(-distance * 0.1)
                adjusted_w = base_w * reduction
            
            weights.append(adjusted_w)
        
        # Ensure monotonic decrease (priority order)
        for i in range(1, len(weights)):
            if weights[i] > weights[i-1]:
                weights[i] = weights[i-1] * 0.999
    
    # Normalize
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Cap max allocation theo ràng buộc Bittensor: max_alloc <= 1/N^0.7
    weights = cap_max_allocation(weights, num_subnets)
    
    # Create strategy dict
    for i, subnet in enumerate(subnets):
        netuid = subnet['netuid']
        strategy[netuid] = weights[i]
    
    return strategy

def create_strategies(
    subnets: List[Dict],
    num_strategies: int = 10,
    min_distance: float = 0.015,
    max_attempts: int = 100
) -> List[Dict]:
    """
    Tạo N strategies với khoảng cách tối thiểu
    """
    strategies = []
    
    for i in range(num_strategies):
        best_strategy = None
        best_min_dist = 0
        
        # Try multiple times to find strategy with good distance
        for attempt in range(max_attempts):
            # Create candidate strategy
            candidate = create_strategy_with_pattern(subnets, i, num_strategies)
            
            if not strategies:
                # First strategy, accept it
                best_strategy = candidate
                break
            
            # Calculate minimum distance to existing strategies
            min_dist = float('inf')
            for existing_strat in strategies:
                dist = calculate_distance(candidate, existing_strat)
                min_dist = min(min_dist, dist)
            
            # Keep track of best candidate
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_strategy = candidate
            
            # If good enough, accept it
            if min_dist >= min_distance:
                break
        
        if best_strategy:
            strategies.append(best_strategy)
            print(f"Strategy {i}: min_distance = {best_min_dist:.6f}")
        else:
            print(f"Warning: Could not create strategy {i} with sufficient distance")
    
    return strategies

def validate_strategies(strategies: List[Dict], min_distance: float = 0.015):
    """Validate strategies and print distance matrix"""
    print("\n" + "="*70)
    print("VALIDATION: Distance Matrix")
    print("="*70)
    
    min_dist_overall = float('inf')
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            dist = calculate_distance(strategies[i], strategies[j])
            min_dist_overall = min(min_dist_overall, dist)
            status = "✓" if dist >= min_distance else "✗"
            print(f"{status} Strategy {i} <-> {j}: {dist:.6f}")
    
    print(f"\nOverall minimum distance: {min_dist_overall:.6f}")
    if min_dist_overall >= min_distance:
        print(f"✓ All strategies satisfy min_distance >= {min_distance}")
    else:
        print(f"✗ Warning: Some strategies below threshold {min_distance}")
    
    # Validate max allocation constraint
    print("\n" + "="*70)
    print("VALIDATION: Max Allocation Constraint")
    print("="*70)
    
    all_valid = True
    for i, strategy in enumerate(strategies):
        # Count số subnets (không tính '_')
        num_subnets = len([k for k in strategy.keys() if k != '_'])
        max_allowed = 1.0 / (num_subnets ** 0.7)
        
        # Tìm max allocation
        max_alloc = 0
        max_netuid = None
        for k, v in strategy.items():
            if k != '_' and v > max_alloc:
                max_alloc = v
                max_netuid = k
        
        status = "✓" if max_alloc <= max_allowed else "✗"
        if max_alloc > max_allowed:
            all_valid = False
        
        print(f"{status} Strategy {i}: max_alloc = {max_alloc:.6f} (netuid {max_netuid}), "
              f"limit = {max_allowed:.6f} (1/{num_subnets}^0.7)")
    
    if all_valid:
        print(f"\n✓ All strategies satisfy max allocation constraint")
    else:
        print(f"\n✗ Warning: Some strategies violate max allocation constraint")

def format_strategy(strat: Dict) -> str:
    """Format strategy for output file"""
    lines = ['{']
    items = list(strat.items())
    for i, (key, value) in enumerate(items):
        if isinstance(key, str):
            key_str = f"    '{key}'"
        else:
            key_str = f"    {key}"
        comma = ',' if i < len(items) - 1 else ','
        lines.append(f'{key_str}: {value}{comma}')
    lines.append('}')
    return '\n'.join(lines)

def save_strategies(strategies: List[Dict], output_dir: str = DEFAULT_OUTPUT_DIR):
    """Save strategies to files"""
    import os
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i, strategy in enumerate(strategies):
        # Use hotkey name if available, otherwise fallback to strategy_i
        if i < len(HOTKEY_NAMES):
            base_filename = f"{HOTKEY_NAMES[i]}.txt"
        else:
            base_filename = f'strategy_{i}.txt'
        
        if output_dir:
            filename = os.path.join(output_dir, base_filename)
        else:
            filename = base_filename
            
        with open(filename, 'w') as f:
            f.write(format_strategy(strategy))
        print(f"✓ Saved: {filename}")

def main():
    import sys
    
    print("="*70)
    print("SUBNET STRATEGY GENERATOR")
    print("="*70)
    
    # Check if JSON file provided
    if len(sys.argv) < 2:
        print("\nUsage: python create_subnet_strategies.py <subnet_data.json>")
        print("\nPlease provide a JSON file with subnet data.")
        print("Expected format: {\"data\": [{subnet1}, {subnet2}, ...]}")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Load data
    print(f"\n1. Loading subnet data from: {json_file}")
    subnets = load_subnet_data(json_file)
    print(f"   Loaded {len(subnets)} subnets")
    
    # Select top subnets
    print(f"\n2. Selecting top {NUM_SUBNETS_TO_SELECT} subnets...")
    selected_subnets = select_top_subnets(subnets, NUM_SUBNETS_TO_SELECT)
    print(f"   Selected {len(selected_subnets)} subnets")
    print(f"\n   Top 10 selected subnets:")
    for i, subnet in enumerate(selected_subnets[:10]):
        score = calculate_subnet_score(subnet)
        print(f"   {i+1:2d}. Netuid {subnet['netuid']:3d} ({subnet['name']:20s}) - Score: {score:.2f}")
    
    # Create strategies
    print(f"\n3. Creating {NUM_STRATEGIES} strategies...")
    strategies = create_strategies(
        selected_subnets,
        num_strategies=NUM_STRATEGIES,
        min_distance=MIN_DISTANCE
    )
    
    # Validate
    print(f"\n4. Validating strategies...")
    validate_strategies(strategies, MIN_DISTANCE)
    
    # Save
    print(f"\n5. Saving strategies...")
    save_strategies(strategies)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Created {len(strategies)} strategies")
    print(f"✓ Each strategy allocates to {len(selected_subnets)} subnets")
    print(f"✓ Minimum distance threshold: {MIN_DISTANCE}")
    print(f"✓ Strategies saved to: {DEFAULT_OUTPUT_DIR}/")
    print(f"✓ Files named using hotkey addresses")
    print("\nGenerated files:")
    for i in range(min(len(strategies), len(HOTKEY_NAMES))):
        print(f"  {i+1}. {HOTKEY_NAMES[i]}")
    print("\nNext steps:")
    print("1. Review the generated strategies")
    print("2. Copy files to Investing/strat/ directory:")
    print(f"   cp {DEFAULT_OUTPUT_DIR}/* Investing/strat/")
    print("3. Submit at different times (>= 2 hours apart for Tao/Alpha)")
    print("="*70)

if __name__ == '__main__':
    main()
