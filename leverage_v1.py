import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_price_path(initial_price, drift, volatility, days, dt=1/365):
    """
    Simulate asset price using Geometric Brownian Motion
    
    Parameters:
    - initial_price: Starting price of the asset
    - drift: Expected annual return (μ)
    - volatility: Annual volatility (σ)
    - days: Number of days to simulate
    - dt: Time step (defaults to daily)
    
    Returns:
    - Array of simulated prices
    """
    steps = int(days / dt)
    prices = np.zeros(steps + 1)
    prices[0] = initial_price
    
    # Generate random shocks
    Z = np.random.normal(0, 1, steps)
    
    # Simulate price path
    for t in range(1, steps + 1):
        prices[t] = prices[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z[t-1])
    
    return prices

def calculate_max_leverage(
    initial_deposit,
    ltv_ratio, 
    liquidation_threshold,
    safety_margin,
    max_iterations=100
):
    """
    Calculate the maximum leverage possible given LTV and liquidation threshold
    with a safety margin using a geometric series approach
    
    Parameters:
    - initial_deposit: Initial collateral amount
    - ltv_ratio: Loan-to-Value ratio (e.g., 0.8 for 80%)
    - liquidation_threshold: Threshold for liquidation (e.g., 0.85 for 85%)
    - safety_margin: Buffer to maintain above liquidation (e.g., 0.1 for 10%)
    - max_iterations: Maximum recursion depth for leverage calculation
    
    Returns:
    - total_collateral: Final collateral value
    - total_borrowed: Final borrowed value
    - leverage: Final leverage ratio
    """
    # Apply safety margin to the LTV ratio
    safe_ltv = ltv_ratio * (1 - safety_margin)
    
    # Calculate theoretical maximum leverage using geometric series formula
    # In recursive leverage, final_value = initial_deposit * (1 + r + r² + r³ + ...)
    # For a geometric series with first term a=1 and ratio r, sum = a/(1-r) if |r|<1
    # Here r = safe_ltv, so theoretical_max_leverage = 1/(1-safe_ltv)
    theoretical_max_leverage = 1 / (1 - safe_ltv)
    
    # For practical implementation, we'll still do the iteration approach
    # to account for minimum borrowing thresholds and other real-world limitations
    total_collateral = initial_deposit
    total_borrowed = 0
    
    for _ in range(max_iterations):
        # How much more we can borrow based on current collateral
        max_additional_borrow = total_collateral * safe_ltv - total_borrowed
        
        # Stop if we can't borrow more meaningfully
        if max_additional_borrow < 0.01:
            break
            
        # Borrow and add to collateral
        total_borrowed += max_additional_borrow
        total_collateral += max_additional_borrow
    
    # Calculate final leverage
    leverage = total_collateral / initial_deposit
    
    return total_collateral, total_borrowed, leverage

def simulate_leveraged_position(
    initial_deposit,
    price_path,
    ltv_ratio,
    liquidation_threshold,
    safety_margin=0.1
):
    """
    Simulate a leveraged position over time with a price path
    
    Parameters:
    - initial_deposit: Initial collateral in ETH
    - price_path: Array of asset prices over time
    - ltv_ratio: Loan-to-Value ratio (e.g., 0.8 for 80%)
    - liquidation_threshold: Threshold for liquidation (e.g., 0.85 for 85%)
    - safety_margin: Buffer to maintain above liquidation (e.g., 0.1 for 10%)
    
    Returns:
    - days_to_liquidation: Days until liquidation (or None if no liquidation)
    - max_leverage: Maximum leverage achieved
    - health_factor_history: Record of health factors over time
    - leverage_history: Record of leverage over time
    """
    initial_price = price_path[0]
    
    # Calculate maximum leverage at the beginning
    total_collateral, total_borrowed, max_leverage = calculate_max_leverage(
        initial_deposit,
        ltv_ratio,
        liquidation_threshold,
        safety_margin
    )
    
    # Convert collateral to ETH units
    collateral_eth = total_collateral / initial_price
    
    # Tracking variables
    days_to_liquidation = None
    health_factor_history = []
    leverage_history = []

    # Simulate each day
    for day, price in enumerate(price_path):
        # Update collateral value
        collateral_value = collateral_eth * price
        
        # Calculate health factor
        health_factor = (collateral_value) / (total_borrowed * liquidation_threshold) if total_borrowed > 0 else float('inf')
        
        # Record history
        health_factor_history.append(health_factor)
        leverage_history.append(total_collateral / initial_deposit)
        
        # Check for liquidation
        if health_factor < 1.0:
            days_to_liquidation = day
            break
    
    return days_to_liquidation, max_leverage, health_factor_history, leverage_history

def run_monte_carlo_simulation(
    num_simulations=1000,
    initial_deposit=1.0,  # 1 ETH
    initial_price=3000,   # $3000 per ETH
    drift=0.05,           # 5% annual return
    volatility=0.80,      # 80% annual volatility
    ltv_ratio=0.80,       # 80% LTV
    liquidation_threshold=0.85,  # 85% liquidation threshold
    safety_margin=0.1,    # 10% safety buffer
    simulation_days=365,  # 1 year simulation
    dt=1              # Daily steps
):
    """
    Run Monte Carlo simulations to find average best leverage before defaulting
    
    Returns:
    - Dictionary of simulation results and statistics
    """
    # Results tracking
    liquidation_days = []
    max_leverages = []
    survived_max_leverages = []
    liquidated_max_leverages = []
    
    # Run simulations
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        # Generate price path
        price_path = simulate_price_path(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            days=simulation_days,
            dt=dt
        )
        
        # Simulate leveraged position
        days_to_liquidation, max_leverage, health_factors, leverage_history = simulate_leveraged_position(
            initial_deposit=initial_deposit,
            price_path=price_path,
            ltv_ratio=ltv_ratio,
            liquidation_threshold=liquidation_threshold,
            safety_margin=safety_margin
        )
        
        # Record results
        max_leverages.append(max_leverage)
        
        if days_to_liquidation is not None:
            liquidation_days.append(days_to_liquidation)
            liquidated_max_leverages.append(max_leverage)
        else:
            survived_max_leverages.append(max_leverage)
    
    # Calculate statistics
    results = {
        "num_simulations": num_simulations,
        "liquidation_rate": len(liquidation_days) / num_simulations * 100,
        "avg_max_leverage": np.mean(max_leverages),
        "median_max_leverage": np.median(max_leverages),
        "avg_days_to_liquidation": np.mean(liquidation_days) if liquidation_days else None,
        "median_days_to_liquidation": np.median(liquidation_days) if liquidation_days else None,
        "avg_survived_leverage": np.mean(survived_max_leverages) if survived_max_leverages else None,
        "avg_liquidated_leverage": np.mean(liquidated_max_leverages) if liquidated_max_leverages else None,
        "max_leverages": max_leverages,
        "liquidation_days": liquidation_days
    }
    
    return results

def plot_simulation_results(results):
    """
    Plot the results of the Monte Carlo simulation
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distribution of maximum leverage
    axes[0, 0].hist(results["max_leverages"], bins=30, alpha=0.7, color='blue')
    axes[0, 0].axvline(results["avg_max_leverage"], color='red', linestyle='--', 
                       label=f'Average: {results["avg_max_leverage"]:.2f}x')
    axes[0, 0].axvline(results["median_max_leverage"], color='green', linestyle='--', 
                        label=f'Median: {results["median_max_leverage"]:.2f}x')
    axes[0, 0].set_title("Distribution of Maximum Leverage")
    axes[0, 0].set_xlabel("Leverage")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    
    # Plot 2: Distribution of days to liquidation (for liquidated positions)
    if results["liquidation_days"]:
        axes[0, 1].hist(results["liquidation_days"], bins=30, alpha=0.7, color='red')
        axes[0, 1].axvline(results["avg_days_to_liquidation"], color='blue', linestyle='--', 
                          label=f'Average: {results["avg_days_to_liquidation"]:.1f} days')
        axes[0, 1].set_title("Days to Liquidation")
        axes[0, 1].set_xlabel("Days")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No liquidations occurred", 
                       horizontalalignment='center', verticalalignment='center')
        axes[0, 1].set_title("Days to Liquidation")
    
    # Plot 3: Survival rate pie chart
    survival_rate = 100 - results["liquidation_rate"]
    axes[1, 0].pie([results["liquidation_rate"], survival_rate], 
                  labels=[f'Liquidated ({results["liquidation_rate"]:.1f}%)', 
                          f'Survived ({survival_rate:.1f}%)'],
                  colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title("Liquidation vs. Survival Rate")
    
    # Plot 4: Comparative leverage box plot
    leverage_data = []
    labels = []
    
    leverage_data.append(results["max_leverages"])
    labels.append("All Positions")
    
    if results.get("avg_survived_leverage") is not None:
        survived_max_leverages = [lev for i, lev in enumerate(results["max_leverages"]) 
                                 if i not in [results["liquidation_days"].index(day) 
                                             for day in results["liquidation_days"]]]
        leverage_data.append(survived_max_leverages)
        labels.append("Survived Positions")
    
    if results.get("avg_liquidated_leverage") is not None:
        liquidated_max_leverages = [lev for i, lev in enumerate(results["max_leverages"]) 
                                   if i in [results["liquidation_days"].index(day) 
                                           for day in results["liquidation_days"]]]
        leverage_data.append(liquidated_max_leverages)
        labels.append("Liquidated Positions")
    
    axes[1, 1].boxplot(leverage_data, labels=labels)
    axes[1, 1].set_title("Leverage Comparison")
    axes[1, 1].set_ylabel("Leverage")
    
    plt.tight_layout()
    return fig

# Run the simulation with default parameters
if __name__ == "__main__":
    # Set simulation parameters
    params = {
        "num_simulations": 1000,        # Number of Monte Carlo runs
        "initial_deposit": 1.0,         # 1 ETH
        "initial_price": 3000,          # $3000 per ETH
        "drift": 0.05,                  # 5% annual return
        "volatility": 0.60,             # 80% annual volatility (crypto is volatile!)
        "ltv_ratio": 0.80,              # 80% LTV
        "liquidation_threshold": 0.85,  # 85% liquidation threshold
        "safety_margin": 0.1,           # 10% safety buffer
        "simulation_days": 365,         # 1 year simulation
        "dt": 1                     # Daily steps
    }
    
    print("Running Monte Carlo Simulation to find optimal leverage...")
    results = run_monte_carlo_simulation(**params)
    
    print("\n=== Simulation Results ===")
    print(f"Number of simulations: {results['num_simulations']}")
    print(f"Liquidation rate: {results['liquidation_rate']:.2f}%")
    print(f"Average maximum leverage: {results['avg_max_leverage']:.2f}x")
    print(f"Median maximum leverage: {results['median_max_leverage']:.2f}x")
    
    if results['avg_days_to_liquidation'] is not None:
        print(f"Average days to liquidation: {results['avg_days_to_liquidation']:.1f} days")
    else:
        print("No liquidations occurred during the simulation period.")
    
    if results['avg_survived_leverage'] is not None:
        print(f"Average leverage (survived positions): {results['avg_survived_leverage']:.2f}x")
    
    if results['avg_liquidated_leverage'] is not None:
        print(f"Average leverage (liquidated positions): {results['avg_liquidated_leverage']:.2f}x")
    
    # Plot the results
    fig = plot_simulation_results(results)
    plt.show()
    
    # Sensitivity Analysis - Safety Margin
    print("\n=== Safety Margin Sensitivity Analysis ===")
    safety_margins = [0.05, 0.1, 0.15, 0.2, 0.25]
    margin_results = []
    
    for margin in safety_margins:
        params["safety_margin"] = margin
        params["num_simulations"] = 200  # Reduce for sensitivity analysis
        
        print(f"Testing safety margin: {margin:.2f}")
        result = run_monte_carlo_simulation(**params)
        margin_results.append({
            "safety_margin": margin,
            "avg_max_leverage": result["avg_max_leverage"],
            "liquidation_rate": result["liquidation_rate"]
        })
        
    # Print sensitivity results
    print("\nSafety Margin | Avg Max Leverage | Liquidation Rate")
    print("-" * 50)
    for res in margin_results:
        print(f"{res['safety_margin']:.2f}        | {res['avg_max_leverage']:.2f}x           | {res['liquidation_rate']:.2f}%")