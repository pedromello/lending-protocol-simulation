import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_price_path(initial_price, drift, volatility, days, dt=1/365):
    """
    Simulate asset price using Geometric Brownian Motion and random walk
    
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
        # Geometric Brownian Motion
        gbm_price = prices[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z[t-1])
        
        # Random walk component
        price_change = np.random.normal(0, volatility)
        random_walk_price = prices[t-1] * (1 + price_change)
        
        # Take average of both models
        prices[t] = max(0.1, random_walk_price)
    
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
    safety_margin=0.1,
    borrow_interest_rate=0.05,  # 5% annual interest on borrows
    supply_interest_rate=0.03   # 3% annual interest on supply
):
    """
    Simulate a leveraged position over time with a price path
    
    Parameters:
    - initial_deposit: Initial collateral in ETH
    - price_path: Array of asset prices over time
    - ltv_ratio: Loan-to-Value ratio (e.g., 0.8 for 80%)
    - liquidation_threshold: Threshold for liquidation (e.g., 0.85 for 85%)
    - safety_margin: Buffer to maintain above liquidation (e.g., 0.1 for 10%)
    - borrow_interest_rate: Annual interest rate on borrowed funds
    - supply_interest_rate: Annual interest rate earned on supplied funds
    
    Returns:
    - days_to_liquidation: Days until liquidation (or None if no liquidation)
    - max_leverage: Maximum leverage achieved
    - health_factor_history: Record of health factors over time
    - leverage_history: Record of leverage over time
    - profit_loss: Final P&L (in USD)
    - roi: Return on Investment (%)
    """
    initial_price = price_path[0]
    dt = 1/365  # Daily time steps
    
    # Calculate maximum leverage at the beginning
    total_collateral, total_borrowed, max_leverage = calculate_max_leverage(
        initial_deposit,
        ltv_ratio,
        liquidation_threshold,
        safety_margin
    )
    
    # Convert collateral to ETH units
    collateral_eth = total_collateral / initial_price
    
    # Initial investment in USD
    initial_investment_usd = initial_deposit * initial_price
    
    # Tracking variables
    days_to_liquidation = None
    health_factor_history = []
    leverage_history = []
    
    # Simulate each day
    for day, price in enumerate(price_path):
        # Apply daily interest to borrowed amount (compound interest)
        total_borrowed *= (1 + borrow_interest_rate * dt)
        
        # Apply supply interest to collateral (in USD terms)
        # Note: This is an approximation; in reality, interest would be in the same asset
        collateral_eth *= (1 + supply_interest_rate * dt)
        
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
            
            # In liquidation, assume recovery of (1 - liquidation_threshold) of the collateral
            # This models a liquidation penalty
            liquidation_penalty = 0.05  # 5% penalty in addition to debt repayment
            remaining_value = max(0, collateral_value - total_borrowed * (1 + liquidation_penalty))
            
            # Calculate P&L
            profit_loss = remaining_value - initial_investment_usd
            roi = (profit_loss / initial_investment_usd) * 100
            
            return days_to_liquidation, max_leverage, health_factor_history, leverage_history, profit_loss, roi
    
    # If no liquidation occurred, calculate final P&L
    final_value = collateral_eth * price_path[-1] - total_borrowed
    profit_loss = final_value - initial_investment_usd
    roi = (profit_loss / initial_investment_usd) * 100
    
    return days_to_liquidation, max_leverage, health_factor_history, leverage_history, profit_loss, roi

def run_monte_carlo_simulation(
    num_simulations=1000,
    initial_deposit=1.0,  # 1 ETH
    initial_price=3000,   # $3000 per ETH
    drift=0.05,           # 5% annual return
    volatility=0.80,      # 80% annual volatility
    ltv_ratio=0.80,       # 80% LTV
    liquidation_threshold=0.85,  # 85% liquidation threshold
    safety_margin=0.1,    # 10% safety buffer
    borrow_interest_rate=0.05,   # 5% annual interest on borrows
    supply_interest_rate=0.03,   # 3% annual interest on supply
    simulation_days=365,  # 1 year simulation
    dt=1/365              # Daily steps
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
    profit_losses = []
    roi_values = []
    survived_pnl = []
    liquidated_pnl = []
    price_path = []
    
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
        days_to_liquidation, max_leverage, health_factors, leverage_history, profit_loss, roi = simulate_leveraged_position(
            initial_deposit=initial_deposit,
            price_path=price_path,
            ltv_ratio=ltv_ratio,
            liquidation_threshold=liquidation_threshold,
            safety_margin=safety_margin,
            borrow_interest_rate=borrow_interest_rate,
            supply_interest_rate=supply_interest_rate
        )
        
        # Record results
        max_leverages.append(max_leverage)
        profit_losses.append(profit_loss)
        roi_values.append(roi)
        
        if days_to_liquidation is not None:
            liquidation_days.append(days_to_liquidation)
            liquidated_max_leverages.append(max_leverage)
            liquidated_pnl.append(profit_loss)
        else:
            survived_max_leverages.append(max_leverage)
            survived_pnl.append(profit_loss)
    
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
        "liquidation_days": liquidation_days,
        
        # Profit/Loss metrics
        "avg_profit_loss": np.mean(profit_losses),
        "median_profit_loss": np.median(profit_losses),
        "avg_roi": np.mean(roi_values),
        "median_roi": np.median(roi_values),
        "profit_loss_std": np.std(profit_losses),
        "roi_std": np.std(roi_values),
        "profit_losses": profit_losses,
        "roi_values": roi_values,
        "survived_pnl": survived_pnl,
        "liquidated_pnl": liquidated_pnl,
        "avg_survived_pnl": np.mean(survived_pnl) if survived_pnl else None,
        "avg_liquidated_pnl": np.mean(liquidated_pnl) if liquidated_pnl else None,
        "price_path": price_path
    }
    
    return results

def plot_simulation_results(results):
    """
    Plot the results of the Monte Carlo simulation including profit/loss metrics
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    
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
    
    # Plot 3: Profit/Loss Distribution
    axes[1, 0].hist(results["profit_losses"], bins=30, alpha=0.7, color='green')
    axes[1, 0].axvline(results["avg_profit_loss"], color='red', linestyle='--', 
                      label=f'Average P&L: ${results["avg_profit_loss"]:.2f}')
    axes[1, 0].axvline(0, color='black', linestyle='-', label='Breakeven')
    axes[1, 0].set_title("Profit/Loss Distribution")
    axes[1, 0].set_xlabel("Profit/Loss ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    
    # Plot 4: ROI Distribution
    axes[1, 1].hist(results["roi_values"], bins=30, alpha=0.7, color='purple')
    axes[1, 1].axvline(results["avg_roi"], color='red', linestyle='--', 
                      label=f'Average ROI: {results["avg_roi"]:.2f}%')
    axes[1, 1].axvline(0, color='black', linestyle='-', label='Breakeven')
    axes[1, 1].set_title("Return on Investment (ROI) Distribution")
    axes[1, 1].set_xlabel("ROI (%)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    
    # Plot 5: Survival rate pie chart
    survival_rate = 100 - results["liquidation_rate"]
    axes[2, 0].pie([results["liquidation_rate"], survival_rate], 
                  labels=[f'Liquidated ({results["liquidation_rate"]:.1f}%)', 
                          f'Survived ({survival_rate:.1f}%)'],
                  colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
    axes[2, 0].set_title("Liquidation vs. Survival Rate")
    
    # Plot 6: P&L Comparison between Survived vs Liquidated Positions
    pnl_data = []
    pnl_labels = []
    
    pnl_data.append(results["profit_losses"])
    pnl_labels.append("All Positions")
    
    if results.get("survived_pnl") and len(results["survived_pnl"]) > 0:
        pnl_data.append(results["survived_pnl"])
        pnl_labels.append(f"Survived\n(Avg: ${results['avg_survived_pnl']:.2f})")
    
    if results.get("liquidated_pnl") and len(results["liquidated_pnl"]) > 0:
        pnl_data.append(results["liquidated_pnl"])
        pnl_labels.append(f"Liquidated\n(Avg: ${results['avg_liquidated_pnl']:.2f})")
    
    axes[2, 1].boxplot(pnl_data, labels=pnl_labels)
    axes[2, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2, 1].set_title("P&L Comparison: Survived vs. Liquidated")
    axes[2, 1].set_ylabel("Profit/Loss ($)")
    
    # Plot 7: Price Path
    axes[3, 0].plot(results["price_path"], label='Price Path', color='blue')
    axes[3, 0].set_title("Simulated Price Path")
    axes[3, 0].set_xlabel("Days")
    axes[3, 0].set_ylabel("Price ($)")
    axes[3, 0].legend()
    
    
    plt.tight_layout()
    return fig

def analyze_optimal_leverage(
    safety_margins=[0.05, 0.1, 0.15, 0.2, 0.25],
    num_simulations=200,
    **sim_params
):
    """
    Analyze the relationship between leverage, risk (safety margin), and returns
    """
    results = []
    
    for margin in tqdm(safety_margins, desc="Analyzing optimal leverage"):
        # Update parameters with current safety margin
        params = sim_params.copy()
        params["safety_margin"] = margin
        params["num_simulations"] = num_simulations
        
        # Run simulation with current parameters
        result = run_monte_carlo_simulation(**params)
        
        # Extract key metrics
        results.append({
            "safety_margin": margin,
            "avg_max_leverage": result["avg_max_leverage"],
            "liquidation_rate": result["liquidation_rate"],
            "avg_profit_loss": result["avg_profit_loss"],
            "avg_roi": result["avg_roi"],
            "profit_loss_std": result["profit_loss_std"]  # Standard deviation of P&L (risk measure)
        })
    
    # Create figure for optimal leverage analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    margins = [r["safety_margin"] for r in results]
    leverages = [r["avg_max_leverage"] for r in results]
    liquidation_rates = [r["liquidation_rate"] for r in results]
    avg_returns = [r["avg_roi"] for r in results]
    risks = [r["profit_loss_std"] for r in results]
    
    # Plot 1: Safety Margin vs. Leverage and Liquidation Rate
    ax1 = axes[0, 0]
    lns1 = ax1.plot(margins, leverages, 'b-', marker='o', label='Avg. Max Leverage')
    ax1.set_xlabel('Safety Margin')
    ax1.set_ylabel('Average Max Leverage', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    lns2 = ax2.plot(margins, liquidation_rates, 'r-', marker='s', label='Liquidation Rate (%)')
    ax2.set_ylabel('Liquidation Rate (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    ax1.set_title('Effect of Safety Margin on Leverage and Risk')
    
    # Plot 2: Safety Margin vs. Average ROI
    ax3 = axes[0, 1]
    ax3.plot(margins, avg_returns, 'g-', marker='o')
    ax3.set_xlabel('Safety Margin')
    ax3.set_ylabel('Average ROI (%)')
    ax3.set_title('Effect of Safety Margin on Returns')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero line
    
    # Plot 3: Leverage vs. Return
    axes[1, 0].plot(leverages, avg_returns, 'purple', marker='o')
    for i, margin in enumerate(margins):
        axes[1, 0].annotate(f"{margin:.2f}", (leverages[i], avg_returns[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
    axes[1, 0].set_xlabel('Average Max Leverage')
    axes[1, 0].set_ylabel('Average ROI (%)')
    axes[1, 0].set_title('Leverage vs. Return')
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero line
    
    # Plot 4: Risk-Return Profile
    axes[1, 1].plot(risks, avg_returns, 'orange', marker='o')
    for i, margin in enumerate(margins):
        axes[1, 1].annotate(f"{margin:.2f}", (risks[i], avg_returns[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
    axes[1, 1].set_xlabel('Risk (P&L Standard Deviation)')
    axes[1, 1].set_ylabel('Average ROI (%)')
    axes[1, 1].set_title('Risk-Return Profile')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero line
    
    plt.tight_layout()
    return fig, results

# Run the simulation with default parameters
if __name__ == "__main__":
    # Set simulation parameters
    params = {
        "num_simulations": 1000,        # Number of Monte Carlo runs
        "initial_deposit": 1.0,         # 1 ETH
        "initial_price": 3000,          # $3000 per ETH
        "drift": 0.05,                  # 5% annual return
        "volatility": 0.10,             # 80% annual volatility (crypto is volatile!)
        "ltv_ratio": 0.80,              # 80% LTV
        "liquidation_threshold": 0.85,  # 85% liquidation threshold
        "safety_margin": 0.1,           # 10% safety buffer
        "borrow_interest_rate": 0.05,   # 5% annual borrow interest
        "supply_interest_rate": 0.03,   # 3% annual supply interest
        "simulation_days": 365,         # 1 year simulation
        "dt": 1                     # Daily steps
    }
    
    print("Running Monte Carlo Simulation to find optimal leverage and profitability...")
    results = run_monte_carlo_simulation(**params)
    
    print("\n=== Simulation Results ===")
    print(f"Number of simulations: {results['num_simulations']}")
    print(f"Liquidation rate: {results['liquidation_rate']:.2f}%")
    print(f"Average maximum leverage: {results['avg_max_leverage']:.2f}x")
    
    print("\n=== Profit/Loss Results ===")
    print(f"Average P&L: ${results['avg_profit_loss']:.2f}")
    print(f"Median P&L: ${results['median_profit_loss']:.2f}")
    print(f"Average ROI: {results['avg_roi']:.2f}%")
    print(f"Median ROI: {results['median_roi']:.2f}%")
    print(f"P&L Standard Deviation: ${results['profit_loss_std']:.2f}")
    
    if results['avg_survived_pnl'] is not None:
        print(f"Average P&L (survived positions): ${results['avg_survived_pnl']:.2f}")
    
    if results['avg_liquidated_pnl'] is not None:
        print(f"Average P&L (liquidated positions): ${results['avg_liquidated_pnl']:.2f}")
    
    if results['avg_days_to_liquidation'] is not None:
        print(f"Average days to liquidation: {results['avg_days_to_liquidation']:.1f} days")
    
    # Plot the results
    fig = plot_simulation_results(results)
    plt.show()
    
    # Analyze optimal leverage
    print("\n=== Analyzing Optimal Leverage ===")
    safety_margins = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Use a subset of simulations for optimization to save time
    opt_params = params.copy()
    opt_params["num_simulations"] = 200
    
    fig_opt, opt_results = analyze_optimal_leverage(
        safety_margins=safety_margins,
        **opt_params
    )
    
    # Print optimal leverage results
    print("\nSafety Margin | Avg Leverage | Liquidation Rate | Avg ROI    | P&L Std Dev")
    print("-" * 80)
    for res in opt_results:
        print(f"{res['safety_margin']:.2f}         | {res['avg_max_leverage']:.2f}x        | "
              f"{res['liquidation_rate']:.2f}%           | {res['avg_roi']:.2f}%     | ${res['profit_loss_std']:.2f}")
    
    plt.show()