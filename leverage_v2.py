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
    # Generate random leverage between 1 and 5
    leverage = np.random.uniform(1, 5)
    
    # Calculate total collateral and borrowed amount based on random leverage
    total_collateral = initial_deposit * leverage
    total_borrowed = total_collateral - initial_deposit
    
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
    price_path = []
    
        # Generate price path
    price_path = simulate_price_path(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        days=simulation_days,
        dt=dt
    )
    
    # Run simulations
    for _ in tqdm(range(num_simulations), desc="Running simulations"):

        
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
        "liquidation_days": liquidation_days,
        "price_path": price_path
    }
    
    return results

def plot_simulation_results(results):
    """
    Plot the results of the Monte Carlo simulation
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Plot 1: Leverage vs Returns
    # Calculate returns for each leverage level
    unique_leverages = sorted(list(set([round(x, 1) for x in results["max_leverages"]])))
    avg_returns = []
    
    for lev in unique_leverages:
        # Find all instances with this leverage level
        indices = [i for i, x in enumerate(results["max_leverages"]) 
                  if round(x, 1) == lev]
        
        # Calculate returns for these positions
        returns = [(results["price_path"][-1] / results["price_path"][0] - 1) * lev 
                  if i not in results["liquidation_days"] else -1 
                  for i in indices]
        
        avg_returns.append(np.mean(returns))
    
    axes[0, 0].plot(unique_leverages, avg_returns, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title("Alavancagem vs. Retornos Esperados")
    axes[0, 0].set_xlabel("Multiplicador de Alavancagem")
    axes[0, 0].set_ylabel("Retorno Médio")
    
    # Find and mark the optimal leverage point
    optimal_leverage = unique_leverages[np.argmax(avg_returns)]
    max_return = max(avg_returns)
    axes[0, 0].plot(optimal_leverage, max_return, 'ro', 
                    label=f'Ótimo: {optimal_leverage:.1f}x')
    axes[0, 0].legend()
    
    # Plot 2: Distribution of days to liquidation
    if results["liquidation_days"]:
        axes[0, 1].hist(results["liquidation_days"], bins=30, alpha=0.7, color='red')
        axes[0, 1].axvline(results["avg_days_to_liquidation"], color='blue', linestyle='--', 
                          label=f'Média: {results["avg_days_to_liquidation"]:.1f} dias')
        axes[0, 1].set_title("Dias até Liquidação")
        axes[0, 1].set_xlabel("Dias")
        axes[0, 1].set_ylabel("Frequência")
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "Nenhuma liquidação ocorreu", 
                       horizontalalignment='center', verticalalignment='center')
        axes[0, 1].set_title("Dias até Liquidação")
    
    # Plot 3: Survival rate pie chart
    survival_rate = 100 - results["liquidation_rate"]
    axes[1, 0].pie([results["liquidation_rate"], survival_rate], 
                  labels=[f'Liquidado ({results["liquidation_rate"]:.1f}%)', 
                          f'Sobreviveu ({survival_rate:.1f}%)'],
                  colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title("Taxa de Liquidação vs. Sobrevivência")
    
    # Plot 4: Comparative leverage box plot
    leverage_data = []
    labels = []
    
    leverage_data.append(results["max_leverages"])
    labels.append("Todas Posições")
    
    if results.get("avg_survived_leverage") is not None:
        survived_max_leverages = [lev for i, lev in enumerate(results["max_leverages"]) 
                                 if i in [results["liquidation_days"].index(day) 
                                             for day in results["liquidation_days"]]]
        leverage_data.append(survived_max_leverages)
        labels.append("Posições Sobreviventes")
    
    if results.get("avg_liquidated_leverage") is not None:
        liquidated_max_leverages = [lev for i, lev in enumerate(results["max_leverages"]) 
                                   if i not in [results["liquidation_days"].index(day) 
                                           for day in results["liquidation_days"]]]
        leverage_data.append(liquidated_max_leverages)
        labels.append("Posições Liquidadas")
    
    axes[1, 1].boxplot(leverage_data, labels=labels)
    axes[1, 1].set_title("Comparação de Alavancagem")
    axes[1, 1].set_ylabel("Alavancagem")
    
    # Plot 5: Price path example
    axes[2, 0].plot(results["price_path"], color='blue')
    axes[2, 0].set_title("Caminho de Preço Simulado")
    axes[2, 0].set_xlabel("Dias")
    axes[2, 0].set_ylabel("Preço ($)")
    
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
        "volatility": 0.30,             # 80% annual volatility (crypto is volatile!)
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