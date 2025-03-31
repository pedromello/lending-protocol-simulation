import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

# Basic dataclasses to represent our agents
@dataclass
class Borrower:
    id: int
    collateral: float
    risk_tolerance: float  # Higher means more willing to default strategically
    liquidation_threshold: float  # Collateral value / Loan value ratio at which liquidation occurs
    
    def decide_to_borrow(self, interest_rate, collateral_factor):
        # Will borrow if interest rate is acceptable based on risk tolerance
        return interest_rate <= (0.08 + self.risk_tolerance * 0.1)
    
    def decide_to_default(self, collateral_price, debt_value):
        collateral_value = self.collateral * collateral_price
        
        # Se o colateral vale significativamente mais que a dívida, ninguém faz default
        if collateral_value > debt_value * 1.5:
            return False
            
        # Default quando o valor do colateral cai abaixo de um limiar
        # Pessoas com alta tolerância ao risco têm um limiar mais baixo (aguentam mais prejuízo)
        return collateral_value < debt_value * (1 - self.risk_tolerance * 0.5)


@dataclass
class Lender:
    id: int
    liquidity: float
    risk_appetite: float  # Higher means more willing to lend at lower rates
    
    def decide_to_lend(self, interest_rate, utilization_rate):
        # Will lend if interest rate is acceptable compared to risk
        min_acceptable_rate = 0.03 + (1 - self.risk_appetite) * 0.15 * utilization_rate
        return interest_rate >= min_acceptable_rate


class LendingProtocol:
    def __init__(self, base_rate=0.03, slope1=0.1, slope2=0.4, kink=0.8):
        self.borrowers = []
        self.lenders = []
        self.loans = {}  # borrower_id -> (loan_amount, interest_rate)
        self.deposits = {}  # lender_id -> liquidity_provided
        self.total_liquidity = 0
        self.total_borrowed = 0
        self.collateral_factor = 0.75  # Max loan-to-value ratio
        
        # Interest rate model parameters (similar to Compound/Aave)
        self.base_rate = base_rate
        self.slope1 = slope1  # Slope before kink
        self.slope2 = slope2  # Slope after kink
        self.kink = kink  # Utilization point where the slope changes
        
        # Market conditions
        self.collateral_price = 1.0
        self.price_volatility = 0.05
        self.default_penalty = 0.1
        
        # Metrics tracking
        self.interest_rates = [base_rate]
        self.utilization_rates = [0]
        self.default_rates = [0]
        self.total_liquidations = 0
        self.collateral_price_history = [self.collateral_price]
        
        # Protocol profit tracking
        self.protocol_fee = 0.1  # 10% of interest goes to protocol
        self.protocol_profit_per_step = [0]
        self.protocol_cumulative_profit = [0]
        self.liquidation_penalty = 0.08  # 8% penalty on defaulted loans that gets liquidated
        
        # New metrics for participant and money tracking
        self.active_lenders_count = [0]
        self.active_borrowers_count = [0]
        self.pool_size_history = [0]  # Total money in the pool in dollars
        self.borrowed_amount_history = [0]  # Amount being borrowed in dollars

    def add_borrower(self, collateral, risk_tolerance=None, liquidation_threshold=None):
        if risk_tolerance is None:
            risk_tolerance = random.uniform(0.1, 0.9)
        if liquidation_threshold is None:
            liquidation_threshold = 0.8  # Common liquidation threshold
            
        borrower = Borrower(
            id=len(self.borrowers),
            collateral=collateral,
            risk_tolerance=risk_tolerance,
            liquidation_threshold=liquidation_threshold
        )
        self.borrowers.append(borrower)
        return borrower
    
    def add_lender(self, liquidity, risk_appetite=None):
        if risk_appetite is None:
            risk_appetite = random.uniform(0.3, 0.9)
            
        lender = Lender(
            id=len(self.lenders),
            liquidity=liquidity,
            risk_appetite=risk_appetite
        )
        self.lenders.append(lender)
        
        # Add liquidity to protocol
        self.deposits[lender.id] = liquidity
        self.total_liquidity += liquidity
        return lender
    
    def calculate_interest_rate(self):
        if self.total_liquidity == 0:
            return self.base_rate
        
        utilization_rate = self.total_borrowed / self.total_liquidity
        
        if utilization_rate <= self.kink:
            return self.base_rate + (utilization_rate / self.kink) * self.slope1
        else:
            return self.base_rate + self.slope1 + ((utilization_rate - self.kink) / (1 - self.kink)) * self.slope2
    
    def get_utilization_rate(self):
        if self.total_liquidity == 0:
            return 0
        return self.total_borrowed / self.total_liquidity
    
    def update_market_conditions(self):
        # Update collateral price with random walk
        price_change = np.random.normal(0, self.price_volatility)
        self.collateral_price *= (1 + price_change)
        
        # Ensure price doesn't go negative
        self.collateral_price = max(0.1, self.collateral_price)
    
    def process_borrowing(self):
        interest_rate = self.calculate_interest_rate()
        utilization_rate = self.get_utilization_rate()
        
        new_loans = 0
        
        for borrower in self.borrowers:
            if borrower.id in self.loans:
                continue  # Already has a loan
                
            max_borrow_amount = borrower.collateral * self.collateral_price * self.collateral_factor
            
            if max_borrow_amount > 0 and borrower.decide_to_borrow(interest_rate, self.collateral_factor):
                # Check if there's enough liquidity
                if max_borrow_amount <= (self.total_liquidity - self.total_borrowed):
                    self.loans[borrower.id] = (max_borrow_amount, interest_rate)
                    self.total_borrowed += max_borrow_amount
                    new_loans += 1
                    
        return new_loans
    
    def process_lending(self):
        interest_rate = self.calculate_interest_rate()
        utilization_rate = self.get_utilization_rate()
        
        for lender in self.lenders:
            # For simplicity, lenders either provide all liquidity or none
            if lender.id not in self.deposits and lender.decide_to_lend(interest_rate, utilization_rate):
                self.deposits[lender.id] = lender.liquidity
                self.total_liquidity += lender.liquidity
    
    def calculate_protocol_profit(self):
        # Calculate interest earned from active loans
        interest_profit = 0
        for loan_amount, loan_interest in self.loans.values():
            # Calculate interest for this step (assuming interest per step, not compound)
            step_interest = loan_amount * loan_interest / 52  # Weekly interest if we assume 52 steps per year
            interest_profit += step_interest * self.protocol_fee
        
        return interest_profit
    
    def process_defaults_and_liquidations(self):
        defaults = 0
        liquidations = 0
        liquidation_profit = 0
        
        borrowers_to_remove = []
        
        for borrower_id, (loan_amount, loan_interest) in list(self.loans.items()):
            borrower = self.borrowers[borrower_id]
            
            # Calculate current debt including interest
            current_debt = loan_amount * (1 + loan_interest)
            
            # Check if borrower decides to default strategically
            if borrower.decide_to_default(self.collateral_price, current_debt):
                self.total_borrowed -= loan_amount
                defaults += 1
                borrowers_to_remove.append(borrower_id)
                
                # When default occurs, protocol gains liquidation fee
                liquidation_profit += current_debt * self.liquidation_penalty
                continue
                
            # Check if liquidation should occur
            collateral_value = borrower.collateral * self.collateral_price
            if collateral_value < current_debt * borrower.liquidation_threshold:
                # Liquidation happens
                self.total_borrowed -= loan_amount
                liquidations += 1
                self.total_liquidations += 1
                borrowers_to_remove.append(borrower_id)
                
                # Protocol gains liquidation fee on liquidations
                liquidation_profit += current_debt * self.liquidation_penalty
        
        # Remove defaulted/liquidated borrowers
        for borrower_id in borrowers_to_remove:
            del self.loans[borrower_id]
            
        default_rate = defaults / len(self.borrowers) if self.borrowers else 0
        return defaults, liquidations, default_rate, liquidation_profit
    
    def simulate_step(self):
        # Poisson distribution for new borrowers and lenders
        new_borrowers = np.random.poisson(1)  # Average of 1 new borrower per step
        new_lenders = np.random.poisson(1)  # Average of 0.5 new lenders per step
        
        # Add new lenders
        for _ in range(new_lenders):
            liquidity = random.uniform(100, 1000)
            self.add_lender(liquidity)
        
        # Add new borrowers
        for _ in range(new_borrowers):
            collateral = random.uniform(50, 500)
            self.add_borrower(collateral)
        
        self.update_market_conditions()
        self.process_lending()
        new_loans = self.process_borrowing()
        
        # Calculate interest-based profit for the protocol
        interest_profit = self.calculate_protocol_profit()
        
        # Process defaults and get liquidation profit
        defaults, liquidations, default_rate, liquidation_profit = self.process_defaults_and_liquidations()
        
        # Total profit for this step is interest profit plus liquidation profit
        step_profit = interest_profit + liquidation_profit
        
        interest_rate = self.calculate_interest_rate()
        utilization_rate = self.get_utilization_rate()
        
        # Record metrics
        self.interest_rates.append(interest_rate)
        self.utilization_rates.append(utilization_rate)
        self.default_rates.append(default_rate)
        self.collateral_price_history.append(self.collateral_price)
        
        # Record profit metrics
        self.protocol_profit_per_step.append(step_profit)
        
        # Calculate and record cumulative profit
        previous_cumulative = self.protocol_cumulative_profit[-1]
        self.protocol_cumulative_profit.append(previous_cumulative + step_profit)
        
        # Track participant and money metrics
        active_lenders = len(self.deposits)
        active_borrowers = len(self.loans)
        self.active_lenders_count.append(active_lenders)
        self.active_borrowers_count.append(active_borrowers)
        self.pool_size_history.append(self.total_liquidity)
        self.borrowed_amount_history.append(self.total_borrowed)
        
        return {
            'interest_rate': interest_rate,
            'utilization_rate': utilization_rate,
            'new_loans': new_loans,
            'defaults': defaults,
            'liquidations': liquidations,
            'default_rate': default_rate,
            'collateral_price': self.collateral_price,
            'total_borrowed': self.total_borrowed,
            'total_liquidity': self.total_liquidity,
            'step_profit': step_profit,
            'cumulative_profit': self.protocol_cumulative_profit[-1],
            'active_lenders': active_lenders,
            'active_borrowers': active_borrowers
        }
    
    def simulate(self, steps):
        results = []
        for _ in range(steps):
            step_result = self.simulate_step()
            results.append(step_result)
        return results
    
    def plot_results(self):
        steps = range(len(self.interest_rates))
        
        # Create a figure with 6 subplots (5 original + 1 new for participants/money)
        fig, axs = plt.subplots(6, 1, figsize=(12, 18))
        
        # Plot interest rates
        axs[0].plot(steps, self.interest_rates, 'b-', label='Interest Rate')
        axs[0].set_title('Interest Rate Over Time')
        axs[0].set_ylabel('Rate')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot utilization rates
        axs[1].plot(steps, self.utilization_rates, 'g-', label='Utilization Rate')
        axs[1].set_title('Utilization Rate Over Time')
        axs[1].set_ylabel('Rate')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot default rates
        axs[2].plot(steps, self.default_rates, 'r-', label='Default Rate')
        axs[2].set_title('Default Rate Over Time')
        axs[2].set_ylabel('Rate')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot collateral price history
        axs[3].plot(steps, self.collateral_price_history, 'm-', label='Collateral Price')
        axs[3].set_title('Collateral Price Over Time')
        axs[3].set_ylabel('Price')
        axs[3].legend()
        axs[3].grid(True)
        
        # Plot protocol cumulative profit
        axs[4].plot(steps, self.protocol_cumulative_profit, 'c-', label='Protocol Cumulative Profit')
        axs[4].set_title('Protocol Cumulative Profit Over Time')
        axs[4].set_ylabel('Profit ($)')
        axs[4].legend()
        axs[4].grid(True)
        
        # NEW PLOT: Participants and Money metrics
        # Create a twin axis for the money metrics
        ax6_twin = axs[5].twinx()
        
        # Plot participant counts on left y-axis
        lender_line, = axs[5].plot(steps, self.active_lenders_count, 'b-', label='Active Lenders')
        borrower_line, = axs[5].plot(steps, self.active_borrowers_count, 'r-', label='Active Borrowers')
        
        # Plot money metrics on right y-axis
        pool_line, = ax6_twin.plot(steps, self.pool_size_history, 'g-', label='Total Pool Size ($)')
        borrowed_line, = ax6_twin.plot(steps, self.borrowed_amount_history, 'y-', label='Total Borrowed ($)')
        
        # Set titles and labels
        axs[5].set_title('Participants and Money in the Pool')
        axs[5].set_xlabel('Simulation Step')
        axs[5].set_ylabel('Number of Participants')
        ax6_twin.set_ylabel('Amount ($)')
        
        # Create combined legend
        lines = [lender_line, borrower_line, pool_line, borrowed_line]
        labels = [line.get_label() for line in lines]
        axs[5].legend(lines, labels, loc='upper left')
        
        axs[5].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_participants_and_money(self):
        """Additional method to plot only the participants and money metrics"""
        steps = range(len(self.active_lenders_count))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Create a second y-axis
        ax2 = ax1.twinx()
        
        # Plot participant counts on left y-axis
        lender_line, = ax1.plot(steps, self.active_lenders_count, 'b-', label='Active Lenders')
        borrower_line, = ax1.plot(steps, self.active_borrowers_count, 'r-', label='Active Borrowers')
        
        # Plot money metrics on right y-axis
        pool_line, = ax2.plot(steps, self.pool_size_history, 'g-', label='Total Pool Size ($)')
        borrowed_line, = ax2.plot(steps, self.borrowed_amount_history, 'y-', label='Total Borrowed ($)')
        
        # Set titles and labels
        ax1.set_title('Participants and Money in the Protocol')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Number of Participants')
        ax2.set_ylabel('Amount ($)')
        
        # Create combined legend
        lines = [lender_line, borrower_line, pool_line, borrowed_line]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
def run_simulation():
    # Initialize protocol
    protocol = LendingProtocol(base_rate=0.02, slope1=0.2, slope2=0.6, kink=0.7)
    
    # Add lenders
    for _ in range(10):
        liquidity = random.uniform(100, 1000)
        protocol.add_lender(liquidity)
    
    # Add borrowers
    for _ in range(30):
        collateral = random.uniform(50, 500)
        protocol.add_borrower(collateral)
    
    # Run simulation
    results = protocol.simulate(1000)
    
    # Plot results
    protocol.plot_results()
    
    # Optionally plot just the participants and money metrics in a separate figure
    # protocol.plot_participants_and_money()
    
    # Return final state
    return {
        'final_interest_rate': protocol.interest_rates[-1],
        'final_utilization': protocol.utilization_rates[-1],
        'total_liquidations': protocol.total_liquidations,
        'borrowers_with_loans': len(protocol.loans),
        'total_borrowed': protocol.total_borrowed,
        'total_liquidity': protocol.total_liquidity,
        'final_cumulative_profit': protocol.protocol_cumulative_profit[-1],
        'final_lender_count': protocol.active_lenders_count[-1],
        'final_borrower_count': protocol.active_borrowers_count[-1]
    }

# Run multiple simulations with different parameters to find optimal strategy
def parameter_sweep():
    results = []
    
    base_rates = [0.01, 0.02, 0.03, 0.05]
    slopes = [0.05, 0.1, 0.15, 0.2]
    kinks = [0.6, 0.7, 0.8, 0.9]
    
    for base_rate in base_rates:
        for slope in slopes:
            for kink in kinks:
                protocol = LendingProtocol(base_rate=base_rate, slope1=slope, slope2=slope*3, kink=kink)
                
                # Add standard set of agents
                for _ in range(10):
                    protocol.add_lender(random.uniform(100, 1000))
                
                for _ in range(30):
                    protocol.add_borrower(random.uniform(50, 500))
                
                # Run simulation
                protocol.simulate(100)
                
                # Collect results
                results.append({
                    'base_rate': base_rate,
                    'slope': slope,
                    'kink': kink,
                    'final_interest_rate': protocol.interest_rates[-1],
                    'average_utilization': np.mean(protocol.utilization_rates),
                    'max_default_rate': max(protocol.default_rates),
                    'total_liquidations': protocol.total_liquidations,
                    'cumulative_profit': protocol.protocol_cumulative_profit[-1],
                    'profit_to_liquidity_ratio': protocol.protocol_cumulative_profit[-1] / protocol.total_liquidity if protocol.total_liquidity > 0 else 0,
                    'average_lenders': np.mean(protocol.active_lenders_count),
                    'average_borrowers': np.mean(protocol.active_borrowers_count),
                    'final_pool_size': protocol.pool_size_history[-1]
                })
    
    # Find optimal parameters based on profit
    results.sort(key=lambda x: x['cumulative_profit'], reverse=True)
    return results[:5]  # Return top 5 parameter combinations

if __name__ == "__main__":
    # Run a single simulation
    final_state = run_simulation()
    print("Final simulation state:", final_state)
    
    # Uncomment to run parameter sweep
    # optimal_params = parameter_sweep()
    # print("Optimal parameters:", optimal_params)