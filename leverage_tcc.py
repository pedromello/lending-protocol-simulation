import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings

# Ignora avisos de média de lista vazia para não poluir o output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def simulate_price_path(
    initial_price,
    drift,
    volatility,
    days,
    jump_intensity=0.1,
    jump_mean=-0.02,
    jump_std=0.08,
    enable_jumps=True,
):
    """
    Simula UMA trajetória de preço de ativo usando Modelo Jump-Diffusion de Merton.
    Combina Movimento Browniano Geométrico (GBM) com saltos estocásticos para
    capturar movimentos extremos típicos de mercados cripto.
    """
    dt = 1 / 365
    steps = days
    prices = np.zeros(steps + 1)
    prices[0] = initial_price

    random_shocks = np.random.normal(0, 1, steps)

    if enable_jumps:
        jump_times = np.random.poisson(jump_intensity * dt, steps)
        jump_components = np.zeros(steps)
        for t in range(steps):
            if jump_times[t] > 0:
                jumps = np.random.normal(jump_mean, jump_std, jump_times[t])
                jump_components[t] = np.sum(jumps)
    else:
        jump_components = np.zeros(steps)

    for t in range(1, steps + 1):
        diffusion_return = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(
            dt
        ) * random_shocks[t - 1]
        jump_return = jump_components[t - 1]
        prices[t] = prices[t - 1] * np.exp(diffusion_return + jump_return)

    return prices


def simulate_leveraged_position(
    initial_deposit_usd,
    price_path,
    ltv,
    liquidation_threshold,
    borrow_interest_rate,
    mechanism="Total",  # Novo parâmetro: 'Total' ou 'Parcial'
    liquidation_penalty=0.08,  # Penalidade de 8% sobre o colateral liquidado
    partial_liquidation_factor=0.5,  # 50% da dívida é paga na liquidação parcial
):
    """
    Simula UMA posição alavancada para uma DADA trajetória de preço, LTV e mecanismo de liquidação.

    Retorna:
    - Dicionário com os resultados:
        - 'liquidated_day': Dia da primeira liquidação. None se sobreviveu.
        - 'roi': ROI final.
        - 'liquidation_events': Contagem de eventos de liquidação (relevante para o modo 'Parcial').
        - 'status': 'Sobreviveu', 'Liquidado_Totalmente'.
    """
    initial_price = price_path[0]
    daily_interest_rate = borrow_interest_rate / 365

    # --- Setup da Posição Inicial ---
    initial_equity_usd = initial_deposit_usd
    total_collateral_value_usd = initial_equity_usd / (1 - ltv)
    total_collateral_units = total_collateral_value_usd / initial_price
    borrowed_amount_usd = total_collateral_value_usd * ltv

    liquidation_events = 0
    first_liquidation_day = None

    # Simulação dia a dia
    for day, current_price in enumerate(price_path[1:], 1):
        borrowed_amount_usd *= 1 + daily_interest_rate
        collateral_value_usd = total_collateral_units * current_price

        if collateral_value_usd > 0:
            current_ltv = borrowed_amount_usd / collateral_value_usd
        else:
            current_ltv = float("inf")

        if current_ltv >= liquidation_threshold:
            if first_liquidation_day is None:
                first_liquidation_day = day
            liquidation_events += 1

            if mechanism == "Total":
                return {
                    "liquidated_day": first_liquidation_day,
                    "roi": -1.0,
                    "liquidation_events": 1,
                    "status": "Liquidado_Totalmente",
                }

            elif mechanism == "Parcial":
                debt_to_repay = borrowed_amount_usd * partial_liquidation_factor
                collateral_to_seize_usd = debt_to_repay * (1 + liquidation_penalty)

                if current_price > 0:
                    collateral_units_to_seize = collateral_to_seize_usd / current_price
                else:
                    collateral_units_to_seize = float("inf")

                borrowed_amount_usd -= debt_to_repay
                total_collateral_units -= collateral_units_to_seize

                if total_collateral_units <= 0:
                    return {
                        "liquidated_day": first_liquidation_day,
                        "roi": -1.0,
                        "liquidation_events": liquidation_events,
                        "status": "Liquidado_Totalmente",
                    }

    # --- Se a posição sobreviveu ---
    final_collateral_value = total_collateral_units * price_path[-1]
    final_equity = final_collateral_value - borrowed_amount_usd
    roi = (final_equity - initial_equity_usd) / initial_equity_usd

    return {
        "liquidated_day": first_liquidation_day,
        "roi": roi,
        "liquidation_events": liquidation_events,
        "status": "Sobreviveu",
    }


def run_full_monte_carlo(
    num_simulations=1000,
    initial_deposit_usd=1000,
    initial_price=3000,
    simulation_days=365,
    market_params={"drift": 0.5, "volatility": 0.9},
    liquidation_threshold=0.85,
    liquidation_penalty=0.08,
    borrow_interest_rate=0.05,
    ltv_levels_to_test=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75],
    mechanisms_to_test=["Total", "Parcial"],
):
    """
    Executa a simulação de Monte Carlo completa para diferentes mecanismos.
    """
    results_by_mechanism = {
        mech: {ltv: [] for ltv in ltv_levels_to_test} for mech in mechanisms_to_test
    }
    price_paths_sample = []

    print(f"Executando simulação para o cenário: {market_params}")
    for i in tqdm(range(num_simulations), desc="Simulando Trajetórias de Preço"):
        price_path = simulate_price_path(
            initial_price,
            market_params["drift"],
            market_params["volatility"],
            simulation_days,
            jump_intensity=market_params.get("jump_intensity", 0.1),
            jump_mean=market_params.get("jump_mean", -0.02),
            jump_std=market_params.get("jump_std", 0.08),
        )
        if i < 100:
            price_paths_sample.append(price_path)

        for mechanism in mechanisms_to_test:
            for ltv in ltv_levels_to_test:
                result = simulate_leveraged_position(
                    initial_deposit_usd,
                    price_path,
                    ltv,
                    liquidation_threshold,
                    borrow_interest_rate,
                    mechanism,
                    liquidation_penalty,
                )
                results_by_mechanism[mechanism][ltv].append(result)

    # --- Agrega e analisa os resultados ---
    analysis = {mech: {} for mech in mechanisms_to_test}
    for mechanism, results_by_ltv in results_by_mechanism.items():
        for ltv, results in results_by_ltv.items():
            all_rois = [r["roi"] for r in results]
            liquidation_count = sum(
                1 for r in results if r["liquidated_day"] is not None
            )
            total_liquidation_count = sum(
                1 for r in results if r["status"] == "Liquidado_Totalmente"
            )
            surviving_rois = [r["roi"] for r in results if r["status"] == "Sobreviveu"]
            liquidation_event_counts = [
                r["liquidation_events"] for r in results if r["liquidation_events"] > 0
            ]

            analysis[mechanism][ltv] = {
                # NOVO: Cálculo do Retorno Esperado (média de todos os ROIs)
                "expected_roi": np.mean(all_rois) if all_rois else 0,
                "first_liquidation_probability": liquidation_count / num_simulations,
                "total_liquidation_probability": total_liquidation_count
                / num_simulations,
                "surviving_rois": surviving_rois,
                "average_roi_survivors": (
                    np.mean(surviving_rois) if surviving_rois else 0
                ),
                "median_roi_survivors": (
                    np.median(surviving_rois) if surviving_rois else 0
                ),
                "average_liquidation_events": (
                    np.mean(liquidation_event_counts) if liquidation_event_counts else 0
                ),
                "liquidation_event_counts": liquidation_event_counts,
            }

    return analysis, price_paths_sample


def plot_analysis_results(analysis, price_paths, market_scenario_name):
    """
    Gera os novos gráficos comparativos, incluindo o Retorno Esperado.
    """
    mechanisms = list(analysis.keys())
    ltv_levels = list(analysis[mechanisms[0]].keys())

    n_ltvs = len(ltv_levels)
    x = np.arange(n_ltvs)
    width = 0.35
    colors = {"Total": "salmon", "Parcial": "skyblue"}

    # Aumenta o layout para 2x2
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(
        f"Análise Comparativa de Mecanismos de Liquidação - Cenário: {market_scenario_name}",
        fontsize=18,
    )

    # --- Gráfico 1: Probabilidade de Primeira Liquidação ---
    ax1 = axes[0, 0]
    for i, mechanism in enumerate(mechanisms):
        probs = [
            analysis[mechanism][ltv]["total_liquidation_probability"] * 100
            for ltv in ltv_levels
        ]
        offset = width / 2 if i == 1 else -width / 2
        ax1.bar(
            x + offset,
            probs,
            width,
            label=mechanism,
            color=colors[mechanism],
            edgecolor="black",
        )
    ax1.set_title("Probabilidade de Ocorrer Liquidação Total")
    ax1.set_ylabel("Probabilidade (%)")
    ax1.set_xlabel("LTV Inicial")
    ax1.set_xticks(x, [f"{ltv:.0%}" for ltv in ltv_levels])
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # --- NOVO Gráfico 2: Esperança de Retorno (E[ROI]) ---
    ax2 = axes[0, 1]
    for i, mechanism in enumerate(mechanisms):
        e_rois = [analysis[mechanism][ltv]["expected_roi"] for ltv in ltv_levels]
        offset = width / 2 if i == 1 else -width / 2
        ax2.bar(
            x + offset,
            e_rois,
            width,
            label=mechanism,
            color=colors[mechanism],
            edgecolor="black",
        )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_title("Esperança de Retorno (E[ROI]) vs. LTV")
    ax2.set_ylabel("Retorno Esperado (E[ROI])")
    ax2.set_xlabel("LTV Inicial")
    ax2.set_xticks(x, [f"{ltv:.0%}" for ltv in ltv_levels])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Gráfico 3: Distribuição do ROI para Posições Sobreviventes ---
    ax3 = axes[1, 0]
    all_rois, positions, box_colors = [], [], []
    for i, ltv in enumerate(ltv_levels):
        for j, mechanism in enumerate(mechanisms):
            rois = analysis[mechanism][ltv]["surviving_rois"]
            if rois:
                all_rois.append(rois)
                positions.append(i + (j - 0.5) * width * 1.2)
                box_colors.append(colors[mechanism])
    bp = ax3.boxplot(
        all_rois,
        positions=positions,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    ax3.axhline(y=0, color="r", linestyle="--", alpha=0.7)
    ax3.set_title("Distribuição do ROI para Posições Sobreviventes")
    ax3.set_ylabel("Retorno sobre Investimento (ROI)")
    ax3.set_xlabel("LTV Inicial")
    ax3.set_xticks(x, [f"{ltv:.0%}" for ltv in ltv_levels])
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.legend([bp["boxes"][0], bp["boxes"][1]], mechanisms)
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Gráfico 4: Fan Chart das Trajetórias de Preço ---
    ax4 = axes[1, 1]
    for path in price_paths:
        ax4.plot(path, color="royalblue", alpha=0.1)
    ax4.set_title("Amostra de Trajetórias de Preço Simuladas")
    ax4.set_xlabel("Dias")
    ax4.set_ylabel("Preço ($)")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    market_scenarios = {
        "Bull Market (Otimista)": {
            "drift": 0.50,
            "volatility": 0.80,
            "jump_intensity": 0.10,
            "jump_mean": -0.02,
            "jump_std": 0.08,
        },
        "Bear Market (Pânico)": {
            "drift": -0.60,
            "volatility": 1.20,
            "jump_intensity": 0.15,
            "jump_mean": -0.04,
            "jump_std": 0.12,
        },
        "Mercado Lateral (Volátil)": {
            "drift": 0.05,
            "volatility": 0.60,
            "jump_intensity": 0.05,
            "jump_mean": -0.01,
            "jump_std": 0.05,
        },
    }

    simulation_params = {
        "num_simulations": 2000,
        "initial_deposit_usd": 1000,
        "initial_price": 3000,
        "simulation_days": 365,
        "liquidation_threshold": 0.85,
        "liquidation_penalty": 0.08,
        "borrow_interest_rate": 0.05,
        "ltv_levels_to_test": [0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.80],
    }

    for name, params in market_scenarios.items():
        analysis_results, price_paths = run_full_monte_carlo(
            market_params=params, **simulation_params
        )

        # --- NOVO: Imprime a Esperança de Retorno no terminal ---
        print("\n" + "=" * 60)
        print(f"Resultado: Esperança de Retorno (E[ROI]) para '{name}'")
        print("=" * 60)
        header = (
            "| LTV   | "
            + " | ".join([f"{mech:<10}" for mech in analysis_results.keys()])
            + " |"
        )
        print(header)
        print("|" + "-" * 7 + "|" + ("-" * 12 + "|") * len(analysis_results.keys()))
        for ltv in simulation_params["ltv_levels_to_test"]:
            row = f"| {ltv:<5.0%} |"
            for mech in analysis_results.keys():
                e_roi = analysis_results[mech][ltv]["expected_roi"]
                row += f" {e_roi:<+10.2%} |"
            print(row)
        print("=" * 60 + "\n")

        # Salvar resultados em JSON, agora incluindo o E[ROI]
        serializable_results = {}
        for mech, data_by_ltv in analysis_results.items():
            serializable_results[mech] = {}
            for ltv, data in data_by_ltv.items():
                serializable_results[mech][ltv] = {
                    k: v
                    for k, v in data.items()
                    if k not in ["surviving_rois", "liquidation_event_counts"]
                }

        with open(f"analysis_results_{name.replace(' ', '_')}.json", "w") as f:
            json.dump(serializable_results, f, indent=4)

        plot_analysis_results(analysis_results, price_paths, name)
