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

    Parâmetros:
    - initial_price: Preço inicial do ativo.
    - drift: Retorno anual esperado (μ) sem considerar saltos.
    - volatility: Volatilidade anual (σ) da difusão contínua.
    - days: Número de dias a simular.
    - jump_intensity: Taxa de chegada de saltos por dia (λ). Default: 0.1 (1 salto a cada 10 dias).
    - jump_mean: Retorno médio de um salto (log-normal). Default: -2% (saltos negativos mais comuns).
    - jump_std: Volatilidade dos saltos. Default: 8%.
    - enable_jumps: Se False, reverte para GBM clássico.

    Retorna:
    - Array de preços simulados para o período.
    """
    dt = 1 / 365
    steps = days
    prices = np.zeros(steps + 1)
    prices[0] = initial_price

    # Gera todos os choques aleatórios de uma vez para eficiência
    random_shocks = np.random.normal(0, 1, steps)

    # Gera componente de saltos se habilitado
    if enable_jumps:
        # Determina quando ocorrem saltos (processo de Poisson)
        jump_times = np.random.poisson(jump_intensity * dt, steps)

        # Para cada passo de tempo, gera os saltos (se houver)
        jump_components = np.zeros(steps)
        for t in range(steps):
            if jump_times[t] > 0:
                # Soma de múltiplos saltos no mesmo período (caso raro)
                jumps = np.random.normal(jump_mean, jump_std, jump_times[t])
                jump_components[t] = np.sum(jumps)
    else:
        jump_components = np.zeros(steps)

    # Simula preços com componente de difusão + saltos
    for t in range(1, steps + 1):
        # Componente de difusão (GBM padrão)
        diffusion_return = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(
            dt
        ) * random_shocks[t - 1]

        # Componente de salto
        jump_return = jump_components[t - 1]

        # Preço final combinando difusão e saltos
        prices[t] = prices[t - 1] * np.exp(diffusion_return + jump_return)

    return prices


def simulate_leveraged_position(
    initial_deposit, price_path, ltv, liquidation_threshold, borrow_interest_rate
):
    """
    Simula UMA posição alavancada para uma DADA trajetória de preço e um LTV específico.

    Parâmetros:
    - initial_deposit: Depósito inicial em unidades de colateral (ex: 1 ETH).
    - price_path: Array de preços do ativo ao longo do tempo.
    - ltv: Loan-to-Value inicial (ex: 0.6 para 60%).
    - liquidation_threshold: Limite de LTV para liquidação (ex: 0.85).
    - borrow_interest_rate: Taxa de juros anual do empréstimo.

    Retorna:
    - Dicionário com os resultados: dia da liquidação (ou None) e ROI (ou -1 se liquidado).
    """
    initial_price = price_path[0]
    daily_interest_rate = borrow_interest_rate / 365

    # --- Setup da Posição Inicial ---
    # O depósito inicial do usuário é o seu capital (equity)
    # A alavancagem é calculada a partir do LTV desejado
    # Equity = Colateral - Dívida  =>  D = C - (C * LTV) => D = C * (1 - LTV) => C = D / (1 - LTV)
    total_collateral_units = initial_deposit / (1 - ltv)
    borrowed_amount_usd = total_collateral_units * initial_price * ltv

    # Simulação dia a dia
    for day, current_price in enumerate(price_path[1:], 1):
        # A dívida aumenta a cada dia devido aos juros compostos
        borrowed_amount_usd *= 1 + daily_interest_rate

        # O valor do colateral flutua com o preço
        collateral_value_usd = total_collateral_units * current_price

        # Calcula o LTV atual da posição
        current_ltv = borrowed_amount_usd / collateral_value_usd

        # Verifica se a posição foi liquidada
        if current_ltv >= liquidation_threshold:
            return {"liquidated_day": day, "roi": -1}  # ROI de -100% na liquidação

    # --- Se a posição sobreviveu, calcula o ROI ---
    final_collateral_value = total_collateral_units * price_path[-1]
    final_equity = final_collateral_value - borrowed_amount_usd
    initial_equity = initial_deposit * initial_price

    roi = (final_equity - initial_equity) / initial_equity

    return {"liquidated_day": None, "roi": roi}


def run_full_monte_carlo(
    num_simulations=1000,
    initial_deposit=1.0,  # 1 ETH
    initial_price=3000,  # $3000 por ETH
    simulation_days=365,  # 1 ano
    market_params={"drift": 0.5, "volatility": 0.9},  # Parâmetros do mercado
    liquidation_threshold=0.85,
    borrow_interest_rate=0.05,  # 5% de juros anuais sobre o empréstimo
    ltv_levels_to_test=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
    ],  # LTVs fixos para testar
):
    """
    Executa a simulação de Monte Carlo completa.
    Gera MÚLTIPLAS trajetórias de preço e testa cada LTV em cada uma delas.
    """
    # Estrutura para armazenar todos os resultados
    # Ex: results_by_ltv[0.5] = [{'roi': 1.2, 'liquidated_day': None}, {'roi': -1, 'liquidated_day': 25}, ...]
    results_by_ltv = {ltv: [] for ltv in ltv_levels_to_test}

    # Armazena algumas trajetórias de preço para o gráfico "fan chart"
    price_paths_sample = []

    print(f"Executando simulação para o cenário: {market_params}")
    for i in tqdm(range(num_simulations), desc="Simulando Trajetórias de Preço"):
        # 1. Gera uma nova trajetória de preço (um futuro possível)
        price_path = simulate_price_path(
            initial_price,
            market_params["drift"],
            market_params["volatility"],
            simulation_days,
            jump_intensity=market_params.get("jump_intensity", 0.1),
            jump_mean=market_params.get("jump_mean", -0.02),
            jump_std=market_params.get("jump_std", 0.08),
            enable_jumps=market_params.get("enable_jumps", True),
        )
        if i < 100:  # Salva as primeiras 100 para visualização
            price_paths_sample.append(price_path)

        # 2. Testa cada nível de LTV nesta trajetória
        for ltv in ltv_levels_to_test:
            result = simulate_leveraged_position(
                initial_deposit,
                price_path,
                ltv,
                liquidation_threshold,
                borrow_interest_rate,
            )
            results_by_ltv[ltv].append(result)

    # --- Agrega e analisa os resultados ---
    analysis = {}
    for ltv, results in results_by_ltv.items():
        liquidation_count = sum(1 for r in results if r["liquidated_day"] is not None)
        surviving_rois = [r["roi"] for r in results if r["liquidated_day"] is None]

        analysis[ltv] = {
            "liquidation_probability": liquidation_count / num_simulations,
            "surviving_rois": surviving_rois,
            "average_roi": np.mean(surviving_rois) if surviving_rois else 0,
            "median_roi": np.median(surviving_rois) if surviving_rois else 0,
        }

    return analysis, price_paths_sample


def plot_analysis_results(analysis, price_paths, market_scenario_name):
    """
    Gera os novos gráficos para a análise aprofundada.
    """
    ltv_levels = list(analysis.keys())
    liquidation_probs = [
        data["liquidation_probability"] * 100 for data in analysis.values()
    ]
    surviving_rois_data = [data["surviving_rois"] for data in analysis.values()]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f"Análise de Risco e Retorno para Cenário: {market_scenario_name}", fontsize=16
    )

    # --- Gráfico 1: Probabilidade de Liquidação vs. LTV ---
    axes[0].bar(
        ltv_levels, liquidation_probs, width=0.04, color="salmon", edgecolor="black"
    )
    axes[0].set_title("Probabilidade de Liquidação vs. LTV Inicial")
    axes[0].set_xlabel("Loan-to-Value (LTV) Inicial")
    axes[0].set_ylabel("Probabilidade de Liquidação (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # --- Gráfico 2: Distribuição do ROI vs. LTV (Boxplot) ---
    axes[1].boxplot(
        surviving_rois_data,
        labels=[f"{ltv:.0%}" for ltv in ltv_levels],
        patch_artist=True,
    )
    axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.7)
    axes[1].set_title("Distribuição do ROI para Posições Sobreviventes")
    axes[1].set_xlabel("Loan-to-Value (LTV) Inicial")
    axes[1].set_ylabel("Retorno sobre Investimento (ROI)")
    # Formata o eixo Y para porcentagem
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # --- Gráfico 3: Fan Chart das Trajetórias de Preço ---
    for path in price_paths:
        axes[2].plot(path, color="royalblue", alpha=0.1)
    axes[2].set_title("Amostra de Trajetórias de Preço Simuladas (Fan Chart)")
    axes[2].set_xlabel("Dias")
    axes[2].set_ylabel("Preço ($)")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # --- Defina os Cenários de Mercado para Testar ---
    market_scenarios = {
        "Bull Market (Alta Volatilidade)": {
            "drift": 0.80,
            "volatility": 0.90,
            "jump_intensity": 0.08,  # Menos saltos em bull market
            "jump_mean": 0.01,  # Saltos ligeiramente positivos
            "jump_std": 0.06,
        },
        "Bear Market (Pânico)": {
            "drift": -0.60,
            "volatility": 1.20,
            "jump_intensity": 0.15,  # Mais saltos em bear market
            "jump_mean": -0.04,  # Saltos mais negativos
            "jump_std": 0.12,
        },
        "Mercado Lateral (Estável)": {
            "drift": 0.05,
            "volatility": 0.40,
            "jump_intensity": 0.05,  # Poucos saltos em mercado estável
            "jump_mean": -0.01,  # Saltos pequenos
            "jump_std": 0.04,
        },
    }

    # --- Parâmetros Gerais da Simulação ---
    simulation_params = {
        "num_simulations": 2000,
        "initial_deposit": 1.0,
        "initial_price": 3000,
        "simulation_days": 180,  # Horizonte de 6 meses
        "liquidation_threshold": 0.85,
        "borrow_interest_rate": 0.05,
        "ltv_levels_to_test": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80],
    }

    # Executa a simulação para cada cenário definido
    for name, params in market_scenarios.items():
        analysis_results, price_paths = run_full_monte_carlo(
            market_params=params, **simulation_params
        )

        # Salva os resultados agregados em um arquivo JSON (opcional)
        # É preciso converter os dados numpy para listas para serializar
        serializable_results = {}
        for ltv, data in analysis_results.items():
            serializable_results[ltv] = {
                "liquidation_probability": data["liquidation_probability"],
                "average_roi": data["average_roi"],
                "median_roi": data["median_roi"],
                "surviving_rois_count": len(data["surviving_rois"]),
            }

        with open(f"analysis_results_{name.replace(' ', '_')}.json", "w") as f:
            json.dump(serializable_results, f, indent=4)

        plot_analysis_results(analysis_results, price_paths, name)
