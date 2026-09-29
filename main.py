import random

from values import VALUES
from PRICE_CALCULATIONS.cycles import OrnsteinUhlenbeckFundamental
from AGENTS.random_agent import RandomAgent
from AGENTS.momentum_agent import MomentumAgent
from AGENTS.mean_reversion_agent import MeanReversionAgent
from AGENTS.buy_and_hold_agent import BuyAndHoldAgent
from AGENTS.contrarian_agent import ContrarianAgent
from AGENTS.volume_pressure_agent import VolumePressureAgent
from AGENTS.value_estimate_agent import ValueEstimateAgent
from AGENTS.risk_managed_agent import RiskManagedAgent
from market import Market


def main():
    random.seed(VALUES["seed"])
    agents = [
        RandomAgent("Random1", trade_fraction=0.20, hard_cap=80),
        RandomAgent("Random2", trade_fraction=0.30, hard_cap=100),
        RandomAgent("Random3", trade_fraction=0.40, hard_cap=120),
        MomentumAgent("Momentum1", lookback=3, trade_fraction=0.15, hard_cap=80),
        MomentumAgent("Momentum2", lookback=8, trade_fraction=0.20, hard_cap=100),
        MomentumAgent("Momentum3", lookback=15, trade_fraction=0.25, hard_cap=120),
        MeanReversionAgent("MeanReversion1", lookback=10, threshold=0.015, trade_fraction=0.15, hard_cap=80),
        MeanReversionAgent("MeanReversion2", lookback=20, threshold=0.02, trade_fraction=0.20, hard_cap=100),
        MeanReversionAgent("MeanReversion3", lookback=40, threshold=0.03, trade_fraction=0.25, hard_cap=120),
        BuyAndHoldAgent("BuyAndHold1", buy_fraction=0.60),
        BuyAndHoldAgent("BuyAndHold2", buy_fraction=0.75),
        BuyAndHoldAgent("BuyAndHold3", buy_fraction=0.90),
        ContrarianAgent("Contrarian1", threshold=0.006, trade_fraction=0.15, hard_cap=80),
        ContrarianAgent("Contrarian2", threshold=0.008, trade_fraction=0.20, hard_cap=100),
        ContrarianAgent("Contrarian3", threshold=0.012, trade_fraction=0.25, hard_cap=120),
        VolumePressureAgent("VolumePressure1", sensitivity=0.20, trade_fraction=0.15, hard_cap=80),
        VolumePressureAgent("VolumePressure2", sensitivity=0.30, trade_fraction=0.20, hard_cap=100),
        VolumePressureAgent("VolumePressure3", sensitivity=0.40, trade_fraction=0.25, hard_cap=120),
        ValueEstimateAgent("ValueEstimate1", lookback=40, threshold=0.02, trade_fraction=0.15, hard_cap=80),
        ValueEstimateAgent("ValueEstimate2", lookback=60, threshold=0.03, trade_fraction=0.20, hard_cap=100),
        ValueEstimateAgent("ValueEstimate3", lookback=80, threshold=0.04, trade_fraction=0.25, hard_cap=120),
        RiskManagedAgent("RiskManaged1", lookback=15, threshold=0.02, trade_fraction=0.15, hard_cap=80),
        RiskManagedAgent("RiskManaged2", lookback=20, threshold=0.025, trade_fraction=0.20, hard_cap=100),
        RiskManagedAgent("RiskManaged3", lookback=30, threshold=0.03, trade_fraction=0.25, hard_cap=120),
    ]


    fundamental_process = OrnsteinUhlenbeckFundamental()#might add other later perchance but
    #for this simulation without gov or taxes, want to use OU because mean reverting

    market = Market(agents, fundamental_process)
    market.run()
    market.print_results()


if __name__ == "__main__":
    main()