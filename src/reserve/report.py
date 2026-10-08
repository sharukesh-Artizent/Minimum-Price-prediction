"""Report generation (completed in Task 16)."""


def explain(row) -> str:
    return (f"Market ${row.m:,.0f}; ~{row.tau:.0f}d to sell, holding value ${row.W:,.0f}; "
            f"reserve ${row.reserve_p2:,.0f} ({row.reserve_p2 / row.m:.0%} of market, "
            f"{row.reserve_p2 / row.reserve_b0 - 1:+.0%} vs current floor); P(sell) ~{row.p2_p_sell:.0%}")
