"""
Report generation for backtest results.

Generates:
    - Equity curve plots
    - Drawdown visualization
    - Monthly returns heatmap
    - HTML report export

Token Optimization:
    - Uses matplotlib for portability
    - HTML uses Jinja2 templates
"""
import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from .engine import BacktestResult
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


class BacktestReporter:
    """
    Generate reports from backtest results.
    
    Creates visualizations and exports HTML reports.
    
    Example:
        >>> reporter = BacktestReporter(result)
        >>> reporter.save_html("report.html")
        >>> reporter.plot_equity_curve()
    """
    
    def __init__(
        self,
        result: BacktestResult,
        title: str = "Backtest Report",
    ):
        """
        Initialize reporter.
        
        Args:
            result: BacktestResult to report on
            title: Report title
        """
        self.result = result
        self.title = title
        self.figures: Dict[str, Any] = {}
    
    def plot_equity_curve(
        self,
        figsize: tuple = (12, 6),
        show_drawdown: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Any]:
        """
        Plot equity curve with optional drawdown.
        
        Args:
            figsize: Figure size
            show_drawdown: Show drawdown subplot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure or None if unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available")
            return None
        
        equity = self.result.equity_curve
        
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # Equity curve
        ax1.plot(equity, color="#2ecc71", linewidth=1.5, label="Portfolio Value")
        ax1.fill_between(range(len(equity)), equity, alpha=0.3, color="#2ecc71")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.set_title(f"{self.title} - Equity Curve", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(equity) - 1)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        
        # Drawdown
        if show_drawdown:
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max * 100
            
            ax2.fill_between(range(len(drawdown)), -drawdown, color="#e74c3c", alpha=0.7)
            ax2.set_ylabel("Drawdown (%)", fontsize=11)
            ax2.set_xlabel("Trading Period", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(min(-drawdown) * 1.1, 0)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved equity curve to {save_path}")
        
        self.figures["equity_curve"] = fig
        return fig
    
    def plot_returns_distribution(
        self,
        figsize: tuple = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Any]:
        """
        Plot returns distribution histogram.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        equity = self.result.equity_curve
        returns = np.diff(equity) / equity[:-1] * 100  # Percentage returns
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram
        n, bins, patches = ax.hist(
            returns, bins=50, density=True, alpha=0.7,
            color="#3498db", edgecolor="white"
        )
        
        # Color negative returns red
        for patch, left, right in zip(patches, bins[:-1], bins[1:]):
            if right <= 0:
                patch.set_facecolor("#e74c3c")
        
        # Statistics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        ax.axvline(mean_ret, color="black", linestyle="--", linewidth=2, label=f"Mean: {mean_ret:.2f}%")
        ax.axvline(0, color="gray", linestyle="-", linewidth=1)
        
        ax.set_xlabel("Daily Return (%)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{self.title} - Returns Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        self.figures["returns_distribution"] = fig
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        figsize: tuple = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Any]:
        """
        Plot monthly returns heatmap.
        
        Only works if result has date information.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            logger.warning("matplotlib or seaborn not available for heatmap")
            return None
        
        # Calculate monthly returns from equity
        equity = self.result.equity_curve
        n_periods = len(equity)
        
        # Assume roughly 252 trading days per year, 21 per month
        periods_per_month = 21
        n_months = n_periods // periods_per_month
        
        if n_months < 2:
            logger.info("Not enough data for monthly heatmap")
            return None
        
        # Calculate approximate monthly returns
        monthly_returns = []
        for i in range(n_months):
            start_idx = i * periods_per_month
            end_idx = min((i + 1) * periods_per_month, n_periods - 1)
            if start_idx < len(equity) and end_idx < len(equity) and equity[start_idx] > 0:
                ret = (equity[end_idx] - equity[start_idx]) / equity[start_idx] * 100
                monthly_returns.append(ret)
        
        if len(monthly_returns) < 2:
            return None
        
        # Create approximate year/month structure
        n_years = (len(monthly_returns) + 11) // 12
        data = np.full((n_years, 12), np.nan)
        
        for i, ret in enumerate(monthly_returns):
            year_idx = i // 12
            month_idx = i % 12
            data[year_idx, month_idx] = ret
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        
        sns.heatmap(
            data,
            annot=True,
            fmt=".1f",
            center=0,
            cmap=cmap,
            cbar_kws={"label": "Return (%)"},
            ax=ax,
            xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            yticklabels=[f"Year {i+1}" for i in range(n_years)],
        )
        
        ax.set_title(f"{self.title} - Monthly Returns", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        self.figures["monthly_heatmap"] = fig
        return fig
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
    def generate_html(self) -> str:
        """
        Generate HTML report.
        
        Returns:
            HTML string
        """
        # Generate plots if not already done
        if "equity_curve" not in self.figures:
            self.plot_equity_curve()
        if "returns_distribution" not in self.figures:
            self.plot_returns_distribution()
        
        # Get metrics
        m = self.result.metrics
        
        # Convert figures to base64
        images = {}
        for name, fig in self.figures.items():
            if fig is not None:
                images[name] = self._fig_to_base64(fig)
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #0f3460;
            --text: #e8e8e8;
            --positive: #2ecc71;
            --negative: #e74c3c;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 1.5rem;
            color: #fff;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 0.25rem;
        }}
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .chart-container {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}
        .chart-container img {{
            width: 100%;
            border-radius: 8px;
        }}
        .section-title {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #fff;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        .trades-table th, .trades-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--accent);
        }}
        .trades-table th {{ background: var(--accent); }}
        footer {{
            margin-top: 2rem;
            text-align: center;
            color: #666;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {self.title}</h1>
        <p style="margin-bottom: 2rem; color: #aaa;">
            Period: {self.result.start_date} to {self.result.end_date} | 
            Symbols: {', '.join(self.result.symbols) if self.result.symbols else 'N/A'}
        </p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if m.total_return >= 0 else 'negative'}">
                    {m.total_return:.2%}
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.sortino_ratio:.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">-{m.max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.volatility:.1%}</div>
                <div class="metric-label">Volatility</div>
            </div>
        </div>
        
        {'<div class="chart-container"><h3 class="section-title">Equity Curve</h3><img src="data:image/png;base64,' + images.get("equity_curve", "") + '" alt="Equity Curve"></div>' if "equity_curve" in images else ''}
        
        {'<div class="chart-container"><h3 class="section-title">Returns Distribution</h3><img src="data:image/png;base64,' + images.get("returns_distribution", "") + '" alt="Returns"></div>' if "returns_distribution" in images else ''}
        
        <footer>
            Generated by PipFlow AI Backtesting Engine | {datetime.now():%Y-%m-%d %H:%M:%S}
        </footer>
    </div>
</body>
</html>
"""
        return html
    
    def save_html(self, path: Union[str, Path]) -> None:
        """
        Save HTML report to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        html = self.generate_html()
        path.write_text(html)
        
        logger.info(f"Saved HTML report to {path}")
    
    def close(self) -> None:
        """Close all matplotlib figures."""
        if MATPLOTLIB_AVAILABLE:
            for fig in self.figures.values():
                if fig is not None:
                    plt.close(fig)
