#!/usr/bin/env python
import os, json, math, random, logging, asyncio, pickle
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from logging.handlers import RotatingFileHandler
from collections import deque, namedtuple

import nest_asyncio
nest_asyncio.apply()

# Alpaca SDK
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, APIError
from alpaca_trade_api.stream import Stream

# PyTorch for RL and pattern prediction
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F

# TA-Lib indicators
import ta
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# HuggingFace Transformers for sentiment analysis (force PyTorch)
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")

###############################################################################
# Cache System for Historical Bars
###############################################################################
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filepath(symbol: str, timeframe: str, days: int) -> str:
    filename = f"{symbol}_{timeframe}_{days}d.pkl"
    return os.path.join(CACHE_DIR, filename)

def load_cached_bars(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    filepath = get_cache_filepath(symbol, timeframe, days)
    if os.path.exists(filepath):
        try:
            df = pd.read_pickle(filepath)
            return df
        except Exception as e:
            logger.error(f"Failed to load cache for {symbol}: {e}", exc_info=True)
    return None

def save_cached_bars(symbol: str, timeframe: str, days: int, df: pd.DataFrame):
    filepath = get_cache_filepath(symbol, timeframe, days)
    try:
        df.to_pickle(filepath)
        logger.info(f"Cached bars saved for {symbol} at {filepath}")
    except Exception as e:
        logger.error(f"Failed to save cache for {symbol}: {e}", exc_info=True)

###############################################################################
# Helper Functions: Market Hours
###############################################################################
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo

def is_market_hours() -> bool:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

def time_until_market_open() -> float:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et < market_open:
        delta = market_open - now_et
    else:
        next_day = now_et + timedelta(days=1)
        market_open_next = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        delta = market_open_next - now_et
    return delta.total_seconds()

###############################################################################
# 1) Load Config and Setup Logging
###############################################################################
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found")

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
API_KEY = config.get("API_KEY")
API_SECRET = config.get("API_SECRET")
PAPER = config.get("PAPER", True)
BASE_URL = config.get("BASE_URL", "https://paper-api.alpaca.markets")
TRADE_OPTIONS = config.get("trade_options", False)

LOG_FILE = config.get("LOG_FILE", "trading_bot.log")
LOG_LEVEL_STR = config.get("LOG_LEVEL", "INFO").upper()

# RL and risk hyperparameters
MEMORY_SIZE = config.get("MEMORY_SIZE", 100000)
BATCH_SIZE = config.get("BATCH_SIZE", 64)
GAMMA = config.get("GAMMA", 0.99)
LEARNING_RATE = config.get("LEARNING_RATE", 0.0005)
TARGET_UPDATE = config.get("TARGET_UPDATE", 1000)
EPSILON_DECAY = config.get("EPSILON_DECAY", 10000)
ALPHA = config.get("ALPHA", 0.6)
BETA_START = config.get("BETA_START", 0.4)

RISK_PER_TRADE = config.get("RISK_PER_TRADE", 0.02)
ATR_PERIOD = config.get("ATR_PERIOD", 14)
VOL_TRAILING_MULTIPLIER = config.get("VOL_TRAILING_MULTIPLIER", 1.5)
MAX_DRAWDOWN = config.get("MAX_DRAWDOWN_LIMIT", 0.2)
TRAILING_STOP_LOSS = config.get("TRAILING_STOP_LOSS", 0.02)
TAKE_PROFIT = config.get("TAKE_PROFIT", 0.03)
MIN_HOLD_DAYS = config.get("MIN_HOLD_DAYS", 1)
MAX_HOLD_DAYS = config.get("MAX_HOLD_DAYS", 10)

# Transformer config for RL agent
TRANSFORMER_LAYERS = config.get("TRANSFORMER_LAYERS", 3)
ATTENTION_HEADS = config.get("ATTENTION_HEADS", 4)

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY or SECRET not found in config.json")

logger = logging.getLogger("MasterpieceTrader")
logger.setLevel(getattr(logging, LOG_LEVEL_STR, logging.INFO))
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

###############################################################################
# Set device and log it
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to use {device}")

###############################################################################
# 2) Alpaca Trader Class
###############################################################################
class AlpacaTrader:
    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        self.rest = REST(
            key_id=config["API_KEY"],
            secret_key=config["API_SECRET"],
            base_url=config.get("BASE_URL", "https://paper-api.alpaca.markets"),
            api_version="v2"
        )
        self.initialize_market_calendar()

    def initialize_market_calendar(self):
        try:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            calendars = self.rest.get_calendar(start=date_str, end=date_str)
            self.market_open = calendars[0].open if calendars else None
            self.market_close = calendars[0].close if calendars else None
            self.logger.info(f"Market Open: {self.market_open}, Market Close: {self.market_close}")
        except Exception as e:
            self.logger.error(f"Error fetching market calendar => {e}", exc_info=True)
            self.market_open = None
            self.market_close = None

    async def get_account(self):
        try:
            return self.rest.get_account()
        except Exception as e:
            self.logger.error(f"get_account => {e}", exc_info=True)
            return None

    async def get_position(self, symbol: str):
        try:
            pos = self.rest.get_position(symbol)
            if float(pos.qty) == 0:
                return None
            return pos
        except APIError as ex:
            if "position does not exist" in str(ex).lower():
                return None
            self.logger.error(f"get_position => {ex}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"get_position => {e}", exc_info=True)
            return None

    async def submit_order(self, **kwargs):
        try:
            kwargs.pop("asset_class", None)
            order = self.rest.submit_order(**kwargs)
            self.logger.info(f"Order Submitted => {order}")
            return order
        except Exception as e:
            self.logger.error(f"submit_order => {e}", exc_info=True)
            return None

    async def close_position(self, symbol: str):
        try:
            orders = self.rest.list_orders(status="open", symbols=[symbol])
            if orders:
                self.logger.info(f"Cancelling {len(orders)} open orders for {symbol}")
                for o in orders:
                    self.rest.cancel_order(o.id)
                await asyncio.sleep(1)
            self.rest.close_position(symbol)
            self.logger.info(f"Close position request sent for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"close_position => {symbol}: {e}", exc_info=True)
            return False

    async def get_latest_price(self, symbol: str):
        try:
            quote = self.rest.get_latest_quote(symbol)
            if quote.bid_price is not None and quote.ask_price is not None:
                return (quote.bid_price + quote.ask_price) / 2
            elif quote.last_price is not None:
                return quote.last_price
            return None
        except Exception as e:
            self.logger.error(f"get_latest_price => {symbol}: {e}", exc_info=True)
            return None

    # NEW: Method to fetch the options chain for an underlying asset.
    def get_options_chain(self, underlying: str):
        try:
            # Adjust this call based on the actual Alpaca options API.
            chain = self.rest.get_options_chain(underlying)
            return chain
        except Exception as e:
            self.logger.error(f"Error fetching options chain for {underlying}: {e}", exc_info=True)
            return []

###############################################################################
# 3) Dynamic Symbol Discovery and Initial History Fetching
###############################################################################
def get_all_tradable_symbols(trader: AlpacaTrader, logger: logging.Logger) -> List[str]:
    try:
        assets = trader.rest.list_assets(status='active', asset_class='us_equity')
        symbols = [asset.symbol for asset in assets if asset.tradable]
        logger.info(f"Found {len(symbols)} tradable symbols.")
        return symbols
    except Exception as e:
        logger.error(f"Error retrieving assets: {e}", exc_info=True)
        return []

async def fetch_initial_bars(symbol: str, trader: AlpacaTrader, logger: logging.Logger,
                               days: int = 5, timeframe: str = "15Min", use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached_df = load_cached_bars(symbol, timeframe, days)
        if cached_df is not None and not cached_df.empty:
            logger.info(f"Loaded cached bars for {symbol}")
            return cached_df

    try:
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=days)
        bars = trader.rest.get_bars(symbol, timeframe, start_utc.isoformat(), end_utc.isoformat(), adjustment="raw").df
        if bars.empty:
            logger.warning(f"{symbol} => No historical bars returned.")
            return pd.DataFrame()
        bars.reset_index(inplace=True)
        bars.rename(columns={"timestamp": "timestamp", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"}, inplace=True)
        save_cached_bars(symbol, timeframe, days, bars)
        return bars
    except Exception as e:
        logger.error(f"{symbol} => fetch_initial_bars: {e}", exc_info=True)
        return pd.DataFrame()

###############################################################################
# 4) Transformer Network Components (for RL Agent)
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.std_init * bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.std_init * bound)
    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)
        eps_in = eps_in.sign().mul_(eps_in.abs().sqrt_())
        eps_out = eps_out.sign().mul_(eps_out.abs().sqrt_())
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class TransformerDQN(nn.Module):
    def __init__(self, in_dim, out_dim, num_clusters, num_layers=3, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(in_dim, 128)
        self.pos_encoder = PositionalEncoding(128, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cnn1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.attention = nn.MultiheadAttention(embed_dim=192, num_heads=4, dropout=0.1)
        self.cluster_layers = nn.ModuleList([
            nn.Sequential(
                NoisyLinear(192, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                NoisyLinear(128, 64)
            ) for _ in range(num_clusters)
        ])
        self.value_stream = NoisyLinear(64, 1)
        self.advantage_stream = NoisyLinear(64, out_dim)
        self.num_clusters = num_clusters
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    def forward(self, x, cluster_ids, history):
        batch_size, seq_len, _ = history.shape
        embedded = self.embedding(history)  # (B, seq_len, 128)
        embedded = embedded.transpose(0, 1)   # (seq_len, B, 128)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer(embedded)  # (seq_len, B, 128)
        cnn_input = transformer_out.permute(1, 2, 0)    # (B, 128, seq_len)
        cnn1_out = F.relu(self.cnn1(cnn_input)).max(dim=-1)[0]
        cnn2_out = F.relu(self.cnn2(cnn_input)).max(dim=-1)[0]
        cnn3_out = F.relu(self.cnn3(cnn_input)).max(dim=-1)[0]
        combined = torch.cat([cnn1_out, cnn2_out, cnn3_out], dim=1)  # (B, 192)
        combined_unsq = combined.unsqueeze(0)  # (1, B, 192)
        attn_out, _ = self.attention(combined_unsq, combined_unsq, combined_unsq)
        attn_out = attn_out.squeeze(0)  # (B, 192)
        cluster_outputs = []
        for i in range(batch_size):
            cid = cluster_ids[i].item() % self.num_clusters
            cluster_outputs.append(self.cluster_layers[cid](attn_out[i].unsqueeze(0)))
        cluster_out = torch.cat(cluster_outputs, dim=0)  # (B, 64)
        values = self.value_stream(cluster_out)
        advantages = self.advantage_stream(cluster_out)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

###############################################################################
# 5) Prioritized Replay Memory and Persistence
###############################################################################
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'cluster_id', 'history', 'next_history'))

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def push(self, transition: tuple):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size: int, beta: float = 0.4):
        if not self.memory:
            return [], [], []
        prios = self.priorities[:self.pos] if len(self.memory) < self.capacity else self.priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights.tolist()
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemoryPersistence:
    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger
    def save_memory(self, memory: PrioritizedReplayMemory):
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump({'memory': memory.memory, 'priorities': memory.priorities, 'pos': memory.pos}, f)
            self.logger.info(f"Saved replay memory to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save replay memory: {e}", exc_info=True)
    def load_memory(self, memory: PrioritizedReplayMemory):
        if not os.path.exists(self.filepath):
            self.logger.warning(f"No replay memory file found at {self.filepath}. Starting fresh.")
            return
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                memory.memory = deque(data['memory'], maxlen=memory.capacity)
                memory.priorities = data['priorities']
                memory.pos = data['pos']
            self.logger.info(f"Loaded replay memory from {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load replay memory: {e}", exc_info=True)

###############################################################################
# 6) TransformerAgent Definition (for RL)
###############################################################################
class TransformerAgent:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_clusters: int,
        device: torch.device,
        memory_size: int,
        batch_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_decay: int,
        alpha: float,
        beta_start: float,
        logger: logging.Logger,
        num_layers: int = 3,
        nhead: int = 4,
        confidence_threshold: float = 0.1
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_clusters = num_clusters
        self.device = device
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.policy_net = TransformerDQN(in_dim, out_dim, num_clusters, num_layers, nhead).to(device)
        self.target_net = TransformerDQN(in_dim, out_dim, num_clusters, num_layers, nhead).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = PrioritizedReplayMemory(memory_size, alpha=alpha)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = 100000
    def select_action(self, state: torch.Tensor, history: torch.Tensor, cluster_id: int) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        self.policy_net.reset_noise()
        if random.random() < epsilon:
            return random.randrange(self.out_dim)
        else:
            with torch.no_grad():
                cids = torch.tensor([cluster_id], dtype=torch.long, device=self.device)
                q_vals = self.policy_net(state, cids, history)
                top2 = q_vals.topk(2, dim=1)
                margin = top2.values[0, 0] - top2.values[0, 1]
                if margin < self.confidence_threshold:
                    self.logger.info("Low confidence (margin {:.4f}). Forcing Hold action.".format(margin))
                    return 0
                return q_vals.argmax(dim=1).item()
    def store_transition(self, s, a, r, ns, d, cid, hist, n_hist):
        self.memory.push((s, a, r, ns, d, cid, hist, n_hist))
    def optimize_model(self, beta: float):
        if len(self.memory) < self.batch_size:
            return
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        if not transitions:
            return
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        cluster_ids = torch.tensor(batch.cluster_id, dtype=torch.long).to(self.device)
        history_batch = torch.cat(batch.history).to(self.device)
        next_history_batch = torch.cat(batch.next_history).to(self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.policy_net.reset_noise()
        current_q_values = self.policy_net(state_batch, cluster_ids, history_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch, cluster_ids, next_history_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch, cluster_ids, next_history_batch).gather(1, next_actions)
            target_q = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values
        td_errors = current_q_values - target_q
        loss = (td_errors.pow(2) * weights_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities.flatten())
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network.")
    def save_checkpoint(self, filepath: str):
        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps_done': self.steps_done
            }, filepath)
            self.logger.info(f"Saved checkpoint => {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath):
            self.logger.warning(f"No checkpoint file {filepath}. Starting fresh.")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint.get('steps_done', 0)
            self.logger.info(f"Loaded checkpoint => {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

###############################################################################
# 7) Enhanced Risk Manager Definitions
###############################################################################
class AdaptiveRiskManager:
    def __init__(self):
        self.win_rate = 0.5
        self.consecutive_losses = 0
        self.risk_multiplier = 1.0
    def update_risk_parameters(self, trade_success: bool):
        if trade_success:
            self.consecutive_losses = 0
            self.win_rate = 0.9 * self.win_rate + 0.1
        else:
            self.consecutive_losses += 1
            self.win_rate = 0.9 * self.win_rate
        if self.consecutive_losses > 2:
            self.risk_multiplier = max(0.5, 1.0 - 0.1 * self.consecutive_losses)
        else:
            self.risk_multiplier = min(2.0, 1.0 + (self.win_rate - 0.5))

class EnhancedRiskManager(AdaptiveRiskManager):
    def get_position_size(self, capital, entry_price, atr, correlation_matrix):
        if correlation_matrix is None or len(correlation_matrix) == 0:
            corr_penalty = 1.0
        else:
            max_corr = np.max(np.abs(correlation_matrix))
            corr_penalty = 1 - max_corr
        risk_multiplier = self.risk_multiplier * corr_penalty
        dynamic_atr_mult = VOL_TRAILING_MULTIPLIER if atr > entry_price * 0.01 else 1.8
        risk_amount = capital * RISK_PER_TRADE * risk_multiplier
        dollar_risk = atr * entry_price * dynamic_atr_mult
        if dollar_risk <= 0:
            return 0
        shares = int(risk_amount / dollar_risk)
        return max(shares, 0)

###############################################################################
# 8) Enhanced Indicators & Normalization
###############################################################################
def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True, drop=False)
    df.sort_index(inplace=True)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    atr_ind = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD)
    df['atr'] = atr_ind.average_true_range()
    adx = ADXIndicator(df['high'], df['low'], df['close'], 14)
    df['adx'] = adx.adx()
    kelt = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
    df['keltner_upper'] = kelt.keltner_channel_hband()
    df['keltner_lower'] = kelt.keltner_channel_lband()
    rolling_high = df['high'].rolling(50).max()
    rolling_low = df['low'].rolling(50).min()
    df['fib_23.6'] = rolling_high - (rolling_high - rolling_low) * 0.236
    df['fib_38.2'] = rolling_high - (rolling_high - rolling_low) * 0.382
    vwma_26 = (df['volume'] * df['close']).rolling(26).sum() / df['volume'].rolling(26).sum()
    vwma_9 = vwma_26.ewm(span=9, adjust=False).mean()
    df['vw_macd'] = vwma_26 - vwma_9
    df['momentum_3d'] = df['close'].pct_change(3)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)
    df = df.ffill().bfill().dropna()
    for col in ['rsi','macd','macd_signal','macd_diff',
                'bb_bbm','bb_bbh','bb_bbl','atr','adx',
                'keltner_upper','keltner_lower','fib_23.6','fib_38.2',
                'vw_macd','momentum_3d','hour_sin','hour_cos','day_sin','day_cos']:
        if col in df.columns:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower, upper)
            cmin = df[col].min()
            cmax = df[col].max() + 1e-6
            df[col] = (df[col] - cmin) / (cmax - cmin)
    return df

def detect_market_regime(df: pd.DataFrame) -> str:
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    trend = "Bullish" if sma_50.iloc[-1] > sma_200.iloc[-1] else "Bearish"
    adx_val = df['adx'].iloc[-1] if 'adx' in df.columns else 20
    volatility = df['atr'].iloc[-1] / df['close'].iloc[-1] if 'atr' in df.columns else 0
    if adx_val > 25 and volatility > 0.015:
        return f"{trend}_Trending"
    elif adx_val < 20 and volatility < 0.01:
        return f"{trend}_Range"
    return f"{trend}_Transitional"

###############################################################################
# 9) Bollinger-Based Risk–Reward Ratio Calculation
###############################################################################
def compute_bollinger_ratios(df: pd.DataFrame) -> Tuple[float, float]:
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    last_price = df.iloc[-1]['close']
    lower = bb.bollinger_lband().iloc[-1]
    upper = bb.bollinger_hband().iloc[-1]
    long_ratio = (upper - last_price) / (last_price - lower) if last_price > lower else 0
    short_ratio = (last_price - lower) / (upper - last_price) if upper > last_price else 0
    return long_ratio, short_ratio

###############################################################################
# 10) Trade State and Reward
###############################################################################
class TradeState:
    def __init__(self):
        self.active_trades = {}
    def update_trade(self, symbol: str, entry_price: float, side: str):
        now = datetime.now(timezone.utc)
        if symbol not in self.active_trades:
            if side == 'long':
                stop_loss = entry_price * (1 - TRAILING_STOP_LOSS)
                take_profit = entry_price * (1 + TAKE_PROFIT)
            else:
                stop_loss = entry_price * (1 + TRAILING_STOP_LOSS)
                take_profit = entry_price * (1 - TAKE_PROFIT)
            self.active_trades[symbol] = {
                'entry_price': entry_price,
                'side': side,
                'entry_time': now,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
    def adjust_stop_loss(self, symbol: str, current_price: float):
        if symbol not in self.active_trades:
            return
        trade = self.active_trades[symbol]
        side = trade['side']
        if side == 'long' and current_price > trade['entry_price']:
            new_sl = current_price * (1 - TRAILING_STOP_LOSS)
            if new_sl > trade['stop_loss']:
                trade['stop_loss'] = new_sl
        elif side == 'short' and current_price < trade['entry_price']:
            new_sl = current_price * (1 + TRAILING_STOP_LOSS)
            if new_sl < trade['stop_loss']:
                trade['stop_loss'] = new_sl
    def check_exit(self, symbol: str, current_price: float, current_time: datetime,
                   min_hold_days: int, max_hold_days: int) -> Optional[str]:
        if symbol not in self.active_trades:
            return None
        trade = self.active_trades[symbol]
        hold_duration = (current_time - trade['entry_time']).days
        side = trade['side']
        if side == 'long':
            if current_price <= trade['stop_loss']:
                del self.active_trades[symbol]
                return 'stop_loss'
            elif current_price >= trade['take_profit']:
                del self.active_trades[symbol]
                return 'take_profit'
        else:
            if current_price >= trade['stop_loss']:
                del self.active_trades[symbol]
                return 'stop_loss'
            elif current_price <= trade['take_profit']:
                del self.active_trades[symbol]
                return 'take_profit'
        if hold_duration >= max_hold_days:
            del self.active_trades[symbol]
            return 'time_exit'
        if hold_duration < min_hold_days:
            return None
        return None

def get_reward(symbol: str, current_price: float, trade_state: TradeState) -> float:
    if symbol in trade_state.active_trades:
        trade = trade_state.active_trades[symbol]
        entry_price = trade['entry_price']
        side = trade['side']
        return (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
    return 0.0

###############################################################################
# 11) RLTrader Wrapper
###############################################################################
class RLTrader:
    def __init__(self, device: torch.device, logger: logging.Logger,
                 checkpoint_path: str = "agent_checkpoint.pth", replay_memory_path: str = "replay_memory.pkl"):
        self.device = device
        self.logger = logger
        self.state_dim = 19
        self.action_dim = 3
        self.num_clusters = 10
        self.agent = TransformerAgent(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            num_clusters=self.num_clusters,
            device=device,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            epsilon_decay=EPSILON_DECAY,
            alpha=ALPHA,
            beta_start=BETA_START,
            logger=logger,
            num_layers=TRANSFORMER_LAYERS,
            nhead=ATTENTION_HEADS,
            confidence_threshold=0.1
        )
        self.checkpoint_path = checkpoint_path
        self.replay_memory_path = replay_memory_path
        self.memory_persistence = PrioritizedReplayMemoryPersistence(self.replay_memory_path, logger)
        self.load_agent()
        self.risk_manager = EnhancedRiskManager()
    def load_agent(self):
        self.agent.load_checkpoint(self.checkpoint_path)
        self.memory_persistence.load_memory(self.agent.memory)
    def save_agent(self):
        self.agent.save_checkpoint(self.checkpoint_path)
        self.memory_persistence.save_memory(self.agent.memory)
    def pick_action(self, state: torch.Tensor, history: torch.Tensor, cluster_id: int) -> int:
        return self.agent.select_action(state, history, cluster_id)
    def store_transition(self, *args):
        self.agent.store_transition(*args)
    def optimize(self, beta: float):
        self.agent.optimize_model(beta)
    def update_target(self):
        self.agent.update_target_net()
    def update_risk_mgr(self, trade_success: bool):
        self.risk_manager.update_risk_parameters(trade_success)
    def get_position_size(self, capital: float, entry_price: float, atr: float, correlation_matrix=None) -> int:
        return self.risk_manager.get_position_size(capital, entry_price, atr, correlation_matrix)

###############################################################################
# 12) Logging Helpers
###############################################################################
def record_trade(trade_details: Dict, logger: logging.Logger):
    try:
        logger.info(f"Trade Recorded: {trade_details}")
        with open("trade_log.json", "a") as f:
            json.dump(trade_details, f)
            f.write('\n')
    except Exception as e:
        logger.error(f"Error recording trade: {e}", exc_info=True)

async def cancel_open_orders_for_symbol(symbol: str, trader: AlpacaTrader, logger: logging.Logger):
    try:
        open_orders = trader.rest.list_orders(status="open", symbols=[symbol])
        if open_orders:
            logger.info(f"Cancelling {len(open_orders)} open orders for {symbol}")
            for order in open_orders:
                trader.rest.cancel_order(order.id)
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"cancel_open_orders_for_symbol => {symbol}: {e}", exc_info=True)

async def close_any_position(symbol: str, trader: AlpacaTrader, logger: logging.Logger):
    pos = await trader.get_position(symbol)
    if pos:
        success = await trader.close_position(symbol)
        if success:
            logger.info(f"Closed position for {symbol}")
            trade_details = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": f"Close_{pos.side}",
                "qty": abs(int(float(pos.qty))),
                "price": float(pos.avg_entry_price),
                "timestamp_closed": datetime.now(timezone.utc).isoformat()
            }
            record_trade(trade_details, logger)

###############################################################################
# 13) Build State and History
###############################################################################
def build_state(df: pd.DataFrame) -> torch.Tensor:
    if df.empty:
        return torch.zeros((1, 19), dtype=torch.float32).to(device)
    last = df.iloc[-1]
    feats = [last.get(col, 0.0) for col in [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'atr', 'adx',
        'keltner_upper', 'keltner_lower', 'fib_23.6', 'fib_38.2',
        'vw_macd', 'momentum_3d', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos'
    ]]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

def build_history(df: pd.DataFrame, window: int = 10) -> torch.Tensor:
    if df.empty:
        return torch.zeros((1, window, 19), dtype=torch.float32).to(device)
    if len(df) < window:
        df_window = pd.concat([df.iloc[-1:]] * window)
    else:
        df_window = df.iloc[-window:]
    seq = []
    for _, row in df_window.iterrows():
        seq.append([row.get(col, 0.0) for col in [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_bbm', 'bb_bbh', 'bb_bbl', 'atr', 'adx',
            'keltner_upper', 'keltner_lower', 'fib_23.6', 'fib_38.2',
            'vw_macd', 'momentum_3d', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos'
        ]])
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

###############################################################################
# 14) PatternPredictor: LSTM for Intraday Pattern Forecasting
###############################################################################
class PatternPredictor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=1, output_dim=1):
        super(PatternPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

pattern_predictor = PatternPredictor().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
pattern_optimizer = optim.Adam(pattern_predictor.parameters(), lr=0.001)
pattern_loss_fn = nn.MSELoss()
pattern_dataset = []

async def train_pattern_predictor():
    global pattern_dataset
    if not pattern_dataset:
        return
    inputs = torch.cat([pair[0] for pair in pattern_dataset], dim=0).to(device)
    targets = torch.cat([pair[1] for pair in pattern_dataset], dim=0).to(device)
    pattern_optimizer.zero_grad()
    outputs = pattern_predictor(inputs)
    loss = pattern_loss_fn(outputs, targets)
    loss.backward()
    pattern_optimizer.step()
    logger.info(f"PatternPredictor training loss: {loss.item():.6f}")
    pattern_dataset = []

###############################################################################
# 15) ProprietaryMarketPredictor: Our Cutting-Edge Neural Model
###############################################################################
class ProprietaryMarketPredictor(nn.Module):
    """
    Our proprietary multi–modal neural model that fuses technical data and Alpaca news sentiment
    to predict the next bar's percentage return.
    """
    def __init__(self, tech_input_dim, sentiment_input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(ProprietaryMarketPredictor, self).__init__()
        self.tech_lstm = nn.LSTM(tech_input_dim, hidden_dim, num_layers, batch_first=True)
        self.sentiment_fc = nn.Sequential(
            nn.Linear(sentiment_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, tech_input, sentiment_input):
        tech_out, _ = self.tech_lstm(tech_input)
        tech_feature = tech_out[:, -1, :]
        sentiment_feature = self.sentiment_fc(sentiment_input)
        combined = torch.cat([tech_feature, sentiment_feature], dim=1)
        output = self.combined_fc(combined)
        return output

proprietary_predictor = ProprietaryMarketPredictor(tech_input_dim=19, sentiment_input_dim=1).to(device)
proprietary_optimizer = optim.Adam(proprietary_predictor.parameters(), lr=0.001)
proprietary_loss_fn = nn.MSELoss()
proprietary_dataset = []

async def train_proprietary_predictor():
    global proprietary_dataset
    if not proprietary_dataset:
        return
    inputs = torch.cat([pair[0] for pair in proprietary_dataset], dim=0).to(device)
    targets = torch.cat([pair[1] for pair in proprietary_dataset], dim=0).to(device)
    proprietary_optimizer.zero_grad()
    sentiment_input = torch.zeros(inputs.size(0), 1).to(device)
    outputs = proprietary_predictor(inputs, sentiment_input)
    loss = proprietary_loss_fn(outputs, targets)
    loss.backward()
    proprietary_optimizer.step()
    logger.info(f"ProprietaryPredictor training loss: {loss.item():.6f}")
    proprietary_dataset = []

###############################################################################
# 15) Options Module: Gamma Scalping with Real Option Orders
###############################################################################
def next_friday_expiration() -> datetime:
    today = datetime.now(timezone.utc)
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_friday = today + timedelta(days=days_ahead)
    expiration = next_friday.replace(hour=16, minute=0, second=0, microsecond=0)
    return expiration

def build_option_symbol(underlying: str, underlying_price: float, is_call: bool) -> str:
    expiration = next_friday_expiration()
    strike = int(round(underlying_price, 2) * 1000)
    option_type = 'C' if is_call else 'P'
    option_symbol = f"{underlying}{expiration.strftime('%y%m%d')}{option_type}{strike:08d}"
    return option_symbol

# UPDATED: In process_options, we fetch the options chain and compare the generated symbol.
async def process_options(underlying: str, underlying_price: float, trader: AlpacaTrader, logger: logging.Logger, side: str):
    is_call = True if side == "long" else False
    generated_symbol = build_option_symbol(underlying, underlying_price, is_call)
    # Fetch options chain using the new method in AlpacaTrader.
    options_chain = trader.get_options_chain(underlying)
    option_symbols = [opt['symbol'] for opt in options_chain] if options_chain else []
    if generated_symbol not in option_symbols:
        logger.error(f"Generated option symbol {generated_symbol} not found in options chain for {underlying}. Available symbols: {option_symbols}")
        return
    logger.info(f"Placing options trade for {underlying}: {generated_symbol}")
    order = await trader.submit_order(
        symbol=generated_symbol,
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc"
    )
    if order:
        logger.info(f"Executed options order for {generated_symbol}")

###############################################################################
# 16) Global Live Bars Storage and Websocket Callback
###############################################################################
live_bars: Dict[str, List[Dict]] = {}
previous_predictor_input: Dict[str, torch.Tensor] = {}

def get_news_sentiment(symbol: str) -> float:
    try:
        news_items = trader_instance.rest.get_news(symbol=symbol, limit=5)
        if not news_items:
            return 0.0
        scores = []
        for item in news_items:
            headline = item.headline
            results = sentiment_analyzer(headline)
            score = 1 if results[0]['label'].lower() == "positive" else -1
            scores.append(score)
        return sum(scores) / len(scores)
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}", exc_info=True)
        return 0.0

async def on_bar_callback(bar):
    if not is_market_hours():
        logger.info("Market is closed. Skipping bar processing.")
        return
    timestamp = getattr(bar, "timestamp", bar._raw.get("timestamp"))
    open_price = getattr(bar, "open", bar._raw.get("open"))
    high_price = getattr(bar, "high", bar._raw.get("high"))
    low_price = getattr(bar, "low", bar._raw.get("low"))
    close_price = getattr(bar, "close", bar._raw.get("close"))
    volume = getattr(bar, "volume", bar._raw.get("volume"))
    symbol = bar.symbol
    bar_dict = {
        "timestamp": timestamp,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume
    }
    if symbol not in live_bars:
        live_bars[symbol] = []
    live_bars[symbol].append(bar_dict)
    if len(live_bars[symbol]) > 50:
        live_bars[symbol] = live_bars[symbol][-50:]
    if len(live_bars[symbol]) >= 10:
        df = pd.DataFrame(live_bars[symbol])
        try:
            df = compute_enhanced_indicators(df)
            predictor_features = df[['rsi', 'macd', 'atr', 'adx', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].tail(10)
            if predictor_features.empty or predictor_features.shape[0] == 0:
                logger.warning(f"{symbol} => Predictor features are empty, skipping pattern prediction.")
                return
            predictor_input = torch.tensor(predictor_features.values, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pattern_prediction = pattern_predictor(predictor_input).item()
            news_sentiment = get_news_sentiment(symbol)
            tech_input = build_history(df, window=10).to(device)
            sentiment_input = torch.tensor([[news_sentiment]], dtype=torch.float32).to(device)
            with torch.no_grad():
                proprietary_prediction = proprietary_predictor(tech_input, sentiment_input).item()
            combined_prediction = (pattern_prediction + proprietary_prediction) / 2
            if symbol in previous_predictor_input:
                prev_df = pd.DataFrame(live_bars[symbol])
                if len(prev_df) >= 2:
                    actual_return = (prev_df.iloc[-1]['close'] - prev_df.iloc[-2]['close']) / prev_df.iloc[-2]['close']
                    pattern_dataset.append((previous_predictor_input[symbol], torch.tensor([[actual_return]], dtype=torch.float32).to(device)))
                    proprietary_dataset.append((tech_input, torch.tensor([[actual_return]], dtype=torch.float32).to(device)))
            previous_predictor_input[symbol] = predictor_input
            asyncio.create_task(process_symbol_live(symbol, df, combined_prediction))
        except Exception as e:
            logger.error(f"Error processing live bars for {symbol}: {e}", exc_info=True)
    if TRADE_OPTIONS:
        if high_price - low_price > 0.02 * close_price:
            side_choice = "long" if random.random() < 0.5 else "short"
            asyncio.create_task(process_options(symbol, close_price, trader_instance, logger, side_choice))

async def process_symbol_live(symbol: str, df: pd.DataFrame, combined_prediction: float):
    if symbol in global_symbols:
        symbol_id = global_symbols.index(symbol)
    else:
        symbol_id = 0
    await process_symbol(symbol, rl_instance, trader_instance, logger, trade_state_instance, symbol_id, {}, global_symbols, combined_prediction)

###############################################################################
# 17) Main Processing: Ensemble Decision Engine for Stocks
###############################################################################
async def process_symbol(symbol: str, rl: RLTrader, trader: AlpacaTrader, logger: logging.Logger,
                         trade_state: TradeState, symbol_id: int, data_map: Dict[str, pd.DataFrame],
                         all_symbols: List[str], combined_prediction: float):
    df = compute_enhanced_indicators(pd.DataFrame(live_bars[symbol]))
    if df.empty:
        logger.info(f"{symbol} => Insufficient data after indicator computation.")
        return
    long_ratio, short_ratio = compute_bollinger_ratios(pd.DataFrame(live_bars[symbol]))
    state = build_state(df).to(rl.device)
    hist_tensor = build_history(df, window=10).to(rl.device)
    regime = detect_market_regime(df)
    ma_signal = 1 if "Bullish" in regime and "Trending" in regime else 0
    rsi_value = state[0, 0].item()
    rl_action = rl.pick_action(state, hist_tensor, symbol_id)
    if combined_prediction > 0.005:
        combined_action = 1
    elif combined_prediction < -0.005:
        combined_action = 2
    else:
        combined_action = rl_action
    if combined_action == 1:
        if (long_ratio < 5) or (rsi_value > 0.8):
            logger.info(f"{symbol} => Filtering Long: MA={ma_signal}, RSI={rsi_value:.2f}, Long Ratio={long_ratio:.2f}. Forcing Hold.")
            action = 0
        else:
            action = 1
    elif combined_action == 2:
        if (short_ratio < 5) or (rsi_value < 0.2):
            logger.info(f"{symbol} => Filtering Short: MA={ma_signal}, RSI={rsi_value:.2f}, Short Ratio={short_ratio:.2f}. Forcing Hold.")
            action = 0
        else:
            action = 2
    else:
        action = combined_action
    logger.info(f"{symbol} => Final stock action: {action} (Combined prediction: {combined_prediction:.4f})")
    current_atr = float(df.iloc[-1]['atr']) if 'atr' in df.columns else 1.0
    acct = await trader.get_account()
    if not acct:
        return
    equity = float(acct.equity)
    corr_mat = np.random.uniform(-0.3, 0.3, (len(all_symbols), len(all_symbols)))
    position_size = rl.get_position_size(equity, float(df.iloc[-1]['close']), current_atr, corr_mat)
    await apply_action(symbol, action, trader, logger, trade_state, rl, current_atr, position_size)
    pos = await trader.get_position(symbol)
    if pos:
        entry_price = float(pos.avg_entry_price)
        if symbol not in trade_state.active_trades:
            side = 'long' if action == 1 else 'short'
            trade_state.update_trade(symbol, entry_price, side)
        current_price = await trader.get_latest_price(symbol)
        if current_price:
            trade_state.adjust_stop_loss(symbol, current_price)
    current_price = await trader.get_latest_price(symbol)
    if current_price:
        exit_reason = trade_state.check_exit(symbol, current_price, datetime.now(timezone.utc), MIN_HOLD_DAYS, MAX_HOLD_DAYS)
        if exit_reason:
            logger.info(f"{symbol} => Exiting trade due to {exit_reason}")
            await close_any_position(symbol, trader, logger)
            trade_state.active_trades.pop(symbol, None)
    reward = get_reward(symbol, current_price, trade_state)
    next_state = build_state(df).to(rl.device)
    next_history = build_history(df, window=10).to(rl.device)
    done = 0
    rl.store_transition(state, action, reward, next_state, done, symbol_id, hist_tensor, next_history)
    rl.update_risk_mgr(trade_success=(reward > 0))

# NEW: Modified apply_action now saves the RL agent's state after a trade is executed.
async def apply_action(symbol: str, action: int, trader: AlpacaTrader, logger: logging.Logger,
                       trade_state: TradeState, rl: RLTrader, atr_value: float, shares: int):
    acct = await trader.get_account()
    if not acct:
        return
    last_price = await trader.get_latest_price(symbol)
    if not last_price or last_price <= 0:
        logger.info(f"{symbol} => Invalid price. Skipping action.")
        return
    await cancel_open_orders_for_symbol(symbol, trader, logger)
    pos = await trader.get_position(symbol)
    current_qty = float(pos.qty) if pos else 0
    if action == 0:
        logger.info(f"{symbol} => Hold action. No trade executed.")
        return
    if shares <= 0:
        logger.info(f"{symbol} => Computed shares=0. Skipping trade.")
        return
    if action == 1:
        if current_qty > 0:
            logger.info(f"{symbol} => Already long. Skipping entry.")
        else:
            if current_qty < 0:
                logger.info(f"{symbol} => Closing short before long entry.")
                await trader.submit_order(symbol=symbol, qty=abs(int(current_qty)), side="buy", type="market", time_in_force="gtc")
                await asyncio.sleep(1)
            order = await trader.submit_order(symbol=symbol, qty=shares, side="buy", type="market", time_in_force="gtc")
            if order:
                logger.info(f"{symbol} => Opened long: {shares} shares.")
                trade_details = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "action": "Buy", "qty": shares, "price": last_price}
                record_trade(trade_details, logger)
                trade_state.update_trade(symbol, last_price, 'long')
                rl.save_agent()  # Save state after trade
    elif action == 2:
        if current_qty < 0:
            logger.info(f"{symbol} => Already short. Skipping entry.")
        else:
            if current_qty > 0:
                logger.info(f"{symbol} => Closing long before short entry.")
                await trader.submit_order(symbol=symbol, qty=abs(int(current_qty)), side="sell", type="market", time_in_force="gtc")
                await asyncio.sleep(1)
            order = await trader.submit_order(symbol=symbol, qty=shares, side="sell", type="market", time_in_force="gtc")
            if order:
                logger.info(f"{symbol} => Opened short: {shares} shares.")
                trade_details = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "action": "Short", "qty": shares, "price": last_price}
                record_trade(trade_details, logger)
                trade_state.update_trade(symbol, last_price, 'short')
                rl.save_agent()  # Save state after trade

###############################################################################
# 17) RL and Proprietary Predictor Optimization Loops
###############################################################################
async def rl_optimization_loop():
    episode = 0
    while True:
        try:
            beta = min(1.0, rl_instance.agent.beta_start + episode * (1.0 - rl_instance.agent.beta_start) / rl_instance.agent.beta_frames)
            rl_instance.optimize(beta)
            if episode % TARGET_UPDATE == 0:
                rl_instance.update_target()
            if episode % 100 == 0:
                rl_instance.save_agent()
            episode += 1
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in RL optimization loop: {e}", exc_info=True)
            await asyncio.sleep(60)

async def proprietary_training_loop():
    while True:
        try:
            await train_proprietary_predictor()
            await asyncio.sleep(120)
        except Exception as e:
            logger.error(f"Error in ProprietaryPredictor training loop: {e}", exc_info=True)
            await asyncio.sleep(120)

async def pattern_training_loop():
    while True:
        try:
            await train_pattern_predictor()
            await asyncio.sleep(120)
        except Exception as e:
            logger.error(f"Error in PatternPredictor training loop: {e}", exc_info=True)
            await asyncio.sleep(120)

###############################################################################
# 18) Main Function: Initialize History, Filter Symbols, Start Websocket Stream, and Loops
###############################################################################
global_symbols: List[str] = []

trader_instance: Optional[AlpacaTrader] = None
rl_instance: Optional[RLTrader] = None
trade_state_instance: Optional[TradeState] = None

async def main():
    global trader_instance, rl_instance, trade_state_instance, global_symbols
    trader_instance = AlpacaTrader(config, logger)
    
    if not is_market_hours():
        seconds_to_wait = time_until_market_open()
        logger.info(f"Market is closed. Waiting for {seconds_to_wait:.0f} seconds until market open.")
        await asyncio.sleep(seconds_to_wait)
    
    all_symbols = get_all_tradable_symbols(trader_instance, logger)
    
    candidate_symbols = []
    performance_metrics = {}
    for sym in all_symbols:
        df_hist = await fetch_initial_bars(sym, trader_instance, logger, days=90, timeframe="15Min")
        if not df_hist.empty and len(df_hist) >= 10:
            live_bars[sym] = df_hist.to_dict(orient="records")
            candidate_symbols.append(sym)
            try:
                first_open = df_hist.iloc[0]['open']
                last_close = df_hist.iloc[-1]['close']
                performance = (last_close - first_open) / first_open
                performance_metrics[sym] = performance
            except Exception as e:
                logger.error(f"Error computing performance for {sym}: {e}", exc_info=True)
        else:
            logger.warning(f"{sym} => Skipped due to insufficient historical bars.")
    
    logger.info(f"Candidate symbols count: {len(candidate_symbols)}")
    top_150 = sorted(candidate_symbols, key=lambda s: performance_metrics.get(s, -float('inf')), reverse=True)[:150]
    global_symbols = top_150
    logger.info(f"Selected top {len(global_symbols)} symbols for trading: {global_symbols}")
    
    rl_instance = RLTrader(device, logger)
    trade_state_instance = TradeState()
    stream = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed="iex")
    stream.subscribe_bars(on_bar_callback, *global_symbols)
    asyncio.create_task(rl_optimization_loop())
    asyncio.create_task(proprietary_training_loop())
    asyncio.create_task(pattern_training_loop())
    stream.run()

###############################################################################
# Program Entry Point
###############################################################################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Saving agents and exiting.")
        if rl_instance:
            rl_instance.save_agent()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        if rl_instance:
            rl_instance.save_agent()
