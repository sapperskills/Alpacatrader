# Transformer-Based Options Trading Bot

## Introduction

Welcome to the Transformer-Based Options Trading Bot repository! This project is an experimental yet production-ready options trading bot that leverages Alpaca’s API and a transformer-based deep Q-network (DQN) for dynamic trade decision making. I started this project with minimal programming experience and have been using prompt engineering to teach myself Python along the way. While I’m sure there are many errors, it’s the pursuit of knowledge that matters most—so please don’t judge too harshly and feel free to share your feedback!

## Project Overview

The bot dynamically screens options contracts from Alpaca, selects optimal trades based on liquidity and key options Greeks, and uses a deep reinforcement learning agent with transformer components to decide whether to execute a call or put option trade. It also features advanced risk management and real-time data processing to adapt to market conditions as they evolve.

## Features

- **Dynamic Options Screening:**  
  Retrieves options chain data via Alpaca’s API and selects top underlyings based on liquidity and Greeks.

- **Deep Reinforcement Learning with Transformers:**  
  Implements a transformer-based deep Q-network (DQN) that uses positional encodings, CNN layers, and multi-head attention to process technical indicators and historical market data for trade decision making.

- **Advanced Risk Management:**  
  Dynamically adjusts stop-loss and take-profit levels, calculates optimal position sizes based on indicators like ATR, and factors in correlations for comprehensive risk management.

- **Real-Time Data Processing:**  
  Utilizes websocket subscriptions to stream live market data for both underlying equities and options contracts, enabling timely decision making.

- **Historical Data Caching:**  
  Efficiently caches historical market data to reduce redundant API calls and speed up processing.

## Requirements

- **Python:** 3.8 or later
- **Libraries:** 
  - `alpaca_trade_api`
  - `torch` (PyTorch)
  - `pandas`
  - `numpy`
  - `ta` (TA-Lib indicators)
  - `transformers` (HuggingFace)
  - `nest_asyncio`
  - Other standard libraries as listed in `requirements.txt`

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transformer-options-trading-bot.git
   cd transformer-options-trading-bot

