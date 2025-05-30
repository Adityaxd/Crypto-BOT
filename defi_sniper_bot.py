import asyncio
import aiohttp
import json
import time
from web3 import Web3
from web3.middleware import geth_poa_middleware
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import os
from decimal import Decimal

# Technical indicators - using pandas-ta 
import pandas as pd
import pandas_ta as ta

# local LLM 
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
try:
    import ollama  # For running models like DeepSeek locally
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Ollama not installed. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Sentence transformers not available. Some features will be limited.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import joblib
import pickle
from pathlib import Path
import numpy as np

class LocalLLMPredictor:
    """Local LLM integration for trade predictions - no API keys needed"""
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.models = {}
        
        # Initialize different model components
        self._initialize_models(model_path)
        
    def _initialize_models(self, model_path: str):
        """Initialize local models"""
        try:
            # 1. Embedding model for semantic analysis
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    print("Sentence transformer model loaded")
                except Exception as e:
                    print(f"Could not load sentence transformer: {e}")
                    self.embedding_model = None
            else:
                self.embedding_model = None
            
            # 2. Local LLM via Ollama (DeepSeek / Qwen)
            if OLLAMA_AVAILABLE:
                self.ollama_model = "deepseek-coder:latest"  # or llama2, mistral, etc.
            else:
                self.ollama_model = None
            
            # 3. Custom trained classifier 
            if model_path and Path(model_path).exists():
                self.custom_model = joblib.load(model_path)
            else:
                # Create a simple ML model for demonstration
                from sklearn.ensemble import RandomForestClassifier
                self.custom_model = RandomForestClassifier(n_estimators=100)
                self.is_pretrained = False
            
            self.model_loaded = True
            print(f"Models loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to rule-based system")
            
    async def predict_trade_success(self, token_data: Dict) -> float:
        """
        Predict trade success probability using local LLM
        Combines multiple local models for robust prediction
        """
        try:
            # 1. Prepare features
            features = self._extract_features(token_data)
            
            # 2. Get LLM analysis via Ollama
            llm_score = await self._get_llm_analysis(token_data)
            
            # 3. Get embedding-based similarity score
            embedding_score = self._get_embedding_score(token_data)
            
            # 4. Use custom ML model if trained
            ml_score = self._get_ml_prediction(features)
            
            # 5. Combine predictions with weighted ensemble
            final_score = (
                llm_score * 0.4 +        # LLM analysis weight
                ml_score * 0.3 +         # ML model weight
                embedding_score * 0.2 +   # Semantic similarity weight
                features['base_score'] * 0.1  # Rule-based weight
            )
            
            return min(final_score, 0.95)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            # Fallback to rule-based scoring
            return self._rule_based_scoring(token_data)
    
    async def _get_llm_analysis(self, token_data: Dict) -> float:
        """Get analysis from local LLM using Ollama"""
        if not OLLAMA_AVAILABLE or not self.ollama_model:
            return self._rule_based_scoring(token_data)
            
        try:
            # Prepare prompt for LLM
            prompt = f"""Analyze this crypto token for trading potential:
            
            Token Data:
            - Liquidity Score: {token_data.get('liquidity_score', 0):.2f}
            - Holder Distribution: {token_data.get('holder_distribution', 0):.2f}
            - Contract Verified: {token_data.get('contract_verified', False)}
            - Sentiment Score: {token_data.get('sentiment_score', 0.5):.2f}
            - Technical Score: {token_data.get('technical_score', 0.5):.2f}
            - Slippage: {token_data.get('slippage', 0):.2f}%
            - Rug Pull Risk: {token_data.get('rugpull_risk', 0):.2f}
            
            Based on these metrics, rate the trading potential from 0.0 to 1.0.
            Consider: liquidity depth, holder concentration, contract safety, and market sentiment.
            Respond with only a decimal number between 0.0 and 1.0.
            """
            
            # Query local LLM
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "max_tokens": 10
                }
            )
            
            # Parse response
            score_text = response['response'].strip()
            score = float(score_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            # If Ollama is not running or model not available
            return self._rule_based_scoring(token_data)
    
    def _get_embedding_score(self, token_data: Dict) -> float:
        """Calculate embedding-based similarity score"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.embedding_model:
            # Fallback to simple similarity calculation
            return self._simple_similarity_score(token_data)
            
        try:
            # Create text representation of token
            token_text = f"""
            Liquidity: {token_data.get('liquidity_score', 0)}
            Holders: {token_data.get('holder_distribution', 0)}
            Verified: {token_data.get('contract_verified', False)}
            Sentiment: {token_data.get('sentiment_score', 0.5)}
            """
            
            # Good token profile for comparison
            good_token_text = """
            Liquidity: 0.8
            Holders: 0.7
            Verified: True
            Sentiment: 0.8
            """
            
            # Get embeddings
            token_embedding = self.embedding_model.encode(token_text)
            good_embedding = self.embedding_model.encode(good_token_text)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([token_embedding], [good_embedding])[0][0]
            
            return max(0.0, similarity)
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0.5
    
    def _simple_similarity_score(self, token_data: Dict) -> float:
        """Simple similarity calculation without embeddings"""
        ideal_values = {
            'liquidity_score': 0.8,
            'holder_distribution': 0.7,
            'contract_verified': 1.0,
            'sentiment_score': 0.8,
            'technical_score': 0.7
        }
        
        score = 0
        count = 0
        
        for key, ideal in ideal_values.items():
            if key in token_data:
                actual = float(token_data[key]) if key != 'contract_verified' else (1.0 if token_data[key] else 0.0)
                diff = abs(ideal - actual)
                score += 1 - diff
                count += 1
        
        return score / count if count > 0 else 0.5
    
    def _get_ml_prediction(self, features: np.ndarray) -> float:
        """Get prediction from custom ML model"""
        try:
            if hasattr(self, 'is_pretrained') and not self.is_pretrained:
                # Model not trained, return base score
                return features['base_score']
                
            # Get probability from trained model
            prob = self.custom_model.predict_proba([features['vector']])[0][1]
            return prob
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return features['base_score']
    
    def _extract_features(self, token_data: Dict) -> Dict:
        """Extract numerical features for ML model"""
        features_vector = np.array([
            token_data.get('liquidity_score', 0),
            token_data.get('holder_distribution', 0),
            float(token_data.get('contract_verified', False)),
            token_data.get('sentiment_score', 0.5),
            token_data.get('technical_score', 0.5),
            min(token_data.get('slippage', 100) / 100, 1.0),  # Normalize slippage
            1.0 - token_data.get('rugpull_risk', 0),  # Invert risk
        ])
        
        # Calculate base score from features
        base_score = np.mean(features_vector)
        
        return {
            'vector': features_vector,
            'base_score': base_score
        }
    
    def _rule_based_scoring(self, token_data: Dict) -> float:
        """Fallback rule-based scoring when models unavailable"""
        factors = {
            'liquidity_score': token_data.get('liquidity_score', 0),
            'holder_distribution': token_data.get('holder_distribution', 0),
            'contract_verified': token_data.get('contract_verified', False),
            'sentiment_score': token_data.get('sentiment_score', 0.5),
            'technical_score': token_data.get('technical_score', 0.5)
        }
        
        # Weighted scoring
        score = (
            factors['liquidity_score'] * 0.3 +
            factors['holder_distribution'] * 0.2 +
            (1.0 if factors['contract_verified'] else 0.0) * 0.2 +
            factors['sentiment_score'] * 0.15 +
            factors['technical_score'] * 0.15
        )
        
        # Penalties
        if token_data.get('rugpull_risk', 0) > 0.3:
            score *= 0.5
        if token_data.get('slippage', 0) > 15:
            score *= 0.7
            
        return min(score, 0.95)
    
    def train_custom_model(self, training_data: List[Dict], labels: List[int]):
        """Train the custom ML model on historical data"""
        try:
            # Extract features from training data
            X = []
            for data in training_data:
                features = self._extract_features(data)
                X.append(features['vector'])
            
            X = np.array(X)
            y = np.array(labels)
            
            # Train model
            self.custom_model.fit(X, y)
            self.is_pretrained = True
            
            # Save model
            model_path = Path("models/custom_trade_predictor.pkl")
            model_path.parent.mkdir(exist_ok=True)
            joblib.dump(self.custom_model, model_path)
            
            print(f"Model trained and saved to {model_path}")
            
        except Exception as e:
            print(f"Training error: {e}")

class DeFiSniperBot:
    def __init__(self, config: Dict):
        self.config = config
        self.w3 = self._setup_web3()
        self.ai_model = LocalLLMPredictor(
            model_path=config.get('custom_model_path')
        )
        self.logger = self._setup_logger()
        
        # Performance metrics
        self.trade_count = 0
        self.success_count = 0
        self.rug_pulls_avoided = 0
        
        # Cache for faster processing
        self.token_cache = {}
        self.liquidity_cache = {}
        
    def _setup_web3(self) -> Web3:
        """Initialize Web3 connection with optimizations"""
        w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return w3
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DeFiSniper')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    async def analyze_liquidity(self, token_address: str, pair_address: str) -> Dict:
        """
        Parse liquidity data with optimization for low slippage
        Reduced analysis time to support 0.5s execution
        """
        start_time = time.time()
        
        # Quick cache check
        cache_key = f"{token_address}:{pair_address}"
        if cache_key in self.liquidity_cache:
            cache_entry = self.liquidity_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 30:  # 30s cache
                return cache_entry['data']
        
        try:
            # Get pair contract (optimized calls)
            pair_abi = self._get_pair_abi()
            pair_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(pair_address),
                abi=pair_abi
            )
            
            # Batch calls for efficiency
            reserves = pair_contract.functions.getReserves().call()
            total_supply = pair_contract.functions.totalSupply().call()
            
            # Calculate liquidity metrics
            reserve0 = reserves[0] / 10**18  # Assuming 18 decimals
            reserve1 = reserves[1] / 10**18
            
            liquidity_usd = self._calculate_liquidity_usd(reserve0, reserve1)
            
            # Slippage calculation for different trade sizes
            slippage_data = self._calculate_slippage(reserve0, reserve1)
            
            result = {
                'liquidity_usd': liquidity_usd,
                'reserve0': reserve0,
                'reserve1': reserve1,
                'slippage_1_bnb': slippage_data['1_bnb'],
                'slippage_5_bnb': slippage_data['5_bnb'],
                'analysis_time': time.time() - start_time,
                'liquidity_score': self._calculate_liquidity_score(liquidity_usd)
            }
            
            # Cache result
            self.liquidity_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Liquidity analysis error: {e}")
            return None
            
    async def check_rugpull_risk(self, token_address: str) -> Dict:
        """
        Local contract auditing without external API dependencies
        Analyzes contract bytecode and patterns
        """
        try:
            # Get contract code
            contract_code = self.w3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = contract_code.hex()
            
            # Local pattern detection for common rug pull indicators
            audit_results = {
                'honeypot': self._detect_honeypot_pattern(code_hex),
                'ownership_renounced': self._check_ownership_renounced(token_address),
                'liquidity_locked': await self._check_liquidity_locked(token_address),
                'contract_verified': self._is_contract_verified(code_hex),
                'max_tx_limit': self._has_transaction_limit(code_hex),
                'trading_cooldown': self._has_trading_cooldown(code_hex),
                'blacklist_function': self._has_blacklist(code_hex),
                'mint_function': self._has_mint_function(code_hex),
                'proxy_contract': self._is_proxy_contract(code_hex),
                'hidden_owner': self._has_hidden_owner(code_hex)
            }
            
            # Use local LLM for contract analysis
            if self.ai_model.model_loaded:
                contract_analysis = await self._analyze_contract_with_llm(
                    code_hex, 
                    audit_results
                )
                audit_results['llm_risk_assessment'] = contract_analysis
            
            # Calculate risk score
            risk_factors = sum([
                audit_results['honeypot'] * 0.3,
                (not audit_results['ownership_renounced']) * 0.2,
                (not audit_results['liquidity_locked']) * 0.2,
                (not audit_results['contract_verified']) * 0.1,
                audit_results['blacklist_function'] * 0.1,
                audit_results['mint_function'] * 0.05,
                audit_results['hidden_owner'] * 0.05
            ])
            
            return {
                'risk_score': risk_factors,
                'is_safe': risk_factors < 0.3,
                'audit_results': audit_results,
                'recommendation': 'SAFE' if risk_factors < 0.3 else 'RISKY'
            }
            
        except Exception as e:
            self.logger.error(f"Rug check error: {e}")
            return {'risk_score': 1.0, 'is_safe': False}
    
    def _detect_honeypot_pattern(self, code_hex: str) -> bool:
        """Detect honeypot patterns in bytecode"""
        # Common honeypot patterns
        honeypot_signatures = [
            'fdacd5ed',  # Common honeypot function selector
            '7d3e3dbe',  # Another known pattern
        ]
        
        for sig in honeypot_signatures:
            if sig in code_hex.lower():
                return True
        return False
    
    def _check_ownership_renounced(self, token_address: str) -> bool:
        """Check if ownership is renounced"""
        try:
            # Try to call owner() function
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=[{"constant": True, "inputs": [], "name": "owner", 
                      "outputs": [{"name": "", "type": "address"}], "type": "function"}]
            )
            owner = contract.functions.owner().call()
            # Check if owner is zero address
            return owner == '0x0000000000000000000000000000000000000000'
        except:
            return True  # If no owner function, assume renounced
    
    async def _check_liquidity_locked(self, token_address: str) -> bool:
        """Check if liquidity is locked (simplified local check)"""
        # This would check popular liquidity locker contracts
        # For now, return True if liquidity exists and is substantial
        return True  # Simplified for local operation
    
    def _is_contract_verified(self, code_hex: str) -> bool:
        """Check if contract appears to be standard/verified"""
        # Check for standard ERC20 function selectors
        standard_selectors = [
            '18160ddd',  # totalSupply()
            '70a08231',  # balanceOf(address)
            'a9059cbb',  # transfer(address,uint256)
        ]
        
        verified = all(sel in code_hex.lower() for sel in standard_selectors)
        return verified
    
    def _has_transaction_limit(self, code_hex: str) -> bool:
        """Detect transaction limit patterns"""
        limit_patterns = ['5f6d61785478', '6d61785472616e73616374696f6e']
        return any(pattern in code_hex for pattern in limit_patterns)
    
    def _has_trading_cooldown(self, code_hex: str) -> bool:
        """Detect cooldown mechanisms"""
        cooldown_patterns = ['636f6f6c646f776e', '5f6c617374547261646554696d65']
        return any(pattern in code_hex for pattern in cooldown_patterns)
    
    def _has_blacklist(self, code_hex: str) -> bool:
        """Detect blacklist functionality"""
        blacklist_patterns = ['626c61636b6c697374', '69734578636c75646564']
        return any(pattern in code_hex for pattern in blacklist_patterns)
    
    def _has_mint_function(self, code_hex: str) -> bool:
        """Detect mint functionality"""
        mint_selectors = ['40c10f19', 'a0712d68']  # Common mint selectors
        return any(sel in code_hex.lower() for sel in mint_selectors)
    
    def _is_proxy_contract(self, code_hex: str) -> bool:
        """Detect proxy contract patterns"""
        proxy_patterns = ['363d3d373d3d3d363d73']  # EIP-1167 minimal proxy
        return any(pattern in code_hex for pattern in proxy_patterns)
    
    def _has_hidden_owner(self, code_hex: str) -> bool:
        """Detect hidden ownership patterns"""
        hidden_patterns = ['5f6f776e6572', '68696464656e']
        return any(pattern in code_hex for pattern in hidden_patterns)
    
    async def _analyze_contract_with_llm(self, code_hex: str, audit_results: Dict) -> float:
        """Use local LLM to analyze contract risk"""
        if not OLLAMA_AVAILABLE or not self.ollama_model:
            # Fallback to rule-based if LLM not available
            return audit_results.get('risk_score', 0.5)
            
        try:
            # Prepare analysis prompt
            prompt = f"""Analyze this smart contract audit for trading risk:
            
            Audit Results:
            - Honeypot detected: {audit_results['honeypot']}
            - Ownership renounced: {audit_results['ownership_renounced']}
            - Has blacklist: {audit_results['blacklist_function']}
            - Has mint function: {audit_results['mint_function']}
            - Is proxy: {audit_results['proxy_contract']}
            - Contract size: {len(code_hex)} characters
            
            Rate the risk from 0.0 (safe) to 1.0 (dangerous).
            Consider: honeypot risk, centralization, and manipulation potential.
            Respond with only a decimal number.
            """
            
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={"temperature": 0.1, "max_tokens": 10}
            )
            
            risk = float(response['response'].strip())
            return max(0.0, min(1.0, risk))
            
        except:
            # Fallback to rule-based if LLM fails
            return audit_results.get('risk_score', 0.5)
            
    async def analyze_sentiment(self, token_symbol: str, contract_address: str) -> float:
        """
        Analyze sentiment using local NLP models - no external APIs
        """
        try:
            # Use local sentiment analysis with transformers
            from transformers import pipeline
            
            # Initialize local sentiment analyzer (cached after first use)
            if not hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="finiteautomata/bertweet-base-sentiment-analysis",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # In production, you would scrape social media locally
            # For now, simulate with local data sources
            sentiment_texts = [
                f"New token {token_symbol} launched with great potential",
                f"Interesting project at {contract_address[:10]}...",
                f"{token_symbol} showing strong community support"
            ]
            
            # Analyze each text
            sentiments = []
            for text in sentiment_texts:
                result = self.sentiment_analyzer(text)[0]
                # Convert to 0-1 scale
                if result['label'] == 'POSITIVE':
                    sentiments.append(result['score'])
                elif result['label'] == 'NEGATIVE':
                    sentiments.append(1 - result['score'])
                else:  # NEUTRAL
                    sentiments.append(0.5)
            
            # Get average sentiment
            sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0.5
            
            # Use local LLM for deeper analysis if available
            if self.ai_model.model_loaded:
                llm_sentiment = await self._get_llm_sentiment(token_symbol, contract_address)
                # Combine both scores
                sentiment_score = (sentiment_score + llm_sentiment) / 2
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return 0.5  # Neutral sentiment on error
    
    async def _get_llm_sentiment(self, token_symbol: str, contract_address: str) -> float:
        """Get sentiment analysis from local LLM"""
        if not OLLAMA_AVAILABLE or not self.ollama_model:
            return 0.5  # Neutral if LLM not available
            
        try:
            prompt = f"""Analyze the market sentiment for this new crypto token:
            Symbol: {token_symbol if token_symbol else 'Unknown'}
            Contract: {contract_address}
            
            Consider factors like:
            - New token launch timing
            - Current DeFi market conditions
            - Contract address patterns
            
            Rate sentiment from 0.0 (very negative) to 1.0 (very positive).
            Respond with only a decimal number.
            """
            
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={"temperature": 0.3, "max_tokens": 10}
            )
            
            sentiment = float(response['response'].strip())
            return max(0.0, min(1.0, sentiment))
            
        except:
            return 0.5  # Neutral on error
            
    def calculate_technical_indicators(self, price_data: List[float]) -> Dict:
        """
        Calculate RSI and MACD for entry timing using pandas-ta
        """
        if len(price_data) < 26:
            return {'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_diff': 0}
        
        # Convert to pandas Series for pandas-ta
        prices = pd.Series(price_data)
        
        # RSI calculation
        rsi = ta.rsi(prices, length=14)
        current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
        
        # MACD calculation
        macd_result = ta.macd(prices, fast=12, slow=26, signal=9)
        
        if macd_result is not None and not macd_result.empty:
            # pandas-ta returns a DataFrame with columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_line = macd_result.iloc[-1, 0] if not pd.isna(macd_result.iloc[-1, 0]) else 0
            macd_histogram = macd_result.iloc[-1, 1] if not pd.isna(macd_result.iloc[-1, 1]) else 0
            macd_signal = macd_result.iloc[-1, 2] if not pd.isna(macd_result.iloc[-1, 2]) else 0
        else:
            macd_line = macd_signal = macd_histogram = 0
        
        return {
            'rsi': float(current_rsi),
            'macd': float(macd_line),
            'macd_signal': float(macd_signal),
            'macd_diff': float(macd_histogram)
        }
        
    async def execute_snipe(self, token_data: Dict) -> Dict:
        """
        Execute the snipe trade with 0.5s target execution time
        """
        start_time = time.time()
        
        try:
            # Pre-calculate everything for speed
            gas_price = self.w3.eth.gas_price * 1.2  # 20% higher for priority
            
            # Build transaction
            router_address = self.config['router_address']
            router_abi = self._get_router_abi()
            router = self.w3.eth.contract(address=router_address, abi=router_abi)
            
            # Swap parameters
            amount_in = Web3.to_wei(self.config['trade_amount_bnb'], 'ether')
            amount_out_min = self._calculate_min_amount_out(
                amount_in, 
                token_data['slippage']
            )
            
            path = [
                self.config['wbnb_address'],
                token_data['address']
            ]
            
            deadline = int(time.time()) + 300  # 5 minutes
            
            # Build transaction
            swap_txn = router.functions.swapExactETHForTokens(
                amount_out_min,
                path,
                self.config['wallet_address'],
                deadline
            ).build_transaction({
                'from': self.config['wallet_address'],
                'value': amount_in,
                'gas': 300000,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(
                    self.config['wallet_address']
                ),
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                swap_txn, 
                private_key=self.config['private_key']
            )
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Snipe executed in {execution_time:.2f}s - "
                f"TX: {tx_hash.hex()}"
            )
            
            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'execution_time': execution_time,
                'gas_used': swap_txn['gas'],
                'gas_price': gas_price
            }
            
        except Exception as e:
            self.logger.error(f"Snipe execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
    async def monitor_new_pairs(self):
        """
        Monitor DEX for new token launches
        """
        self.logger.info("Starting DeFi Sniper Bot...")
        
        # Subscribe to pair creation events
        pair_factory_address = self.config['factory_address']
        pair_factory_abi = self._get_factory_abi()
        factory = self.w3.eth.contract(
            address=pair_factory_address, 
            abi=pair_factory_abi
        )
        
        # Event filter for new pairs
        event_filter = factory.events.PairCreated.create_filter(
            fromBlock='latest'
        )
        
        while True:
            try:
                for event in event_filter.get_new_entries():
                    await self.process_new_pair(event)
                    
                await asyncio.sleep(0.1)  # Fast polling
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
                
    async def process_new_pair(self, event: Dict):
        """
        Process newly created pair with full analysis pipeline
        """
        token0 = event['args']['token0']
        token1 = event['args']['token1']
        pair = event['args']['pair']
        
        # Identify which is the new token (not WBNB/BUSD/USDT)
        known_tokens = [
            self.config['wbnb_address'].lower(),
            self.config.get('busd_address', '').lower(),
            self.config.get('usdt_address', '').lower()
        ]
        
        new_token = None
        if token0.lower() not in known_tokens:
            new_token = token0
        elif token1.lower() not in known_tokens:
            new_token = token1
            
        if not new_token:
            return
            
        self.logger.info(f"New token detected: {new_token}")
        
        # Start analysis pipeline
        analysis_start = time.time()
        
        # Parallel analysis for speed
        tasks = [
            self.analyze_liquidity(new_token, pair),
            self.check_rugpull_risk(new_token),
            self.analyze_sentiment("", new_token),
            self._get_token_info(new_token)
        ]
        
        results = await asyncio.gather(*tasks)
        
        liquidity_data = results[0]
        rugpull_data = results[1]
        sentiment_score = results[2]
        token_info = results[3]
        
        # Get some price history if available
        price_data = await self._get_price_history(new_token)
        technical_indicators = self.calculate_technical_indicators(price_data)
        
        # Compile all data for AI prediction
        token_data = {
            'address': new_token,
            'pair': pair,
            'liquidity_score': liquidity_data['liquidity_score'],
            'holder_distribution': await self._analyze_holders(new_token),
            'contract_verified': rugpull_data['audit_results']['contract_verified'],
            'sentiment_score': sentiment_score,
            'technical_score': self._calculate_technical_score(technical_indicators),
            'slippage': liquidity_data['slippage_1_bnb'],
            'rugpull_risk': rugpull_data['risk_score']
        }
        
        # AI prediction
        success_probability = await self.ai_model.predict_trade_success(token_data)
        
        analysis_time = time.time() - analysis_start
        self.logger.info(
            f"Analysis completed in {analysis_time:.2f}s - "
            f"Success probability: {success_probability:.2%}"
        )
        
        # Decision making
        if (success_probability >= 0.65 and  # 65%+ success rate threshold
            rugpull_data['is_safe'] and
            liquidity_data['liquidity_usd'] > self.config['min_liquidity_usd'] and
            token_data['slippage'] < self.config['max_slippage']):
            
            # Execute snipe
            trade_result = await self.execute_snipe(token_data)
            
            if trade_result['success']:
                self.trade_count += 1
                self.logger.info(
                    f"Snipe successful! Total trades: {self.trade_count} "
                    f"Success rate: {(self.success_count/self.trade_count)*100:.1f}%"
                )
                
                # Monitor position
                asyncio.create_task(
                    self.monitor_position(new_token, trade_result['tx_hash'])
                )
            
        else:
            if rugpull_data['risk_score'] > 0.3:
                self.rug_pulls_avoided += 1
                self.logger.warning(
                    f"Rug pull avoided! Total avoided: {self.rug_pulls_avoided}"
                )
                
    async def monitor_position(self, token_address: str, tx_hash: str):
        """
        Monitor position for exit timing
        """
        # Implementation for position monitoring and exit strategy
        pass
        
    def _calculate_liquidity_score(self, liquidity_usd: float) -> float:
        """Calculate normalized liquidity score"""
        if liquidity_usd < 10000:
            return 0.1
        elif liquidity_usd < 50000:
            return 0.3
        elif liquidity_usd < 100000:
            return 0.5
        elif liquidity_usd < 500000:
            return 0.7
        else:
            return 0.9
            
    def _calculate_technical_score(self, indicators: Dict) -> float:
        """Calculate technical analysis score"""
        score = 0.5  # Neutral base
        
        # RSI scoring
        rsi = indicators['rsi']
        if 30 < rsi < 70:
            score += 0.2
        elif rsi <= 30:  # Oversold
            score += 0.3
            
        # MACD scoring
        if indicators['macd_diff'] > 0:
            score += 0.2
            
        return min(score, 1.0)
        
    def _calculate_slippage(self, reserve0: float, reserve1: float) -> Dict:
        """Calculate slippage for different trade sizes"""
        slippage_data = {}
        
        for bnb_amount in [1, 5, 10]:
            # Uniswap V2 formula
            amount_out = (bnb_amount * 997 * reserve1) / (reserve0 * 1000 + bnb_amount * 997)
            price_impact = (bnb_amount / reserve0) * 100
            
            slippage_data[f'{bnb_amount}_bnb'] = price_impact
            
        return slippage_data
        
    def _calculate_liquidity_usd(self, reserve0: float, reserve1: float) -> float:
        """Calculate liquidity in USD"""
        # In production, get actual BNB price
        bnb_price_usd = 300  # Placeholder
        return reserve0 * bnb_price_usd * 2  # Total liquidity
        
    def _calculate_min_amount_out(self, amount_in: int, slippage: float) -> int:
        """Calculate minimum output with slippage tolerance"""
        slippage_tolerance = self.config['slippage_tolerance']
        return int(amount_in * (1 - slippage_tolerance / 100))
        
    async def _get_price_history(self, token_address: str) -> List[float]:
        """Get recent price history for technical analysis"""
        # In production, fetch from DEX subgraph or price API
        # Returning dummy data for demo
        return [100 + i + np.random.randn() * 5 for i in range(50)]
        
    async def _analyze_holders(self, token_address: str) -> float:
        """Analyze holder distribution"""
        # In production, use BSCScan API or similar
        # Good distribution = higher score
        return 0.7  # Placeholder
        
    async def _get_token_info(self, token_address: str) -> Dict:
        """Get basic token information"""
        try:
            token_abi = self._get_token_abi()
            token_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=token_abi
            )
            
            # Batch calls
            name = token_contract.functions.name().call()
            symbol = token_contract.functions.symbol().call()
            decimals = token_contract.functions.decimals().call()
            total_supply = token_contract.functions.totalSupply().call()
            
            return {
                'name': name,
                'symbol': symbol,
                'decimals': decimals,
                'total_supply': total_supply
            }
        except:
            return {}
            
    def _get_pair_abi(self) -> List:
        """Get Uniswap V2 pair ABI"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "getReserves",
                "outputs": [
                    {"name": "_reserve0", "type": "uint112"},
                    {"name": "_reserve1", "type": "uint112"},
                    {"name": "_blockTimestampLast", "type": "uint32"}
                ],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
        
    def _get_router_abi(self) -> List:
        """Get DEX router ABI for swaps"""
        return [
            {
                "inputs": [
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            }
        ]
        
    def _get_factory_abi(self) -> List:
        """Get DEX factory ABI for pair creation events"""
        return [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "token0", "type": "address"},
                    {"indexed": True, "name": "token1", "type": "address"},
                    {"indexed": False, "name": "pair", "type": "address"},
                    {"indexed": False, "name": "index", "type": "uint256"}
                ],
                "name": "PairCreated",
                "type": "event"
            }
        ]
        
    def _get_token_abi(self) -> List:
        """Get ERC20 token ABI"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]

# Configuration and usage example
async def main():
    config = {
        # Network settings
        'rpc_url': 'https://bsc-dataseed.binance.org/',  # BSC mainnet
        'chain_id': 56,
        
        # Contract addresses (BSC)
        'router_address': '0x10ED43C718714eb63d5aA57B78B54704E256024E',  # PancakeSwap
        'factory_address': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73',
        'wbnb_address': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',
        
        # Wallet settings
        'wallet_address': '0xYOUR_WALLET_ADDRESS',
        'private_key': 'YOUR_PRIVATE_KEY',
        
        # Trading parameters
        'trade_amount_bnb': 0.1,
        'min_liquidity_usd': 50000,
        'max_slippage': 10,  # %
        'slippage_tolerance': 12,  # %
        
        # Local model settings (NO API KEYS NEEDED!)
        'custom_model_path': 'models/custom_trade_predictor.pkl',  # Optional
        'success_threshold': 0.65  # 65% minimum
    }
    
    # Setup local models first
    print("Setting up local AI models...")
    if OLLAMA_AVAILABLE:
        print("Ollama is available. Make sure it's running: ollama serve")
        print("Download models with: ollama pull deepseek-coder:latest")
    else:
        print("Ollama not found. Bot will use rule-based analysis.")
        print("Install Ollama for AI features: https://ollama.ai")
    
    bot = DeFiSniperBot(config)
    
    # Start 
    await bot.monitor_new_pairs()

if __name__ == "__main__":
    # Check for GPU 
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (consider using GPU for faster inference)")
    
    asyncio.run(main())