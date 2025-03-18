import streamlit as st
import httpx
import os
import base64
import asyncio
import json
from datetime import datetime
from httpx_ws import aconnect_ws
import logging
from swaps_tokens.contract_to_tea import ContractToTea
from swaps_tokens.tea_to_contract import TeaToContract 
from data import data
from web3 import Web3
from decimal import Decimal
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.ensemble import IsolationForest
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="teaAssam-testnet",
    page_icon="awokawok",
    layout="wide",
    initial_sidebar_state="expanded"
)

RPC_URL = "https://assam-rpc.tea.xyz"
CHAIN_ID = 93384
SYMBOL = "$TEA"
w3 = Web3(Web3.HTTPProvider(RPC_URL))


ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "success", "type": "bool"}],
        "type": "function"
    }
]

def load_custom_font():
    try:
        font_path = os.path.join('static', 'fonts', 'maple.otf')
        if not os.path.exists(font_path):
            st.error("⚠️ Font file not found!")
            logger.error("Font file not found at path: %s", font_path)
            return ""
        with open(font_path, "rb") as font_file:
            font_data = base64.b64encode(font_file.read()).decode()
        return f"""
        <style>
        @font-face {{
            font-family: 'Maple';
            src: url(data:font/ttf;charset=utf-8;base64,{font_data}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        html, body, [class*="st-"], .sidebar-content, .stChatMessage, .stButton button {{
            font-family: 'Maple', sans-serif !important;
        }}
        .markdown-text {{
            user-select: text !important;
            -webkit-user-select: text !important;
        }}
        .scrollable-markdown {{
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #FF4B4B;
            padding: 15px;
            background-color: #0a0a0a;
            border-radius: 15px;
        }}
        .stMetric {{
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }}
        .stButton > button {{
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }}
        </style>
        """
    except Exception as e:
        st.error(f"⚠️ Error loading font: {str(e)}")
        logger.error("Error loading font: %s", str(e))
        return ""

custom_font_css = load_custom_font()
if custom_font_css:
    st.markdown(custom_font_css, unsafe_allow_html=True)

API_KEY = "fw_3Zhy9f3UZMdTtiQTiQ12hU6X"
CHAT_URL = "https://assam-rpc.tea.xyz/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

AVAILABLE_MODELS = [
    "accounts/fireworks/models/deepseek-r1",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
]

MODEL_NAMES = {
    "accounts/fireworks/models/deepseek-r1": "DeepSeek (R1)",
    "accounts/fireworks/models/deepseek-v3": "DeepSeek (v3)",
    "accounts/fireworks/models/llama-v3p1-405b-instruct": "Qwent (2.5 max)",
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct": "Qwent (coding)",
}

if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[0]
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.6
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 16384
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "top_k" not in st.session_state:
    st.session_state.top_k = 50
if "presence_penalty" not in st.session_state:
    st.session_state.presence_penalty = 0.0
if "frequency_penalty" not in st.session_state:
    st.session_state.frequency_penalty = 0.0
if "current_extension" not in st.session_state:
    st.session_state.current_extension = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Kontol AI! (uncensored, search query, slow)"}
    ]
if "token_balance" not in st.session_state:
    st.session_state.token_balance = 0.0
if "contract_balance" not in st.session_state:
    st.session_state.contract_balance = 0.0
if "contract_address" not in st.session_state:
    st.session_state.contract_address = ""

def get_tea_balance(private_key):
    try:
        account = w3.eth.account.from_key(private_key)
        address = account.address
        balance_wei = w3.eth.get_balance(address)
        balance_tea = w3.from_wei(balance_wei, 'ether')
        logger.info(f"Balance fetched for {address}: {balance_tea} $TEA")
        return float(balance_tea)
    except Exception as e:
        st.error(f"Error fetching $TEA balance: {str(e)}")
        logger.error(f"Error fetching $TEA balance: {str(e)}")
        return 0.0

def get_contract_balance(private_key, contract_address):
    try:
        if not w3.is_address(contract_address):
            return 0.0
        account = w3.eth.account.from_key(private_key)
        address = account.address
        contract = w3.eth.contract(address=contract_address, abi=ERC20_ABI)
        balance_wei = contract.functions.balanceOf(address).call()
        balance_token = w3.from_wei(balance_wei, 'ether')          
        logger.info(f"Contract balance fetched for {address} on {contract_address}: {balance_token}")
        return float(balance_token)
    except Exception as e:
        st.error(f"Error fetching contract balance: {str(e)}")
        logger.error(f"Error fetching contract balance: {str(e)}")
        return 0.0

def swaptoken(user, private, contract, amount, gas):
    st = ContractToTea(
        user_address=user,
        private_key=private,
        contract_address=contract,
        amount_in=amount,
        gasprice=gas
    )
    return st.eksekusi_swap()

def withdraw_contract_to_tea(user_address, private_key, contract_address, amount, gas_price_gwei):
    try:
        account = w3.eth.account.from_key(private_key)
        contract = w3.eth.contract(address=contract_address, abi=ERC20_ABI)
        amount_wei = w3.to_wei(amount, 'ether')  

        nonce = w3.eth.get_transaction_count(account.address)
        gas_price = w3.to_wei(gas_price_gwei, 'gwei')
        tx = contract.functions.transfer(account.address, amount_wei).build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 200000, 
            'gasPrice': gas_price,
            'chainId': CHAIN_ID
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if tx_receipt.status == 1:
            return {
                "status": "success",
                "tx_hash": tx_hash.hex(),
                "amount": amount,
                "contract_address": contract_address
            }
        else:
            return {
                "status": "failed",
                "tx_hash": tx_hash.hex(),
                "error": "Transaction failed"
            }
    except Exception as e:
        logger.error(f"Error withdrawing from contract: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def get_recent_blocks(w3, num_blocks=5):
    latest_block = w3.eth.block_number
    blocks = []
    for i in range(latest_block, max(latest_block - num_blocks, 0), -1):
        try:
            block = w3.eth.get_block(i, full_transactions=True)
            blocks.append(block)
        except Exception as e:
            st.error(f"Error retrieving block {i}: {e}")
    return blocks

def detect_abnormal_transactions(blocks):
    txs = []
    for block in blocks:
        for tx in block.transactions:
            txs.append(tx)
    
    if len(txs) == 0:
        return np.array([])

    X = np.array([[float(tx["value"])] for tx in txs])

    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(X)

    abnormal_flags = (preds == -1)
    return abnormal_flags

def build_graph(blocks, abnormal_flags):
    net = Network(
        height="100vh",
        width="100vw",
        directed=True,
        bgcolor="black",
        font_color="white"
    )

    net.force_atlas_2based(gravity=-30)

    i = 0
    for block in blocks:
        for tx in block.transactions:
            abnormal = abnormal_flags[i] if i < len(abnormal_flags) else False
            from_addr = tx["from"]
            to_addr = tx["to"]
            value_wei = tx["value"]
            value_eth = w3.from_wei(value_wei, 'ether')

            net.add_node(from_addr, label=f"Sender\n{from_addr}", color="blue")

            if to_addr:
                net.add_node(to_addr, label=f"Receiver\n{to_addr}", color="#0a7e89")
                edge_color = "red" if abnormal else "#888888"
                net.add_edge(from_addr, to_addr, value=value_wei, color=edge_color, title=f"{value_eth} TEA")
            
            i += 1
    return net

def get_ai_response(messages):
    payload = {
        "model": st.session_state.selected_model,
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "top_k": st.session_state.top_k,
        "presence_penalty": st.session_state.presence_penalty,
        "frequency_penalty": st.session_state.frequency_penalty,
        "messages": messages
    }
    try:
        logger.info("Sending request to AI model: %s", st.session_state.selected_model)
        response = httpx.post(CHAT_URL, headers=HEADERS, json=payload, timeout=60.0)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        logger.info("Received response from AI: %s", content[:50])
        return content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error("Error getting AI response: {str(e)}")
        return None

def update_balances(contract_address):
    private_key = data()["pk"]
    st.session_state.token_balance = get_tea_balance(private_key)
    if contract_address:
        st.session_state.contract_balance = get_contract_balance(private_key, contract_address)
    else:
        st.session_state.contract_balance = 0.0

def format_transaction_log(tx_result, is_swap=False):
    if is_swap:
        try:
            result = json.loads(tx_result)
            return (
                f"**Token Balance:** {result['token_balance']}\n"
                f"**Swap Path:** {result['swap_path']}\n"
                f"**Approval Tx Hash:** {result['approval_tx_hash']}\n"
                f"**Swap Tx Hash:** {result['swap_tx_hash']}\n"
                f"**Approval Status:** {result['approval_status']}\n"
                f"**Swap Status:** {result['swap_status']}"
            )
        except Exception as e:
            logger.error(f"Error formatting transaction log: {str(e)}")
            return tx_result
    else:
        if tx_result["status"] == "success":
            return (
                f"**Amount Withdrawn:** {tx_result['amount']:.5f} Token\n"
                f"**Contract Address:** {tx_result['contract_address']}\n"
                f"**Tx Hash:** {tx_result['tx_hash']}"
            )
        else:
            return f"**Error:** {tx_result.get('error', 'Unknown error')}"

with st.sidebar:
    st.title("Settings")
    selected_model_name = st.selectbox(
        "Select AI Model",
        options=[MODEL_NAMES[model] for model in AVAILABLE_MODELS],
        index=AVAILABLE_MODELS.index(st.session_state.selected_model),
        help="Choose the AI model you want to use."
    )
    technical_model_name = list(MODEL_NAMES.keys())[list(MODEL_NAMES.values()).index(selected_model_name)]
    if technical_model_name != st.session_state.selected_model:
        st.session_state.selected_model = technical_model_name
        st.rerun()
    
    st.divider()
    st.subheader("Generation Parameters")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.01)
    st.session_state.max_tokens = st.number_input("Max Tokens", 1, 20480, st.session_state.max_tokens, 1)
    st.session_state.top_p = st.slider("Top P", 0.0, 1.0, st.session_state.top_p, 0.01)
    st.session_state.top_k = st.number_input("Top K", 1, 100, st.session_state.top_k, 1)
    st.session_state.presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, st.session_state.presence_penalty, 0.01)
    st.session_state.frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, st.session_state.frequency_penalty, 0.01)
    
    st.divider()
    st.subheader("TeaAssam (tools)")
    extension_options = ["swaptoken", "swapcontract", "dexchecker", "transfer", "transferbatch", "addliqudity"]
    selected_extension = st.selectbox("", options=["None"] + extension_options, index=0)
    st.session_state.current_extension = selected_extension if selected_extension != "None" else None
    
    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to Kontol AI! (uncensored, search query, slow)"}]
        st.rerun()

if st.session_state.current_extension:
    extension = st.session_state.current_extension
    
    if extension == "swaptoken":
        st.header("Swap $TEA → Contract")
        with st.form(key="swaptoken_form"):
            token_amount = st.number_input(
                "Jumlah $TEA:", 
                min_value=0.0,
                max_value=float(st.session_state.token_balance),
                value=0.0,
                step=0.1,
                key="token_swap",
                help="Masukkan jumlah $TEA yang ingin Anda swap."
            )
            contract_address = st.text_input(
                "Alamat Kontrak:", 
                key="contract_swaptoken",
                help="Masukkan alamat kontrak tujuan."
            )
            gas_price = st.number_input(
                "Gas Price (Gwei):", 
                min_value=1, 
                value=2000, 
                key="gas_swaptoken",
                help="Masukkan harga gas dalam Gwei (minimum 1 Gwei)."
            )
            submit_button = st.form_submit_button("Swap $TEA → Contract")

            if submit_button:
                if token_amount <= 0:
                    st.error("Jumlah $TEA harus lebih dari 0!")
                elif not contract_address.strip():
                    st.error("Alamat kontrak tidak boleh kosong!")
                elif gas_price < 1:
                    st.error("Gas price harus minimal 1 Gwei!")
                else:
                    with st.spinner("Processing transaction..."):
                        try:
                            tx_result = swaptoken(
                                user=data()["us"],
                                private=data()["pk"],
                                contract=contract_address,
                                amount=token_amount,
                                gas=gas_price
                            )
                            formatted_result = format_transaction_log(tx_result, is_swap=True)
                            st.success("Transaksi berhasil!")
                            st.markdown(formatted_result, unsafe_allow_html=True)
                            st.session_state.token_balance = float(Decimal(st.session_state.token_balance) - Decimal(token_amount))
                            st.session_state.contract_balance = float(Decimal(st.session_state.contract_balance) + Decimal(token_amount))
                            update_balances(contract_address)
                        except Exception as e:
                            st.error(f"Transaksi gagal: {str(e)}")
                            logger.error(f"Transaction failed: {str(e)}")

        st.subheader("Saldo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Saldo $TEA Anda:", f"{st.session_state.token_balance:.5f} $TEA")
        with col2:
            st.metric("Saldo Kontrak:", f"{st.session_state.contract_balance:.5f} Token")
    
    elif extension == "swapcontract":
        st.header("Withdraw Contract → $TEA")
        with st.form(key="swapcontract_form"):
            contract_address = st.text_input(
                "Alamat Kontrak:", 
                value="0xC4341CB2C976306AE9169efb3d8301ea287a3128",
                key="contract_swapcontract",
                help="Masukkan alamat kontrak sumber."
            )
            
            if contract_address != st.session_state.contract_address:
                st.session_state.contract_address = contract_address
                update_balances(contract_address)
                st.rerun()

            withdraw_amount = st.number_input(
                "Jumlah Penarikan:", 
                min_value=0.0,
                max_value=float(st.session_state.contract_balance),
                value=0.0,
                step=0.1,
                key="withdraw_swap",
                help="Masukkan jumlah yang ingin Anda tarik."
            )
            gas_price = st.number_input(
                "Gas Price (Gwei):", 
                min_value=1, 
                value=2000, 
                key="gas_swapcontract",
                help="Masukkan harga gas dalam Gwei (minimum 1 Gwei)."
            )
            submit_button = st.form_submit_button("Withdraw Contract → $TEA")

            if submit_button:
                if withdraw_amount <= 0:
                    st.error("Jumlah penarikan harus lebih dari 0!")
                elif not contract_address.strip():
                    st.error("Alamat kontrak tidak boleh kosong!")
                elif gas_price < 1:
                    st.error("Gas price harus minimal 1 Gwei!")
                else:
                    with st.spinner("Processing withdrawal..."):
                        try:
                            user_address = data()["us"]
                            private_key = data()["pk"]
                            tx_result = withdraw_contract_to_tea(
                                user_address=user_address,
                                private_key=private_key,
                                contract_address=contract_address,
                                amount=withdraw_amount,
                                gas_price_gwei=gas_price
                            )
                            formatted_result = format_transaction_log(tx_result, is_swap=False)
                            if tx_result["status"] == "success":
                                st.success("Penarikan berhasil!")
                                st.markdown(formatted_result, unsafe_allow_html=True)
                                update_balances(contract_address)
                            else:
                                st.error("Penarikan gagal!")
                                st.markdown(formatted_result, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Penarikan gagal: {str(e)}")
                            logger.error(f"Withdrawal failed: {str(e)}")

        st.subheader("Saldo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Saldo $TEA Anda:", f"{st.session_state.token_balance:.5f} $TEA")
        with col2:
            st.metric("Saldo Kontrak:", f"{st.session_state.contract_balance:.5f} Token")
    
    elif extension == "dexchecker":
        st.header("TEA Assam Real-Time Transaction Trace")
        if not w3.is_connected():
            st.error("Gagal terhubung ke RPC TEA Assam.")
        else:
            with st.form(key="dexchecker_form"):
                num_blocks = st.slider(
                    "Jumlah Blok Terbaru:", 
                    min_value=1, 
                    max_value=20, 
                    value=5, 
                    key="num_blocks",
                    help="Pilih jumlah blok terbaru yang ingin ditampilkan."
                )
                submit_button = st.form_submit_button("Refresh Transactions")

                if submit_button:
                    with st.spinner("Mengambil data transaksi..."):
                        blocks = get_recent_blocks(w3, num_blocks=num_blocks)
                        abnormal_flags = detect_abnormal_transactions(blocks)
                        net = build_graph(blocks, abnormal_flags)
                        html_str = net.generate_html()
                        st.session_state.dexchecker_html = html_str

            if "dexchecker_html" in st.session_state:
                components.html(st.session_state.dexchecker_html, height=800, scrolling=False)
            else:
                st.info("Tekan 'Refresh Transactions' untuk menampilkan graf transaksi.")

    elif extension == "transfer":
        st.write("This is the screen for the Toggle extension.")
    elif extension == "transferbatch":
        st.write("This is the screen for the Functor extension.")
    elif extension == "addliqudity":
        st.write("This is the screen for the Dawn extension.")
else:
    for message in st.session_state.messages:
        avatar = "static/icons/user_icon.png" if message["role"] == "user" else "static/icons/assistant_icon.png"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(
                f'<div class="markdown-text scrollable-markdown">{message["content"]}</div>',
                unsafe_allow_html=True
            )
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="static/icons/user_icon.png"):
            st.markdown(
                f'<div class="markdown-text scrollable-markdown">{prompt}</div>',
                unsafe_allow_html=True
            )
        with st.spinner("Processing..."):
            response = get_ai_response(st.session_state.messages)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant", avatar="static/icons/assistant_icon.png"):
                    st.markdown(
                        f'<div class="markdown-text scrollable-markdown">{response}</div>',
                        unsafe_allow_html=True
                    )
                st.rerun()
