#examples ...
"""
if __name__ == "__main__":
    deployer = ERC20Deployer(
        private_key="",
        initial_supply=1000000 * 10**18,
        show_abi=True,
        add_icon="https://github.com/diter89/tgemini/blob/master/static/tea.png"
    )
    
    results = deployer.run()
    for result in results:
        print(f"Step: {result['step']}")
        print(f"Status: {result['status']}")
        print(f"Message: {result.get('message', '')}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        print("---")
"""

from web3 import Web3
import solcx
import json
import os
from getpass import getpass

class ERC20Deployer:
    def __init__(self, private_key=None, initial_supply=None, show_abi=False, add_icon=None):
        """
        Initialize ERC20Deployer class for deploying ERC-20 contracts
        Args:
            private_key (str, optional): Sender's private key. If None, will prompt via input
            initial_supply (int, optional): Initial token supply in wei. If None, will prompt via input
            show_abi (bool, optional): Whether to show ABI in output. Default False
            add_icon (str, optional): URL/path to token icon image. If None, no icon added
        """
        self.rpc_url = "https://assam-rpc.tea.xyz"
        self.chain_id = 93384
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        self.private_key = private_key
        self.initial_supply = initial_supply
        self.show_abi = show_abi
        self.add_icon = add_icon
        
        # ERC-20 contract source code
        self.contract_source = '''pragma solidity ^0.8.28;

contract MyToken {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(uint256 initialSupply) {
        name = "interface";
        symbol = "interface";
        decimals = 18;
        totalSupply = initialSupply;
        balanceOf[msg.sender] = initialSupply;
        emit Transfer(address(0), msg.sender, initialSupply);
    }

    function transfer(address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid destination address");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool success) {
        require(spender != address(0), "Invalid spender address");
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid destination address");
        require(value <= balanceOf[from], "Insufficient balance");
        require(value <= allowance[from][msg.sender], "Allowance limit exceeded");
        
        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        emit Transfer(from, to, value);
        return true;
    }
}'''

    def _log_step(self, step, status, message=None, error=None, **kwargs):
        """Helper to create consistent log entries"""
        entry = {"step": step, "status": status}
        if message: entry["message"] = message
        if error: entry["error"] = error
        entry.update(kwargs)
        return entry

    def run(self):
        result_list = []
        
        # 1. Input validation
        if not self.private_key:
            self.private_key = getpass("Enter your private key: ")
            
        if not self.initial_supply:
            try:
                token_amount = int(input("Enter initial supply (whole tokens): "))
                self.initial_supply = token_amount * 10**18
            except ValueError:
                return [self._log_step("input_validation", "failed", 
                                     error="Invalid initial supply input. Please enter a number.")]

        # 2. Setup Solidity compiler
        try:
            solcx.install_solc('0.8.28')
            solcx.set_solc_version('0.8.28')
            result_list.append(self._log_step("setup_solc", "success", 
                                             message="Solidity 0.8.28 version set successfully"))
        except Exception as e:
            return [self._log_step("setup_solc", "failed", error=str(e))]

        # 3. Compile contract
        try:
            compiled_sol = solcx.compile_source(self.contract_source, output_values=['abi', 'bin'])
            contract_interface = compiled_sol['<stdin>:MyToken']
            self.abi = contract_interface['abi']
            self.bytecode = contract_interface['bin']
            
            result_list.append(self._log_step("compile_contract", "success", 
                                            message="Contract compiled successfully",
                                            abi=self.abi if self.show_abi else "Hidden",
                                            bytecode=self.bytecode[:50]+"..." if self.show_abi else "Hidden"))
        except Exception as e:
            return [self._log_step("compile_contract", "failed", error=str(e))]

        # 4. Save ABI
        os.makedirs("data/contracts", exist_ok=True)
        try:
            with open("data/contracts/abi.json", 'w') as f:
                json.dump(self.abi, f, indent=4)
            result_list.append(self._log_step("save_abi", "success", 
                                            message="ABI saved to data/contracts/abi.json"))
        except Exception as e:
            return [self._log_step("save_abi", "failed", error=str(e))]

        # 5. Check network connection
        if not self.web3.is_connected():
            return [self._log_step("connect_network", "failed", 
                                 error="Failed to connect to tea-assam network")]
        result_list.append(self._log_step("connect_network", "success", 
                                         message=f"Connected to tea-assam network (Chain ID: {self.chain_id})"))

        # 6. Setup account
        try:
            account = self.web3.eth.account.from_key(self.private_key)
            result_list.append(self._log_step("setup_account", "success", 
                                            message=f"Using account: {account.address}"))
        except Exception as e:
            return [self._log_step("setup_account", "failed", error=str(e))]

        # 7. Deploy contract
        MyToken = self.web3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        try:
            gas_estimate = MyToken.constructor(self.initial_supply).estimate_gas({'from': account.address})
            gas_price = self.web3.eth.gas_price * 2  # 2x gas price for priority
            nonce = self.web3.eth.get_transaction_count(account.address)
            
            construct_txn = MyToken.constructor(self.initial_supply).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'chainId': self.chain_id
            })
            
            result_list.append(self._log_step("estimate_gas", "success", 
                                            message=f"Gas estimate: {gas_estimate} | Gas price: {self.web3.from_wei(gas_price, 'gwei')} Gwei",
                                            nonce=nonce))
        except Exception as e:
            return [self._log_step("estimate_gas", "failed", error=str(e))]

        # 8. Sign transaction
        try:
            signed_txn = account.sign_transaction(construct_txn)
            result_list.append(self._log_step("sign_transaction", "success", 
                                            message="Transaction signed successfully"))
        except Exception as e:
            return [self._log_step("sign_transaction", "failed", error=str(e))]

        # 9. Send transaction
        try:
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            tx_hash_hex = self.web3.to_hex(tx_hash)
            result_list.append(self._log_step("send_transaction", "success", 
                                            message=f"Transaction hash: {tx_hash_hex}",
                                            tx_hash=tx_hash_hex))
        except Exception as e:
            return [self._log_step("send_transaction", "failed", error=str(e))]

        # 10. Wait for confirmation
        try:
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            result_list.append(self._log_step("wait_receipt", "success", 
                                            message=f"Contract deployed at: {tx_receipt.contractAddress}",
                                            contract_address=tx_receipt.contractAddress))
        except Exception as e:
            return [self._log_step("wait_receipt", "failed", error=str(e))]

        # 11. Verify contract
        try:
            contract = self.web3.eth.contract(address=tx_receipt.contractAddress, abi=self.abi)
            token_details = {
                "name": contract.functions.name().call(),
                "symbol": contract.functions.symbol().call(),
                "decimals": contract.functions.decimals().call(),
                "total_supply": contract.functions.totalSupply().call(),
                "balance": contract.functions.balanceOf(account.address).call()
            }
            
            result_list.append(self._log_step("verify_contract", "success", 
                                            message=f"Token verification success: {token_details['name']} ({token_details['symbol']})",
                                            token_details=token_details))
        except Exception as e:
            result_list.append(self._log_step("verify_contract", "failed", error=str(e)))

        # 12. Additional information
        if self.add_icon:
            result_list.append(self._log_step("add_icon", "info", 
                                            message=f"Token icon: {self.add_icon}",
                                            note="Upload to block explorer service after verification"))
        else:
            result_list.append(self._log_step("add_icon", "info", message="No icon added"))

        result_list.append(self._log_step("verification_info", "info", 
                                        message=f"Verify contract at https://assam.tea.xyz with:\n"
                                                f"- Address: {tx_receipt.contractAddress}\n"
                                                f"- Source code: See contract above\n"
                                                f"- Constructor parameters: {self.initial_supply}"))

        return result_list


