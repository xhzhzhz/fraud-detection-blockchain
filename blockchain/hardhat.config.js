require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: "../.env" });

const ALCHEMY_URL    = process.env.ALCHEMY_SEPOLIA_URL  || "";
const DEPLOYER_KEY   = process.env.DEPLOYER_PRIVATE_KEY || "";

if (!ALCHEMY_URL && process.env.HARDHAT_NETWORK === "sepolia") {
  throw new Error("ALCHEMY_SEPOLIA_URL tidak diisi di .env");
}

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,   // Optimasi untuk frekuensi pemanggilan sedang
      },
    },
  },
  networks: {
    // Jaringan lokal untuk unit testing (otomatis oleh Hardhat)
    hardhat: {
      chainId: 31337,
    },
    // Sepolia testnet untuk deployment proof-of-concept
    sepolia: {
      url:      ALCHEMY_URL,
      accounts: DEPLOYER_KEY ? [DEPLOYER_KEY] : [],
      chainId:  11155111,
      gasPrice: "auto",
    },
  },
  gasReporter: {
    enabled:  true,
    currency: "USD",
    coinmarketcap: process.env.COINMARKETCAP_API_KEY || "",
    outputFile: "../reports/gas_report.txt",
    noColors:   true,
  },
  paths: {
    sources:   "./contracts",
    tests:     "./test",
    cache:     "./cache",
    artifacts: "./artifacts",
  },
};