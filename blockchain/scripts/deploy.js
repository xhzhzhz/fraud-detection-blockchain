const { ethers, network } = require("hardhat");
const fs   = require("fs");
const path = require("path");

async function main() {
    console.log(`\nDeploying FraudAudit ke jaringan: ${network.name}`);
    console.log("=".repeat(55));

    const [deployer] = await ethers.getSigners();
    const balance    = await ethers.provider.getBalance(deployer.address);

    console.log(`Deployer address : ${deployer.address}`);
    console.log(`Saldo deployer   : ${ethers.formatEther(balance)} ETH`);

    if (balance < ethers.parseEther("0.01")) {
        throw new Error("Saldo tidak cukup! Minimal 0.01 ETH Sepolia diperlukan.");
    }

    // Deploy
    console.log("\nMen-deploy kontrak...");
    const FraudAudit = await ethers.getContractFactory("FraudAudit");
    const contract   = await FraudAudit.deploy();
    await contract.waitForDeployment();

    const contractAddress = await contract.getAddress();
    const deployTx        = contract.deploymentTransaction();
    const receipt         = await deployTx.wait();

    console.log(`\nKontrak berhasil di-deploy!`);
    console.log(`Contract address : ${contractAddress}`);
    console.log(`Transaction hash : ${deployTx.hash}`);
    console.log(`Block number     : ${receipt.blockNumber}`);
    console.log(`Gas digunakan    : ${receipt.gasUsed.toString()}`);

    // Verifikasi deployment 
    const [total] = await contract.getAuditStats();
    console.log(`\nVerifikasi: totalRecords = ${total.toString()} (diharapkan: 0) ✓`);

    const ownerIsValidator = await contract.isValidator(deployer.address);
    console.log(`Verifikasi: deployer adalah validator = ${ownerIsValidator} ✓`);

    // Simpan info deployment ke JSON
    const deploymentInfo = {
        network:         network.name,
        contractAddress: contractAddress,
        deployerAddress: deployer.address,
        txHash:          deployTx.hash,
        blockNumber:     receipt.blockNumber,
        gasUsed:         receipt.gasUsed.toString(),
        deployedAt:      new Date().toISOString(),
        explorerUrl:     network.name === "sepolia"
            ? `https://sepolia.etherscan.io/address/${contractAddress}`
            : "local"
    };

    // Simpan ke evaluation folder
    const outPath = path.join(__dirname, "../../models/evaluation/deployment_info.json");
    fs.writeFileSync(outPath, JSON.stringify(deploymentInfo, null, 2));
    console.log(`\nInfo deployment disimpan: ${outPath}`);

    // Update .env dengan CONTRACT_ADDRESS
    const envPath    = path.join(__dirname, "../../.env");
    let   envContent = fs.readFileSync(envPath, "utf-8");

    if (envContent.includes("CONTRACT_ADDRESS=")) {
        envContent = envContent.replace(
            /CONTRACT_ADDRESS=.*/,
            `CONTRACT_ADDRESS=${contractAddress}`
        );
    } else {
        envContent += `\nCONTRACT_ADDRESS=${contractAddress}`;
    }

    if (envContent.includes("VALIDATOR_ADDRESS=")) {
        envContent = envContent.replace(
            /VALIDATOR_ADDRESS=.*/,
            `VALIDATOR_ADDRESS=${deployer.address}`
        );
    } else {
        envContent += `\nVALIDATOR_ADDRESS=${deployer.address}`;
    }

    fs.writeFileSync(envPath, envContent);
    console.log(".env diperbarui dengan CONTRACT_ADDRESS dan VALIDATOR_ADDRESS ✓");

    if (network.name === "sepolia") {
        console.log(`\nVerifikasi di Etherscan:`);
        console.log(`  ${deploymentInfo.explorerUrl}`);
    }

    console.log("\nDeployment selesai.");
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});