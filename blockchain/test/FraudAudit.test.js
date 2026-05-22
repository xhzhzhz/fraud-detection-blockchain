const { expect }         = require("chai");
const { ethers }         = require("hardhat");
const { loadFixture }    = require("@nomicfoundation/hardhat-toolbox/network-helpers");

// Helper: buat bytes32 dari string
function toBytes32(str) {
    return ethers.keccak256(ethers.toUtf8Bytes(str));
}

// Helper: unix timestamp saat ini
function nowTs() {
    return Math.floor(Date.now() / 1000);
}

// ================================================================
// FIXTURE: deploy fresh contract untuk setiap test
// ================================================================

async function deployFraudAuditFixture() {
    const [owner, validator1, validator2, attacker] = await ethers.getSigners();

    const FraudAudit = await ethers.getContractFactory("FraudAudit");
    const contract   = await FraudAudit.deploy();
    await contract.waitForDeployment();

    return { contract, owner, validator1, validator2, attacker };
}

// Contoh data audit transaksi fraud untuk testing
const sampleRecord = {
    txHash:     toBytes32("TX-TEST-001-errorBalance-0.92"),
    fraudScore: 92,
    shapHash:   toBytes32("errorBalanceOrig=0.41|amountToBalance=0.28|type_CASH_OUT=0.19"),
    status:     2,   // FRAUD
    timestamp:  nowTs()
};

// ================================================================
// DEPLOYMENT
// ================================================================

describe("Deployment", function () {
    it("Owner harus terdaftar sebagai validator awal", async function () {
        const { contract, owner } = await loadFixture(deployFraudAuditFixture);
        expect(await contract.isValidator(owner.address)).to.be.true;
    });

    it("totalRecords harus 0 saat deploy", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const [total] = await contract.getAuditStats();
        expect(total).to.equal(0n);
    });
});

// ================================================================
// FUNGSIONAL : recordTransaction
// ================================================================

describe("recordTransaction — Fungsional", function () {

    it("TC-01: Validator berwenang dapat mencatat transaksi fraud", async function () {
        const { contract, owner } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await expect(
            contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp)
        ).to.emit(contract, "TransactionRecorded")
         .withArgs(txHash, fraudScore, status, owner.address, timestamp);

        expect(await contract.recordExists(txHash)).to.be.true;
        const [total, fraud] = await contract.getAuditStats();
        expect(total).to.equal(1n);
        expect(fraud).to.equal(1n);
    });

    it("TC-02: Mencatat transaksi suspicious (status=1)", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const ts = nowTs();

        await contract.recordTransaction(
            toBytes32("TX-SUSP-001"), 65, toBytes32("shap-susp"), 1, ts
        );
        const [total,, suspicious] = await contract.getAuditStats();
        expect(total).to.equal(1n);
        expect(suspicious).to.equal(1n);
    });

    it("TC-03: Mencatat transaksi legitimate (status=0)", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const ts = nowTs();

        await contract.recordTransaction(
            toBytes32("TX-LEGIT-001"), 10, toBytes32("shap-legit"), 0, ts
        );
        const [total,,, legitimate] = await contract.getAuditStats();
        expect(total).to.equal(1n);
        expect(legitimate).to.equal(1n);
    });

    it("TC-04: Duplikat txHash harus di-revert", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp);

        await expect(
            contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp + 1)
        ).to.be.revertedWith("FraudAudit: record already exists");
    });

    it("TC-05: fraudScore > 100 harus di-revert", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.recordTransaction(
                toBytes32("TX-INVALID"), 101,
                toBytes32("shap"), 2, nowTs()
            )
        ).to.be.revertedWith("FraudAudit: fraudScore must be 0-100");
    });

    it("TC-06: status > 2 harus di-revert", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.recordTransaction(
                toBytes32("TX-INVALID-STATUS"), 80,
                toBytes32("shap"), 3, nowTs()
            )
        ).to.be.revertedWith("FraudAudit: status must be 0, 1, or 2");
    });

    it("TC-07: txHash bytes32(0) harus di-revert", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.recordTransaction(
                ethers.ZeroHash, 80,
                toBytes32("shap"), 2, nowTs()
            )
        ).to.be.revertedWith("FraudAudit: txHash cannot be zero");
    });
});

// ================================================================
// FUNGSIONAL: getTransaction
// ================================================================

describe("getTransaction — Fungsional", function () {

    it("TC-08: Mengambil data catatan yang ada dengan benar", async function () {
        const { contract, owner } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp);

        const [rTxHash, rTs, rScore, rShap, rStatus, rValidator]
            = await contract.getTransaction(txHash);

        expect(rTxHash).to.equal(txHash);
        expect(rTs).to.equal(BigInt(timestamp));
        expect(rScore).to.equal(fraudScore);
        expect(rShap).to.equal(shapHash);
        expect(rStatus).to.equal(status);
        expect(rValidator).to.equal(owner.address);
    });

    it("TC-09: getTransaction pada record tidak ada harus revert", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.getTransaction(toBytes32("TIDAK-ADA"))
        ).to.be.revertedWith("FraudAudit: record does not exist");
    });
});

// ================================================================
// FUNGSIONAL: verifyIntegrity
// ================================================================

describe("verifyIntegrity — Fungsional", function () {

    it("TC-10: Verifikasi dengan data yang cocok harus return true", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp);

        const result = await contract.verifyIntegrity(
            txHash, fraudScore, shapHash, status, timestamp
        );
        expect(result).to.be.true;
    });

    it("TC-11: Verifikasi dengan fraudScore berbeda harus return false", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp);

        const result = await contract.verifyIntegrity(
            txHash, 50, shapHash, status, timestamp
        );
        expect(result).to.be.false;
    });

    it("TC-12: Verifikasi dengan shapHash berbeda harus return false", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        await contract.recordTransaction(txHash, fraudScore, shapHash, status, timestamp);

        const result = await contract.verifyIntegrity(
            txHash, fraudScore, toBytes32("SHAP-BERBEDA"), status, timestamp
        );
        expect(result).to.be.false;
    });

    it("TC-13: Verifikasi record tidak ada harus return false", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        const result = await contract.verifyIntegrity(
            toBytes32("TIDAK-ADA"), fraudScore, shapHash, status, timestamp
        );
        expect(result).to.be.false;
    });
});

// ================================================================
// VALIDATOR MANAGEMENT
// ================================================================

describe("Validator Management", function () {

    it("TC-14: Owner dapat menambahkan validator baru", async function () {
        const { contract, validator1 } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.addValidator(validator1.address)
        ).to.emit(contract, "ValidatorAdded")
         .withArgs(validator1.address);

        expect(await contract.isValidator(validator1.address)).to.be.true;
    });

    it("TC-15: Validator baru dapat mencatat transaksi", async function () {
        const { contract, validator1 } = await loadFixture(deployFraudAuditFixture);

        await contract.addValidator(validator1.address);
        const ts = nowTs();

        await expect(
            contract.connect(validator1).recordTransaction(
                toBytes32("TX-V1-001"), 88,
                toBytes32("shap-v1"), 2, ts
            )
        ).to.emit(contract, "TransactionRecorded");
    });

    it("TC-16: Owner dapat menghapus validator", async function () {
        const { contract, validator1 } = await loadFixture(deployFraudAuditFixture);

        await contract.addValidator(validator1.address);
        await expect(
            contract.removeValidator(validator1.address)
        ).to.emit(contract, "ValidatorRemoved")
         .withArgs(validator1.address);

        expect(await contract.isValidator(validator1.address)).to.be.false;
    });

    it("TC-17: Owner tidak bisa dihapus dari validator", async function () {
        const { contract, owner } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.removeValidator(owner.address)
        ).to.be.revertedWith("FraudAudit: cannot remove owner");
    });

    it("TC-18: Non-owner tidak bisa menambahkan validator", async function () {
        const { contract, attacker, validator1 }
            = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.connect(attacker).addValidator(validator1.address)
        ).to.be.revertedWithCustomError(contract, "OwnableUnauthorizedAccount");
    });
});

// ================================================================
// SECURITY TESTING
// ================================================================

describe("Security — Access Control", function () {

    it("TC-19: Attacker (bukan validator) tidak bisa mencatat", async function () {
        const { contract, attacker } = await loadFixture(deployFraudAuditFixture);

        await expect(
            contract.connect(attacker).recordTransaction(
                toBytes32("TX-ATTACK"), 99,
                toBytes32("shap-attack"), 2, nowTs()
            )
        ).to.be.revertedWith("FraudAudit: caller is not a registered validator");
    });

    it("TC-20: Validator yang dihapus tidak bisa mencatat lagi", async function () {
        const { contract, validator1 } = await loadFixture(deployFraudAuditFixture);

        await contract.addValidator(validator1.address);
        await contract.removeValidator(validator1.address);

        await expect(
            contract.connect(validator1).recordTransaction(
                toBytes32("TX-REVOKED"), 88,
                toBytes32("shap"), 2, nowTs()
            )
        ).to.be.revertedWith("FraudAudit: caller is not a registered validator");
    });
});

describe("Security — Integer Overflow", function () {
    it("TC-21: fraudScore tepat 100 harus berhasil", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        await expect(
            contract.recordTransaction(
                toBytes32("TX-MAX-SCORE"), 100,
                toBytes32("shap"), 2, nowTs()
            )
        ).to.not.be.reverted;
    });

    it("TC-22: fraudScore 0 (legitimate) harus berhasil", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        await expect(
            contract.recordTransaction(
                toBytes32("TX-MIN-SCORE"), 0,
                toBytes32("shap"), 0, nowTs()
            )
        ).to.not.be.reverted;
    });
});

// ================================================================
// GAS ESTIMATION
// Estimasi gas dicatat untuk analisis biaya 
// ================================================================

describe("Gas Estimation", function () {
    it("Gas untuk recordTransaction dicatat ke konsol", async function () {
        const { contract } = await loadFixture(deployFraudAuditFixture);
        const { txHash, fraudScore, shapHash, status, timestamp } = sampleRecord;

        const tx  = await contract.recordTransaction(
            txHash, fraudScore, shapHash, status, timestamp
        );
        const rcpt = await tx.wait();
        const gas  = rcpt.gasUsed;

        console.log(`\n  Gas recordTransaction: ${gas.toString()} units`);
        // Target: < 250,000 gas per transaksi
        expect(gas).to.be.lessThan(250000n);
    });
});