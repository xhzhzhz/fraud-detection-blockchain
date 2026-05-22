// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract FraudAudit is Ownable, ReentrancyGuard {

    // ================================================================
    // STRUCTS
    // ================================================================

    /**
     * @dev Catatan audit per transaksi.
     *      fraudScore: probabilitas fraud diskala 0-100 (uint8 hemat gas).
     *      status: 0=legitimate, 1=suspicious, 2=fraud
     *      shapHash: SHA-256 dari top-3 SHAP values (dihasilkan off-chain).
     *      transactionHash: SHA-256 dari seluruh isi file audit JSON.
     */
    struct AuditRecord {
        bytes32 transactionHash;
        uint256 timestamp;
        uint8   fraudScore;
        bytes32 shapHash;
        uint8   status;
        address validator;
        bool    exists;
    }

    // ================================================================
    // STATE VARIABLES
    // ================================================================

    // transactionId (bytes32) : AuditRecord
    mapping(bytes32 => AuditRecord) private auditRecords;

    // Daftar validator yang berwenang memanggil recordTransaction
    mapping(address => bool) private validators;

    // Counter dan statistik
    uint256 public totalRecords;
    uint256 public totalFraud;
    uint256 public totalSuspicious;
    uint256 public totalLegitimate;

    // Untuk iterasi stats (timestamp first/last)
    uint256 public firstRecordTimestamp;
    uint256 public lastRecordTimestamp;

    // ================================================================
    // EVENTS
    // ================================================================

    event TransactionRecorded(
        bytes32 indexed transactionHash,
        uint8   fraudScore,
        uint8   status,
        address indexed validator,
        uint256 timestamp
    );

    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);

    // ================================================================
    // MODIFIERS
    // ================================================================

    modifier onlyValidator() {
        require(
            validators[msg.sender],
            "FraudAudit: caller is not a registered validator"
        );
        _;
    }

    modifier recordMustExist(bytes32 txHash) {
        require(
            auditRecords[txHash].exists,
            "FraudAudit: record does not exist"
        );
        _;
    }

    // ================================================================
    // CONSTRUCTOR
    // ================================================================

    constructor() Ownable(msg.sender) {
        // Owner otomatis terdaftar sebagai validator pertama
        validators[msg.sender] = true;
        emit ValidatorAdded(msg.sender);
    }

    // ================================================================
    // FUNGSI UTAMA
    // ================================================================

    /**
     * @notice Mencatat hasil deteksi fraud ke blockchain.
     * @param txHash        SHA-256 dari isi file audit JSON (bytes32)
     * @param fraudScore    Probabilitas fraud diskala 0-100
     * @param shapHash      SHA-256 dari top-3 SHAP feature values
     * @param status        0=legitimate, 1=suspicious, 2=fraud
     * @param recordTimestamp Unix timestamp saat transaksi dideteksi
     *
     * Checks-Effects-Interactions pattern diterapkan untuk
     * mencegah reentrancy attack.
     */
    function recordTransaction(
        bytes32 txHash,
        uint8   fraudScore,
        bytes32 shapHash,
        uint8   status,
        uint256 recordTimestamp
    )
        external
        nonReentrant
        onlyValidator
    {
        // CHECK: validasi input
        require(txHash != bytes32(0),   "FraudAudit: txHash cannot be zero");
        require(fraudScore <= 100,      "FraudAudit: fraudScore must be 0-100");
        require(status <= 2,            "FraudAudit: status must be 0, 1, or 2");
        require(
            !auditRecords[txHash].exists,
            "FraudAudit: record already exists"
        );
        require(
            recordTimestamp > 0 && recordTimestamp <= block.timestamp + 300,
            "FraudAudit: invalid timestamp"
        );

        // EFFECT: simpan ke state
        auditRecords[txHash] = AuditRecord({
            transactionHash: txHash,
            timestamp:       recordTimestamp,
            fraudScore:      fraudScore,
            shapHash:        shapHash,
            status:          status,
            validator:       msg.sender,
            exists:          true
        });

        // Update statistik
        totalRecords++;
        if (status == 2)      totalFraud++;
        else if (status == 1) totalSuspicious++;
        else                  totalLegitimate++;

        if (firstRecordTimestamp == 0) {
            firstRecordTimestamp = recordTimestamp;
        }
        lastRecordTimestamp = recordTimestamp;

        // INTERACTION: emit event
        emit TransactionRecorded(
            txHash,
            fraudScore,
            status,
            msg.sender,
            recordTimestamp
        );
    }

    /**
     * @notice Mengambil catatan audit berdasarkan transaction hash.
     * @dev Fungsi view : tidak mengonsumsi gas saat dipanggil off-chain.
     */
    function getTransaction(bytes32 txHash)
        external
        view
        recordMustExist(txHash)
        returns (
            bytes32 transactionHash,
            uint256 timestamp,
            uint8   fraudScore,
            bytes32 shapHash,
            uint8   status,
            address validator
        )
    {
        AuditRecord storage record = auditRecords[txHash];
        return (
            record.transactionHash,
            record.timestamp,
            record.fraudScore,
            record.shapHash,
            record.status,
            record.validator
        );
    }

    /**
     * @notice Memverifikasi integritas catatan audit.
     * @param txHash        Hash yang akan diverifikasi
     * @param fraudScore    Nilai fraud score untuk dicocokkan
     * @param shapHash      Hash SHAP untuk dicocokkan
     * @param status        Status untuk dicocokkan
     * @param recordTimestamp Timestamp untuk dicocokkan
     * @return true jika semua parameter cocok dengan catatan tersimpan
     */
    function verifyIntegrity(
        bytes32 txHash,
        uint8   fraudScore,
        bytes32 shapHash,
        uint8   status,
        uint256 recordTimestamp
    )
        external
        view
        returns (bool)
    {
        if (!auditRecords[txHash].exists) return false;

        AuditRecord storage record = auditRecords[txHash];
        return (
            record.fraudScore  == fraudScore  &&
            record.shapHash    == shapHash    &&
            record.status      == status      &&
            record.timestamp   == recordTimestamp
        );
    }

    /**
     * @notice Memeriksa apakah record untuk txHash sudah ada.
     */
    function recordExists(bytes32 txHash) external view returns (bool) {
        return auditRecords[txHash].exists;
    }

    /**
     * @notice Statistik agregat audit trail.
     */
    function getAuditStats()
        external
        view
        returns (
            uint256 total,
            uint256 fraud,
            uint256 suspicious,
            uint256 legitimate,
            uint256 firstTs,
            uint256 lastTs
        )
    {
        return (
            totalRecords,
            totalFraud,
            totalSuspicious,
            totalLegitimate,
            firstRecordTimestamp,
            lastRecordTimestamp
        );
    }

    // ================================================================
    // VALIDATOR MANAGEMENT
    // ================================================================

    function addValidator(address account) external onlyOwner {
        require(account != address(0), "FraudAudit: zero address");
        require(!validators[account],  "FraudAudit: already a validator");
        validators[account] = true;
        emit ValidatorAdded(account);
    }

    function removeValidator(address account) external onlyOwner {
        require(validators[account], "FraudAudit: not a validator");
        // Owner tidak bisa dihapus dari validator
        require(account != owner(),  "FraudAudit: cannot remove owner");
        validators[account] = false;
        emit ValidatorRemoved(account);
    }

    function isValidator(address account) external view returns (bool) {
        return validators[account];
    }
}