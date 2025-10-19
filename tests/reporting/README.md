# Reporting Module Tests

This directory contains comprehensive test suites for the `reporting_bids.py` module.

## Test Files

### Unit Tests
**`test_reporting_bids.py`** (30 tests)

Covers all individual functions:
- ID normalization (subject and session)
- Identifier building
- Options setup
- Confidence score computation
- Anatomical annotation
- Result formatting
- Filename generation
- File saving (default, in-place, custom path)
- BIDS layout queries
- Integration tests
- Data consistency

### Integration Tests
**`test_reporting_bids_integration.py`** (9 tests)

End-to-end testing:
- Full pipeline processing with mock data
- ID and session normalization consistency
- Result reproducibility
- Error handling (missing files, no clusters)
- Backwards compatibility (filename format, CSV columns)

## Running Tests

### Run All Tests
```bash
# From project root
pytest tests/reporting/ -v
```

### Run Specific Test File
```bash
# Unit tests only
pytest tests/reporting/test_reporting_bids.py -v

# Integration tests only
pytest tests/reporting/test_reporting_bids_integration.py -v
```

### Run Specific Test Class
```bash
pytest tests/reporting/test_reporting_bids.py::TestSaveResults -v
```

### Run Specific Test
```bash
pytest tests/reporting/test_reporting_bids.py::TestSaveResults::test_save_results_in_place_flag -v
```

### Run with Short Traceback
```bash
pytest tests/reporting/ -v --tb=short
```

## Test Coverage

Total: **39 tests** (30 unit + 9 integration)

### Test Classes

**Unit Tests:**
- `TestIDNormalization` (5 tests) - Subject and session ID normalization
- `TestIdentifierBuilder` (2 tests) - Full identifier construction
- `TestSetupOptions` (2 tests) - Options dictionary setup
- `TestConfidenceScores` (3 tests) - Confidence score computation
- `TestAnatomicalAnnotation` (2 tests) - Harvard-Oxford atlas labeling
- `TestResultFormatting` (3 tests) - Result DataFrame formatting
- `TestFilenameGeneration` (2 tests) - BIDS-compliant filename generation
- `TestSaveResults` (3 tests) - File saving with different output modes
- `TestFindPredictionFiles` (4 tests) - BIDS layout file discovery
- `TestIntegration` (1 test) - Cluster extraction and processing
- `TestDataConsistency` (3 tests) - Deterministic and idempotent operations

**Integration Tests:**
- `TestRealDataProcessing` (3 tests) - Full pipeline and normalization
- `TestResultReproducibility` (2 tests) - Multiple runs produce same results
- `TestErrorHandling` (2 tests) - Missing files and empty results
- `TestBackwardsCompatibility` (2 tests) - Filename and column format preservation

## Test Status

All tests passing ✅

Last run: October 18, 2025
- Unit tests: 30/30 PASSED
- Integration tests: 9/9 PASSED

## Dependencies

Tests require:
- pytest >= 8.3.4
- numpy
- pandas
- nibabel
- pybids (bids)
- unittest.mock (standard library)

## Import Path

Tests import from `app/utils/reporting_bids.py` using:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app' / 'utils'))
```

## Documentation

For detailed test documentation, see:
- `../../docs/reporting/TESTING.md` - Comprehensive test documentation
- `../../docs/reporting/API_REFERENCE.md` - Function API reference

## Module Location

Source code: `../../app/utils/reporting_bids.py`
