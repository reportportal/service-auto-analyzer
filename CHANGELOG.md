# Changelog

## [5.11.0]
### Added
- `MINIO_USE_TLS` environment variable to address [Issue #136](https://github.com/reportportal/service-auto-analyzer/issues/136), by @HardNorth
- `CURRENT_AND_THE_SAME_NAME` and `PREVIOUS_LAUNCH` analyze options handling, by @HardNorth
- `launch_number` data parameter indexing and usage, by @HardNorth
- `previousLaunchId` request parameter handling, by @HardNorth
### Changed
- Suggestions now do not hide other launches, which do not suite Analyzer configuration, just put them lower in priority, by @HardNorth


## [5.10.0]
### Changed
- Global structure update
- Deprecated `unique_id` field changed to `test_case_hash`, by @HardNorth
- `ES_BOOST_UNIQUE_ID` environment variable renamed to `ES_BOOST_TEST_CASE_HASH`, by @HardNorth

## [5.7.2]
