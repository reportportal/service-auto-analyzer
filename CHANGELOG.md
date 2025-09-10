# Changelog
## [Unreleased]

## [5.14.5]
### Added
- Newly trained suggestion model, by @HardNorth
### Fixed
- Text similarity calculation doesn't use IDF now, to be less confusing for users, by @HardNorth

## [5.14.4]
### Added
- `_is_analyzed_manually` feature in `BoostingFeaturizer` class, by @HardNorth
- Newly trained models, by @HardNorth
### Fixed
- Text similarity calculation for cases when a text contains only stop words, by @HardNorth
### Changed
- Refactor request calling in `esclient` class, by @HardNorth
- Text similarity calculation logic replaced with less sophisticated one, by @HardNorth
- Comparison logic of `potential_status_codes` and `only_numbers` fields, by @HardNorth
### Removed
- `potential_status_codes` field from minimal threshold validation as it unreliable, by @HardNorth
- Similarity weights model, by @HardNorth

## [5.14.3]
### Fixed
* Base URL generation with credentials for ES/OS healthcheck and indices endpoints, by @HardNorth in https://github.com/reportportal/service-auto-analyzer/pull/216

## [5.14.2]
### Added
- Ability to use different ML models for Suggestions, by @HardNorth
- Ability to switch ML model for Suggestions with `ML_MODEL_FOR_SUGGESTIONS` environment variable, by @HardNorth
### Changed
- Handling of AMQP messages now moved to separate processes: "train" (only for model training) and "main" (for all other messages), by @HardNorth
- Handling of AMQP messages now performed in two separate queues: `train` and `all`, by @HardNorth
- Handling of AMQP messages now performed in in strict order, by @HardNorth
- Analyzer exchange type is changed to `fanout`, by @HardNorth
- Many logging messages were refactored and improved, by @HardNorth
- Choosing custom/global ML model now performed by Launch ID hash only, by @HardNorth
### Fixed
- 11 Sonar issues, by @HardNorth
### Removed
- `analyzer-train` service, as it is no longer needed, by @HardNorth

## [5.14.1]
### Added
- `ES_BOOST_MA` environment variable to boost manually analyzed test cases in ES, by @HardNorth
### Changed
- AMQP exchange declaration now performs on every connection, to avoid issues with exchange not being declared on RabbitMQ restarts, by @HardNorth
- `ES_BOOST_AA` environment variable default value changed to `0.0`, by @HardNorth
- `ES_TIME_WEIGHT_DECAY` environment variable default value changed to `0.999`, by @HardNorth
- `flask-cors` dependency updated to `6.0.0` to address vulnerabilities, by @HardNorth
### Fixed
- 18 Sonar issues, by @HardNorth

## [5.14.0]
### Added:
- `AMQP_MAX_RETRY_TIME`, `AMQP_INITIAL_RETRY_INTERVAL`, `AMQP_BACKOFF_FACTOR`, `AMQP_HEARTBEAT_INTERVAL` environment variables to configure AMQP client, by @HardNorth
- `ANALYZER_ENABLE_MEMORY_DUMP` environment variable to print memory dump on healthcheck calls for debugging purpose, by @HardNorth
### Changed
- AMQP client was rewritten to better handle connection issues, by @HardNorth
- `uWSGI` version updated to `2.0.29`, by @HardNorth
- Improved the way of URL and path extraction on data preparation stage, by @HardNorth
### Fixed
- 4 Sonar issues, by @HardNorth

## [5.13.2]
### Fixed
- Issue [#196](https://github.com/reportportal/service-auto-analyzer/issues/196): Analyzer picks all path tokens as bucket name except the first one, by @HardNorth

## [5.13.1]
### Fixed
- Data storing permissions and path issues for Filesystem binary storage type, by @HardNorth

## [5.13.0]
### Added:
- Issue [#149](https://github.com/reportportal/service-auto-analyzer/issues/149): Support of single bucket binary storage, by @HardNorth
### Changed
- Dockerfile updated to use `ubi9` as base image, by @raikbitters
- Dependency versions updated to address vulnerabilities, by @HardNorth
### Fixed
- Unique error view, by @HardNorth

## [5.12.0]
### Added:
- Message-through logging with Correlation ID, to ease debugging and understanding of logs, by @HardNorth

### Updated:
- Refactoring: data-preparation logic joined and put into common place, by @HardNorth
- Refactoring: model train logic standardised and prepared for future join, by @HardNorth
- Lots of type annotations added, by @HardNorth

### Fixed
- Re-train logic. Custom re-trained models for big enough projects do not affect negatively auto-analysis now, by @HardNorth
- CVE addressed: CVE-2023-45853, CVE-2023-6246, CVE-2023-6779, CVE-2023-6780, CVE-2023-49468, CVE-2023-49467, CVE-2023-49465

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
