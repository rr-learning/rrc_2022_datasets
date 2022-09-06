# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Print warnings if the step-function of SimTriFingerCubeEnv is called at a too low
  rate.  This way the users get some indication if their policy is taking too much time.

### Fixed
- Count environment steps instead of robot steps to determine end of the episode.  This
  fixes an issue that calling step() at a too low rate could cause in too many
  environment steps to be executed.


## [1.1.0] - 2022-08-02
### Added
- Add datasets for the real-robot stage

## [1.0.1] - 2022-07-04
### Changed
- Do not sleep in each step if visualisation is disabled.

### Fixed
- Fix URL for downloading the pre-stage lifting dataset


## [1.0.0] - 2022-06-30

First release


[Unreleased]: https://github.com/rr-learning/rrc_2022_datasets/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/rr-learning/rrc_2022_datasets/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/rr-learning/rrc_2022_datasets/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/rr-learning/rrc_2022_datasets/releases/tag/v1.0.0
