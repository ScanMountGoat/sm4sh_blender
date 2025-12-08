# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 0.3.1 - 2025-12-08
### Fixed
* Fixed an issue where exported nud models would not have correctly normalized vertex weights in some cases.

## 0.3.0 - 2025-12-03
### Added
* Added an import option for recreating in game shaders with Blender's material nodes. This option is still experimental and disabled by default.

### Changed
* Improved the error message when importing a nud model without the required model.vbn file.

### Fixed
* Fixed an issue where importing would fail for compressed model.nut files. 

## 0.2.1 - 2025-11-12
### Fixed
* Fixed a compatibility issue preventing animation import in Blender 5.0.

## 0.2.0 - 2025-08-30
### Changed
* Improved error messages when importing or exporting a nud model without the required vbn file.
* Improved generated material nodes to include basic assignments for color and normal textures.

### Fixed
* Fixed an issue where animations would import with incorrect bone rotations in some cases.

## 0.1.0 - 2025-08-07
First public release!
