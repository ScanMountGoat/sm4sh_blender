# sm4sh_blender [![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ScanMountGoat/sm4sh_blender?include_prereleases)](https://github.com/ScanMountGoat/sm4sh_blender/releases/latest) [![wiki](https://img.shields.io/badge/wiki-guide-success)](https://github.com/scanmountgoat/sm4sh_blender/wiki)
A Blender addon for importing and exporting models and animations for Smash 4 for Wii U.

Report bugs or request new features in [issues](https://github.com/ScanMountGoat/sm4sh_blender/issues). Download the latest version from [releases](https://github.com/ScanMountGoat/sm4sh_blender/releases). Check the [wiki](https://github.com/ScanMountGoat/sm4sh_blender/wiki) for more usage information.

## Getting Started
* Download the latest version of the addon supported by your Blender version from [releases](https://github.com/ScanMountGoat/sm4sh_blender/releases).
* Install the .zip file in Blender using Edit > Preferences > Addons > Install...
* Enable the addon if it is not already enabled.
* Import any of the supported file types using the new menu options under File > Import.  

## Updating
Update the addon by reinstalling the latest version from [releases](https://github.com/ScanMountGoat/sm4sh_blender/releases). MacOS and Linux users can update without any additional steps.

> [!IMPORTANT]
> Windows users may need to disable the addon, restart Blender, remove the addon, and install the new version to update.

## Building
Clone the repository with `git clone https://github.com/ScanMountGoat/sm4sh_blender --recursive`. sm4sh_blender uses [sm4sh_model_py](https://github.com/ScanMountGoat/sm4sh_model_py) for simplifying the addon code and achieving better performance than pure Python addons. sm4sh_model_py must be compiled from source after [installing Rust](https://www.rust-lang.org/tools/install).
