{
  description = "pallas-cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ]; # Add other architectures as needed

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        
        # Override nixpkgs to allow unfree packages
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Shell environment
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            libusb1
            cmake
            opencv
            spdlog
            expected-lite
            gtest
            nlohmann_json
            curl
            curl.dev
            # Include explicit CUDA packages
            cudatoolkit
            cudaPackages.cudnn
            cudaPackages.cuda_cudart
            # cudaPackages_12_6.cudatoolkit
            # cudaPackages_12_6.cudnn
            (onnxruntime.override { 
              cudaSupport = true;
              cudaPackages = cudaPackages; #pkgs.cudaPackages_12_6;
            })            
            python312 # Python dependencies
            python3Packages.nanobind
            python3Packages.opencv4
            
            # We need GCC 12+ for C++23 support including std::expected
            gcc13
            
            # Include stdenv compiler as well
            stdenv.cc.cc
          ];

          shellHook = ''
            export PYTHONPATH=$PWD/build:$PYTHONPATH

            export CUDA_PATH=${pkgs.cudatoolkit}

            # Use GCC 13 for C++23 support
            export CC=${pkgs.gcc13}/bin/gcc
            export CXX=${pkgs.gcc13}/bin/g++
            
            # Add GCC 13 to PATH
            export PATH=${pkgs.gcc13}/bin:$PATH

            # Add necessary paths for dynamic linking
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
              "/run/opengl-driver" # Needed to find libGL.so
              pkgs.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.stdenv.cc.cc.lib  # Add system libstdc++
              pkgs.gcc13.cc.lib      # Add libstdc++ from gcc13
              pkgs.opencv            # Add OpenCV libs explicitly
              ]
            }:$LD_LIBRARY_PATH

            # Set LIBRARY_PATH to help the linker find the CUDA static libraries
            export LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
              pkgs.stdenv.cc.cc.lib  # Add system libstdc++
              pkgs.gcc13.cc.lib      # Add libstdc++ from gcc13
              pkgs.opencv            # Add OpenCV libs explicitly
              ]
            }:$LIBRARY_PATH

            # CUDA runtime environment setup for onnxruntime CUDA EP
            # export CUDA_PATH=${pkgs.cudaPackages_12_6.cudatoolkit}
            # export CUDNN_PATH=${pkgs.cudaPackages_12_6.cudnn}
            # export LD_LIBRARY_PATH=${pkgs.cudaPackages_12_6.cudatoolkit}/lib:${pkgs.cudaPackages_12_6.cudnn}/lib:$LD_LIBRARY_PATH
            # export NVIDIA_DRIVER_CAPABILITIES=compute,utility
            
            PS1="\[\033[0;32m\][nix-shell]\[\033[0m\] $PS1"
          '';
        };
      };
    };
}
