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
          overlays = [
            (final: prev: {
              onnxruntime_1_20_2 = final.callPackage "${./.}/onnxruntime-1.20.2" { };
            })
          ];
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
            cudaPackages_12_4.cudatoolkit
            cudaPackages_12_4.cudnn
            (onnxruntime_1_20_2.override { 
              cudaSupport = true;
              cudaPackages = pkgs.cudaPackages_12_4;
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

            # export CUDA_PATH=${pkgs.cudaPackages_12_4.cudatoolkit}

            # Use GCC 13 for C++23 support
            export CC=${pkgs.gcc13}/bin/gcc
            export CXX=${pkgs.gcc13}/bin/g++
            
            # Add GCC 13 to PATH
            export PATH=${pkgs.gcc13}/bin:$PATH

            # Add necessary paths for dynamic linking
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
              "/run/opengl-driver" # Needed to find libGL.so
              pkgs.cudaPackages_12_4.cudatoolkit
              pkgs.cudaPackages_12_4.cudnn
              pkgs.stdenv.cc.cc.lib  # Add system libstdc++
              pkgs.gcc13.cc.lib      # Add libstdc++ from gcc13
              pkgs.opencv            # Add OpenCV libs explicitly
              ]
            }:$LD_LIBRARY_PATH

            # Set LIBRARY_PATH to help the linker find the CUDA static libraries
            export LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages_12_4.cudatoolkit
              pkgs.stdenv.cc.cc.lib  # Add system libstdc++
              pkgs.gcc13.cc.lib      # Add libstdc++ from gcc13
              pkgs.opencv            # Add OpenCV libs explicitly
              ]
            }:$LIBRARY_PATH

            # CUDA runtime environment setup for onnxruntime CUDA EP
            export CUDA_PATH=${pkgs.cudaPackages_12_4.cudatoolkit}
            
            # Set CUDNN_PATH to the specific store path where libcudnn.so is located
            export CUDNN_PATH=/nix/store/wwv1v940drvc3788bhhigx6n0h14h1hr-cudnn-9.7.1.26-lib
            
            # Make specific library paths visible for debugging
            echo "CUDA libraries at: ${pkgs.cudaPackages_12_4.cudatoolkit}/lib"
            echo "CUDNN libraries at: ${pkgs.cudaPackages_12_4.cudnn}/lib"
            
            # Add the specific path to where libcudnn.so was found
            echo "Adding direct path to libcudnn.so: /nix/store/wwv1v940drvc3788bhhigx6n0h14h1hr-cudnn-9.7.1.26-lib/lib"
            export LD_LIBRARY_PATH=/nix/store/wwv1v940drvc3788bhhigx6n0h14h1hr-cudnn-9.7.1.26-lib/lib:${pkgs.cudaPackages_12_4.cudnn}/lib:$LD_LIBRARY_PATH
            
            # Fix the ONNX Runtime CUDA segfault issue
            export ORT_DISABLE_STACK_TRACE=1
            export ORT_CUDA_USE_ARENA=1
            export CUDA_VISIBLE_DEVICES=0
            
            # Add GCC 14 library path to fix CXXABI_1.3.15 issue
            # export LD_LIBRARY_PATH=${pkgs.gcc14.cc.lib}/lib:$LD_LIBRARY_PATH
            
            export NVIDIA_DRIVER_CAPABILITIES=compute,utility
            
            PS1="\[\033[0;32m\][nix-shell]\[\033[0m\] $PS1"
          '';
        };
      };
    };
}
