{
  description = "pallas-cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      # Architectures
      systems = [ "x86_64-linux"]; # Mac for now; "x86_64-linux" "aarch64-linux" "aarch64-darwin" 
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        
        # Override nixpkgs to allow unfree packages
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };        

        # Shell
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            opencv
            ftxui
            spdlog
            expected-lite
            gtest
            onnxruntime
            python312 # start of py packages
            python3Packages.nanobind
            python3Packages.opencv4
          ];

          shellHook = ''
            export PYTHONPATH=$PWD/build:$PYTHONPATH
              
            # Change PS1 color when entering nix-shell
            PS1="\[\033[0;32m\][nix-shell]\[\033[0m\] $PS1"
          '';          

          # TODO if i need this shit 
        #   shellHook = ''
        #     # Save the original PS1

        #     ORIGINAL_PS1="$PS1"

        #     # Change PS1 color when entering nix-shell
        #     PS1="\[\033[0;32m\][nix-shell]\[\033[0m\] $PS1"

        #     # Force CMake to use the specified Python version
        #     # export CMAKE_PREFIX_PATH="${pkgs.python312}:$CMAKE_PREFIX_PATH"
        #     # export Python3_ROOT_DIR="${pkgs.python312}"
        #     # export Python3_EXECUTABLE="${pkgs.python312}/bin/python3"
        #     # export CMAKE_ARGS="-DPython3_ROOT_DIR=${pkgs.python312} -DPython3_EXECUTABLE=${pkgs.python312}/bin/python3"
        # '';
        };
      };
    };
}
