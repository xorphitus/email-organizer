{
  description = "email-organizer dev shell (uv + CUDA-visible env for pip-installed torch)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # Shared libs that manylinux wheels (torch, bitsandbytes, …) expect to find.
      runtimeLibs = with pkgs; [ stdenv.cc.cc.lib zlib zstd libGL glib ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.uv pkgs.python312 ];
        env = {
          UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          UV_PYTHON_DOWNLOADS = "never";
        };
        shellHook = ''
          # /run/opengl-driver/lib provides libcuda.so from the NixOS NVIDIA driver.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          # Triton locates libcuda via `/sbin/ldconfig -p`, which doesn't exist on NixOS.
          export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
        '';
      };
    };
}
