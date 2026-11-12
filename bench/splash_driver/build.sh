#!/bin/bash
# Build a tiny driver around SPLASH's own projection routine (interpolate3D_projection) for the benchmark. Needs gfortran and a SPLASH source tree
# (https://github.com/danieljprice/splash). SPLASH is GPL: only built here, nothing of it is copied into this repository.
#   bash bench/splash_driver/build.sh /path/to/splash  [outdir]      ->  outdir/splash_proj   (export SPLOSH_SPLASH_DRIVER=outdir/splash_proj)
set -e
SPLASH=${1:?usage: build.sh SPLASH_DIR [OUTDIR]}; OUT=${2:-bench_out/splash_driver}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; cd "$OUT"
F="-O3 -fopenmp -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -w"
S=$SPLASH/src
gfortran $F -c $S/asciiutils.f90 $S/timing.f90 $S/kernels.f90 $S/interpolate3D_projection.f90
gfortran $F "$HERE/driver.f90" asciiutils.o timing.o kernels.o interpolate3D_projection.o -o splash_proj
echo "built $PWD/splash_proj"
