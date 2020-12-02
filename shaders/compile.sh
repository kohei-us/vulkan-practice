#!/usr/bin/env sh

THIS_DIR=$(dirname $0)

cd $THIS_DIR
glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv
