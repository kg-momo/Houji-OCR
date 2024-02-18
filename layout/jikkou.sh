#!/bin/bash

# 変数を定義
op_input="./input/Doho_19400515/"
op_output="./output/Doho_19400515/"
op_canvas="3500"

# Pythonスクリプトを呼び出し、変数を渡す
python craft_layout.py \
       "--input_dir" "$op_input" \
       "--out_dir" "$op_output" \
       "--canvas_size" "$op_canvas"
