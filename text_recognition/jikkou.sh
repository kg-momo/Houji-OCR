#!/bin/bash                                                                     

# 変数を定義                                                                    
FILE="Doho_19400515_3500_ver2"

cd deep-text-recognition-benchmark
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd ..
python text_recognition.py \
       --saved_model models/ndlenfixed64-mj0-synth1.pth \
       --character "〓$(cat models/mojilist_NDL.txt| tr -d '\n')" \
       --batch_max_length 300 --PAD \
       --batch_size 90 \
       --imgW 1200 \
       --db_path "./input_dir/$FILE/imgxml/" \
       --db_type xmlraw \
       --diff none \
       --xml "./output_dir/$FILE/" \
       --font_path "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc" \
       --Transformation None --FeatureExtraction ResNet \
       --SequenceModeling None --Prediction CTC
