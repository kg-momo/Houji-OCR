#!/bin/bash                                                                     


# 変数を定義
input_name="test" # Directory name in [./input]
output_name="test"
op_input="$(pwd)/input/$input_name/"
op_output="$(pwd)/output/$output_name/"
# 1000~5000程度で指定
# 大きすぎるとruntimeErrorになるので注意
op_canvas="3000"
# 新聞内一記事あたりの大きさによって変更するとよい
# デフォルトは40
op_segsize="50"




python ./layout/craft_layout.py \
       "--input_dir" "$op_input" \
       "--out_dir" "$op_output" \
       "--canvas_size" "$op_canvas" \
       "--segsize" "$op_segsize"

cd ./text_recognition/
cd ./deep-text-recognition-benchmark
export PYTHONPATH=$PYTHONPATH:$(pwd)


cd ..
python text_recognition.py \
       --saved_model models/ndlenfixed64-mj0-synth1.pth \
       --character "〓$(cat models/mojilist_NDL.txt| tr -d '\n')" \
       --batch_max_length 300 --PAD \
       --batch_size 60 \
       --imgW 1200 \
       --db_path "$op_output/imgxml/" \
       --db_type xmlraw \
       --diff none \
       --xml "$op_output/text/" \
       --font_path "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc" \
       --Transformation None --FeatureExtraction ResNet \
       --SequenceModeling None --Prediction CTC

cd ..
python ./output-utils/xml-to-image.py \
       --resize_dir "$op_output/imgxml/img/" \
       --xml_dir "$op_output/text/" \
       --out_dir "$op_output"

python ./output-utils/seglineXML-to-image.py \
       --dir_name "$op_output" \
       --out_dir "$op_output"
