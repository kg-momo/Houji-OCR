ocr_dir=$(pwd)
docker run -it -v $ocr_dir:/code --gpus all --name socr_runner momokg/socr-py37
