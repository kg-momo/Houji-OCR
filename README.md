# 邦字新聞に対応したOCRシステム

作成済みDockerイメージを利用して、実行が可能です。

## 環境構築

### 1. 動作確認済みの環境
【Linux】  

OS: Ubuntu 22.04.3  

GPU:  NVIDIA AD102 [GeForce RTX 4090]  

NVIDIA Driver 525.125.06  

【Windows】

OS: Windows 11 22H2 (Windows10 ver.1903 以上が必要)

GPU: NVIDIA TU106 [GeForce RTX 2060 SUPER] 

NVIDIA Driver: 535.129.03

### 2. Dockerの利用

以下のリンクに従ってDockerのインストールを行ってください。  
https://docs.docker.com/engine/install/

#### Windowsを利用する場合
【WSL2によるDocker環境構築】

【Nvidia CUDA Toolkitの確認】


## 実行
### 1. 必要ファイルのダウンロード
文字認識モジュール用ファイルのダウンロード
```bash
wget https://lab.ndl.go.jp/dataset/ndlocr/text_recognition/mojilist_NDL.txt -P ./text_recognition/models
wget https://lab.ndl.go.jp/dataset/ndlocr/text_recognition/ndlenfixed64-mj0-synth1.pth -P ./text_recognition/models
```
作成済みDocker imageのダウンロード
```bash
cd Houji-OCR
docker pull momokg/socr-py37
```

### 2. Dockerコンテナの起動
【Linuxの場合】
```
sh dockerrun.sh
```
【Windowsの場合】
```
$ocr_dir= Convert-Path .
docker run -it -v ${ocr_dir}:/code --gpus all --name socr_runner socr-py37
```
Docker起動後は、以下でログインできます。
```
docker exec -it socr_runner bash
```

### 3. ディレクトリ構成について
以下のようなディレクトリ構成になります。
```
Houji-OCR
├── socr-exe.sh: 邦字新聞OCRを実行するためのスクリプト　
├── dockerrun.sh: dockerイメージを起動するためのスクリプト　　
├── input: OCRに入力するためのファイルを格納するためのディレクトリ　　
│ (例)└── test: 
│ └── test01.jpg  
├── layout: レイアウト解析モジュールプログラムコード  
├── text_recognition: 文字認識モジュールプログラムコード(NDLOCR ver.1.0を利用)
├── output_utils: 後処理プログラムコード  
├── fonts: フォント設定ファイル  
└── layout: レイアウト解析モジュールプログラムコード
```
文字認識モジュールは、
[NDLOCR ver 1.0の文字認識モジュール](https://github.com/ndl-lab/text_recognition/tree/ea196e064a8003a6f8ec3ab1d986e096987c8036)を邦字新聞OCRに利用可能な形に編集したものを利用しています。


