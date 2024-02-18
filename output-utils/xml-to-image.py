import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as gfg
from PIL import Image, ImageDraw, ImageFont
import glob
from argparse import ArgumentParser

def get_options(args):
    parser = ArgumentParser()
    parser.add_argument('--resize_dir', type=str,
                           help='Resize image Directory')
    parser.add_argument('--xml_dir', type=str,
                           help='textrecognition result xml Directory')
    parser.add_argument('--out_dir', type=str,
                           help='Output dir')
    options = parser.parse_args(args)
    return options

# xmlファイルからファイル名(img)を取り出す
def xml_getName(root, f_list, f_name):

    for pagename in root.iter("PAGE"):
        name = pagename.attrib["IMAGENAME"]
        f_list.append(name)
        _filename, _ = os.path.splitext(os.path.basename(name))
        f_name.append(_filename)

# xmlファイルからデータ(x,y,w,h,name)を取り出す
def xml_getData(root, datas, name):

    for filename in root:
        f_name = filename.attrib["IMAGENAME"]
        
        for line in filename.iter("LINE"):
            if f_name == name:
                s = line.attrib["STRING"]
                h = line.attrib["HEIGHT"]
                w = line.attrib["WIDTH"]
                x = line.attrib["X"]
                y = line.attrib["Y"]
                
                data = {"string": s, "height": int(h), "width": int(w),
                        "x": int(x), "y": int(y)}
                
                datas.append(data)
        
def output_text(datas, img_pth, out_dir, name):

    im = Image.open(img_pth).convert(mode="RGB")
    font_name = "fontsfile/ipamp.ttf"
    font_size = 15
    font = ImageFont.truetype(font_name, font_size)
    im_w, im_h = im.size
    d = ImageDraw.Draw(im)

    for data in datas:
        s, _h, w, _x, y = data.values()
        x = _x + w
        d.text((x, y), s, (0, 0, 110), font=font, direction="ttb")

    out = out_dir+"text/"
    if not os.path.exists(out):
        os.mkdir(out)
    im.save(out+name+"-xmltext.jpg")

def output_box(datas, img_pth, out_dir, name):

    im = cv2.imread(img_pth)
    
    for data in datas: # data: s, h, w, x, y
        _, h, w, x1, y1 = data.values()
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(im,
                      pt1=(x1, y1), pt2=(x2, y2),
                      color=(0, 0, 128),
                      thickness = 2)

    out = out_dir+"box/"
    if not os.path.exists(out):
        os.mkdir(out)
    name = out+name+"-xmlbox.jpg"
    cv2.imwrite(name, im)

    return name

def output_textfile(datas, out_dir, name):

    out_txtfile = out_dir+name+"_text.txt"
            
    with open(out_txtfile, mode='w', encoding='utf-8') as f:

        for data in datas:
            lines, _, _, _, _ = data.values()
            #print(lines)
            f.write(lines+"\n")
            #for line in lines:
                #print(line)
                #text = line["string"]
                #f.write(text+"\n")


def main():

    try:
        options = get_options(sys.argv[1:])
    except FileNotFoundError as err:
        print(err)
        sys.exit()

    INPUT_PATH = options.resize_dir
    
    OUT_PATH = options.out_dir
    out_dir = OUT_PATH+"result/"
    os.makedirs(out_dir, exist_ok=True)

    XML_PATH = glob.glob(options.xml_dir+"*.xml")
    xml = XML_PATH[0]

    # XMLfile
    xml_tree = gfg.parse(xml)
    # get root
    root = xml_tree.getroot()

    # imagefile data
    f_list = []
    f_name = []
    xml_getName(root, f_list, f_name)
    print(f_list)
    for i, name in enumerate(f_list):

        # 重ねる画像（リサイズ済み）
        img_path = glob.glob(INPUT_PATH+name, recursive=True)[0]
        
        # data
        datas = []       
        xml_getData(root, datas, name)

        output_text(datas, img_path, out_dir, f_name[i])
        
        box_pth = output_box(datas, img_path, out_dir, f_name[i])       
        output_text(datas, box_pth, out_dir, f_name[i]+"-box")
        output_textfile(datas, out_dir+"/text/", f_name[i])
        
if __name__ == '__main__':

    main()
