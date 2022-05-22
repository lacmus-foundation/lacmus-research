import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import collections
import numpy as np


def parse_voc_xml(node: ET.Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def get_imgSize_and_list_of_yxyx(xml_path):
    mytree = parse_voc_xml(ET.parse(xml_path).getroot())
    xyxy=[]
    if 'annotation' not in mytree:
        return ((0,0),())
    for p in mytree['annotation']['object']:
        y1=int(p['bndbox']['ymin'])
        x1=int(p['bndbox']['xmin'])
        y2=int(p['bndbox']['ymax'])
        x2=int(p['bndbox']['xmax'])
        xyxy.append(((x1,y1),(x2,y2)))
    size=(int(mytree['annotation']['size']['width']),int(mytree['annotation']['size']['height']))
    return size, xyxy


def get_xywh_from_point(size,points,file_id=None):
    w,h=size
    x=((points[0][0]+points[1][0])/2)/w
    y=((points[0][1]+points[1][1])/2)/h
    im_w=abs((points[0][0]-points[1][0]))/w
    im_h=abs((points[0][1]-points[1][1]))/h
    return [x,y,im_w,im_h,file_id]

