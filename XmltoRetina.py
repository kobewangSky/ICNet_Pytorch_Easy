import cv2, os, shutil
import xml.etree.ElementTree as ET
import glob
import csv


def readXmlAnno(im_fn, ann_DIR):
    anno_pn = os.path.join(ann_DIR, im_fn + '.xml')
    # print 'On annotation: {}'.format(anno_pn)
    tree = ET.parse(anno_pn)
    root = tree.getroot()

    p_anno = {}
    size = root.find('size')
    d_size = {"width": size.find('width').text,
              "height": size.find('height').text,
              "depth": size.find('depth').text
              }
    p_anno['size'] = d_size

    l_obj = []
    for obj in root.findall('object'):
        d_obj = {"name": obj.find('name').text, "truncated": '0.0', "difficult": '0.0', "occluded": '0.0',
                 "xmin": float(obj.find('bndbox').find('xmin').text),
                 "ymin": float(obj.find('bndbox').find('ymin').text),
                 "xmax": float(obj.find('bndbox').find('xmax').text),
                 "ymax": float(obj.find('bndbox').find('ymax').text),
                 }
        l_obj.append(d_obj)

    p_anno['l_obj'] = l_obj

    if len(l_obj) > 0:
        return p_anno
    else:
        return None


if __name__ == '__main__':
    CsvData = []
    XmlfileDir = '/home/kobe/easy-faster-rcnn.pytorch/data/VOCdevkit/VOCCar/Annotations/'

    csvpath = '/home/kobe/easy-faster-rcnn.pytorch/data/VOCdevkit/VOCCar/'

    xmlfiles = sorted(glob.glob(XmlfileDir + '*.xml'))
    for im_fn  in xmlfiles:
        im_fn = os.path.basename(im_fn).split('.')[0]
        p_anno = readXmlAnno(im_fn, XmlfileDir)

        annolist = p_anno['l_obj']
        for it in annolist:
            tempdata = []
            tempdata.append(im_fn + '.png')
            tempdata.append(it['xmin'])
            tempdata.append(it['ymin'])
            tempdata.append(it['xmax'])
            tempdata.append(it['ymax'])
            tempdata.append(it['name'])
            CsvData.append(tempdata)


    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    with open(csvpath + 'person1.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in CsvData:
            writer.writerow(row)




