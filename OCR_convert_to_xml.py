import os
import argparse
import shutil
from skimage.io import imread
FLAGS=None
ann = ['<annotation>', '</annotation>']
fldr = '<folder>ITUA</folder>'
obj = ['<object>','</object>']
fname = ['<filename>','</filename>']
src = '<source><database>ITUA</database><annotation>ITUA</annotation><image>flickr</image></source>'
w  =['<width>','</width>']
h = ['<height>','</height>']
d = ['<depth>','</depth>']
xmin = ['<xmin>','</xmin>']
xmax = ['<xmax>','</xmax>']
ymin = ['<ymin>','</ymin>']
ymax = ['<ymax>','</ymax>']
bbox = ['<bndbox>','</bndbox>']
size = ['<size>','</size>']
diff = ['<difficult>','</difficult>']
trunc = '<truncated>0</truncated>'
seg = '<segmented>0</segmented>'
pose  ='<pose>Frontal</pose>'
ocl = '<occluded>0</occluded>'

def create_xml(l, k):
        [fn, cl, id, xi, xa, yi, ya, _] = l
        ff = os.path.join(FLAGS.img_dir,fn)
        xml_str = ann[0] + fldr + fname[0] + cl+'_'+str(k)+'.jpg' + fname[1] + src
        xml_str = xml_str + size[0] + w[0] + str(imread(ff).shape[0]) + w[1]
        xml_str = xml_str + h[0] + str(imread(ff).shape[1]) + h[1]
        xml_str = xml_str+d[0]+str(imread(ff).shape[2])+d[1]+size[1]+seg+obj[0]+'<name>logo</name>'
        xml_str = xml_str+pose+trunc+ocl+bbox[0]+xmin[0]+xi+xmin[1]+xmax[0]+xa+xmax[1]
        xml_str = xml_str+ymin[0]+yi+ymin[1]+ymax[0]+ya+ymax[1]+bbox[1]
        xml_str = xml_str+diff[0] + str(0) + diff[1]+obj[1]+ann[1]
        file = os.path.join(FLAGS.xml_dir, cl+'_'+str(k)+'.xml')
        hlr = open(file, 'w')
        hlr.write(xml_str)
        hlr.close()
        #print(xml_str)
def create_label_map(lset):
        file = os.path.join(FLAGS.xml_dir, 'label_map.pbtxt')
        h = open(file, 'w')
        for i, s in enumerate(lset):
                h.write('item { \n')
                h.write('\tid: ' + str(i+1) + '\n')
                h.write('\tname: ' + "'"+s+"'" + '\n')
                h.write('} \n\n')
        h.close()
        
def rename_file(l, k):
        src = os.path.join(FLAGS.img_dir, l[0])
        des = os.path.join(FLAGS.xml_dir, l[1]+'_'+str(k)+'.jpg')
        shutil.copy2(src, des)
def main():
        #file = '../flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt'
        cnt = 0
        file = FLAGS.ann_file
        f = os.path.join(FLAGS.xml_dir, "trainval.txt")
        h = open(f, 'w')
        label_set = set()
        hndlr = open(file, 'r')
        for line in hndlr:
                l = line.split(' ')
                h.write(l[1]+'_'+str(cnt+1)+'\n')
                label_set.add(l[1])
                create_xml(l, cnt+1)
                #rename_file(l, cnt+1)
                cnt = cnt + 1
        #print(label_set)
        #create_label_map(label_set)
        hndlr.close()
        h.close()
 if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--ann_file',
      type=str,
      default='pwd',
      help='path and file name to the annotations file.'
  )
  parser.add_argument(
      '--img_dir',
      type=str,
      default='pwd',
      help="path to images"
  )
  parser.add_argument(
      '--xml_dir',
      type=str,
      default='pwd',
      help="path to store xml files"
  )
  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
