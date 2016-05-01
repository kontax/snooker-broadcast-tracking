#!/usr/bin/python
# convert.py
# Converts the image metafiles into something that can be read by darknet
# ie. "Label x y w h" in float

from random import randint
from os import listdir
from os.path import join, splitext

width = 1280
height = 720
folder = 'labels'
labels = ['white','red','yellow','green','brown','blue','pink','black']

# Images where the specified pocket is covered
rm_top_left = ['00012-41','00052-32','00108-33','00144-48','00217-20','00221-25','00250-07']
rm_top_right = ['00010-24','00052-35','00184-00','00213-23','00052-32']
rm_bot_left = ['00026-48','00048-10','00050-17','00066-47','00095-29','00150-26','00165-06',
               '00199-03','00220-35','00241-00']
rm_bot_right = ['00034-23','00066-08','00089-23','00106-38','00120-03','00124-14','00125-19',
                '00149-04','00169-03','00178-10','00210-14','00211-20','00242-13','00250-21','00256-05']


def to_xml(l, x, y, w, h):
    s = '\t<object>\n' + \
        '\t\t<name>{}</name>\n'.format(l) + \
        '\t\t<truncated>0</truncated>\n' + \
        '\t\t<difficult>0</difficult>\n' + \
        '\t\t<bndbox>\n' + \
        '\t\t\t<xmin>{}</xmin>\n'.format(x) + \
        '\t\t\t<ymin>{}</ymin>\n'.format(y) + \
        '\t\t\t<xmax>{}</xmax>\n'.format(x + w) + \
        '\t\t\t<ymax>{}</ymax>\n'.format(y + h) + \
        '\t\t</bndbox>\n' +  \
        '\t</object>\n'
    return s


def get_xy(label, x, y, x_offset=0, y_offset=0):
    return (int(label[x].split(':')[1]) + x_offset,
            int(label[y].split(':')[1]) + y_offset)


def label_corners(label, filename):

    im = splitext(filename)[0][:8]
    xml = ''

    if im not in rm_bot_left:
        p1 = get_xy(label, 2, 3)
        p1 = (p1[0] - 60, p1[1])
        xml = to_xml('pocket', p1[0], p1[1], 60, 30)

    if im not in rm_top_left:
        #p2 = get_xy(label, 4, 5, 10, 10)
        p2 = get_xy(label, 4, 5)
        p2 = (p2[0] - 40, p2[1] - 15)
        xml = xml + to_xml('pocket', p2[0], p2[1], 40, 15)

    if im not in rm_top_right:
        r = randint(-2, 2) # Random +- for mess up y value
        p3 = get_xy(label, 6, 5) # messed up this y value
        p3 = (p3[0], p3[1] - 15 + r)
        xml = xml + to_xml('pocket', p3[0], p3[1], 40, 15)

    if im not in rm_bot_right:
        p4 = get_xy(label, 8, 9)
        xml = xml + to_xml('pocket', p4[0], p4[1], 60, 30)

    return xml


if __name__ == '__main__':

    files = [f for f in listdir(folder) if f.endswith('.txt')]

    for data in files:
        print 'Updating ' + data

        newFile = join('fixed', data)
        oldFile = join(folder, data)
        updated = []

        xml = '<annotation>\n' + \
              '\t<folder>snooker</folder>\n' + \
              '\t<filename>{}</filename>\n'.format(data) + \
              '\t<size>\n' + \
              '\t\t<width>{}</width>\n'.format(width) + \
              '\t\t<height>{}</height>\n'.format(height) + \
              '\t\t<depth>3</depth>\n' + \
              '\t</size>\n'

        with open(oldFile) as f:
            for line in f.readlines():
                s = line.split()
                if len(s) < 2:
                    continue
                elif s[1] == '#FF00FF00':
                    xml = xml + label_corners(s, data)
                else:
                    label = s[1].split(':')[1]
                    x = int(s[2].split(':')[1])
                    y = int(s[3].split(':')[1])
                    w = int(s[4].split(':')[1])
                    h = int(s[5].split(':')[1])

                    label = labels[int(label)]
                    xml = xml + to_xml(label, x, y, w, h)

        xml = xml + '</annotation>\n'

        with open(newFile, 'w') as n:
            n.write(xml)
