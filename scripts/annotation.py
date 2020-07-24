import os
import argparse


def convert_annotation(data_path, data_type, anno_file):
    img_inds_file = os.path.join(data_path, 'imgsets', data_type + '.txt')
    with open(img_inds_file, 'r') as f_img_inds_file:
        txt = f_img_inds_file.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_file, 'a') as f_anno_file:
        for idx, image_ind in enumerate(image_inds):
            label_file = os.path.join(data_path, 'annos', image_ind + '.txt')
            if not os.path.exists(label_file):
                print('idx=%d, label_file=%s not exist' % (idx, label_file))
                continue

            visible_img_file = os.path.join(data_path, 'images/visible', image_ind + '.jpg')
            if not os.path.exists(visible_img_file):
                print('idx=%d, visible_img_file=%s not exist' % (idx, visible_img_file))
                continue

            lwir_img_file = os.path.join(data_path, 'images/lwir', image_ind + '.jpg')
            if not os.path.exists(lwir_img_file):
                print('idx=%d, lwir_img_file=%s not exist' % (idx, lwir_img_file))
                continue    
            
            annotation = visible_img_file + ' ' + lwir_img_file
            with open(label_file, 'r') as f_label_file:
                lines = f_label_file.readlines()

            is_write = False
            for obj in lines:
                info = obj.split()
                cls = info[0]

                if ('person' in cls) or ('people' in cls):
                    cls_id = 0
                    xmin = int(info[1]) if int(info[1]) >= 0 else 0
                    ymin = int(info[2]) if int(info[2]) >= 0 else 0
                    xmax = xmin + int(info[3])
                    ymax = ymin + int(info[4])
                    
                    annotation += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(cls_id)])
                    if not is_write:
                        is_write = True
            
            print('idx=%d, annotation=%s' % (idx, annotation), 'is_write=', is_write)
            if is_write:
                f_anno_file.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dataset/kaist")
    parser.add_argument("--train_annotation", default="data/dataset/pedestrian_train.txt")
    parser.add_argument("--val_annotation",  default="data/dataset/pedestrian_val.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):
        os.remove(flags.train_annotation)

    if os.path.exists(flags.val_annotation):
        os.remove(flags.val_annotation)

    train_num = convert_annotation(flags.data_path, 'train', flags.train_annotation)
    val_num = convert_annotation(flags.data_path, 'val', flags.val_annotation)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (train_num, val_num))