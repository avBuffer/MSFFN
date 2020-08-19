import os
import sys
import random


if __name__ == '__main__':
    """
    argv = sys.argv
    if len(argv) < 2:
        print('usage: python create_list.py data_path')
        sys.exit()

    data_path = argv[1]
    if not os.path.exists(data_path):
        print(data_path, ' not exist')
        sys.exit()
    """

    data_path = 'D:/datasets/Pedestrians'
    print('create_list data_path=%s' % data_path)

    split = 0.9
    anno_path = os.path.join(data_path, 'annos')
    if not os.path.exists(anno_path):
        print(anno_path, ' not exist')
        sys.exit()

    visible_path = os.path.join(data_path, 'images/visible')
    if not os.path.exists(visible_path):
        print(visible_path, ' not exist')
        sys.exit()

    lwir_path = os.path.join(data_path, 'images/lwir')
    if not os.path.exists(lwir_path):
        print(lwir_path, ' not exist')
        sys.exit()

    imgset_path = os.path.join(data_path, 'imgsets')
    if not os.path.exists(imgset_path):
        os.makedirs(imgset_path)

    db = []
    annos = os.listdir(anno_path)
    for idx, anno in enumerate(annos):
        anno_file = os.path.join(anno_path, anno)
        #print('idx=', idx, 'anno_file', anno_file)
        if not os.path.exists(anno_file):
            print('idx=', idx, anno_file, ' not exist')
            continue

        visible_file = os.path.join(visible_path, anno.replace('.txt', '.jpg'))
        if not os.path.exists(visible_file):
            #print('idx=', idx, 'anno=', anno, 'visible_file=', visible_file, 'not exist')
            visible_file = os.path.join(visible_path, anno.replace('.txt', '.jpeg'))
            if not os.path.exists(visible_file):
                print('idx=', idx, 'anno=', anno, 'visible_file=', visible_file, 'not exist')
                continue
        
        lwir_file = os.path.join(lwir_path, anno.replace('.txt', '.jpg'))
        if not os.path.exists(lwir_file):
            #print('idx=', idx, 'anno=', anno, 'lwir_file=', lwir_file, 'not exist')
            lwir_file = os.path.join(lwir_path, anno.replace('.txt', '.jpeg'))
            if not os.path.exists(lwir_file):
                print('idx=', idx, 'anno=', anno, 'lwir_file=', lwir_file, 'not exist')
                continue

        db_str = anno.replace('.txt', '') + '\n'
        db.append(db_str)

    random.shuffle(db)
    split_num = int(split * len(db))
    train_db = db[:split_num]
    val_db = db[split_num:]
    print('annos.len=', len(annos), ' db=', len(db), ' train_db=', len(train_db), ' val_db=', len(val_db))

    types = ['train', 'val']
    for type in types:
        if type == 'train':
            type_db = train_db
        else:
            type_db = val_db

        type_file = os.path.join(imgset_path, type + '.txt')
        if os.path.exists(type_file):
            os.remove(type_file)

        f_type_file = open(type_file, 'w')
        for idx, file_name in enumerate(type_db):
            #print('type=', type, ' idx=', idx, ' file_name=', file_name)
            f_type_file.write(file_name)

        f_type_file.close()
        print('type=', type, ' type_db.len=', len(type_db), ' type_file=', type_file)
