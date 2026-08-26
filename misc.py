import os

if __name__ == '__main__':
    prefix = os.path.join('z:/', 'yue', 'pepple')
    sub_dirs = [x for x in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, x)) and 'vol' in x]
    test_volumes = []
    for x in sub_dirs:
        x_dirs = os.listdir(os.path.join(prefix, x))
        test_volumes += x_dirs

    print(test_volumes)
    with open(os.path.join(prefix, 'test_volumes.csv'), 'w') as fout:
        fout.write('\n'.join(test_volumes))
    fout.close()

    # # blue harddrive
    # test_vols2 = os.listdir(os.path.join('D:', 'Pepple OCT TIFF Stacks', 'Inflamed'))
    # test_vols2 += os.listdir(os.path.join('D:', 'Pepple OCT TIFF Stacks', 'Uninflamed'))
    # with open(os.path.join(prefix, 'test_volumes_blue.csv'), 'w') as fout:
    #     fout.write('\n'.join(test_vols2))
    # fout.close()

    # red harddrive
    test_vols3 = os.listdir(os.path.join('D:'))
    with open(os.path.join(prefix, 'test_volumes_red.csv'), 'w') as fout:
        fout.write('\n'.join(test_vols3))
    fout.close()