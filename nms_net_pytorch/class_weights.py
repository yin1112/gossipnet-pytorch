import numpy as np
from nms_net_pytorch import cfg
def get_class_counts(imdb):
    freq = np.ones((imdb['num_classes'] + 1,), dtype=np.int64)
    for roi in imdb['roidb']:
        num_pos = 0
        if 'gt_classes' in roi:#每一个框的类别
            num_pos = roi['gt_classes'].size
            for cls in roi['gt_classes']:
                freq[cls] += 1
        if 'det_classes' in roi:#每一个框的类别
            num_bg = max(0, roi['det_classes'].size - num_pos)
            freq[0] += num_bg
    #返回所有图片的分类情况，0表示det框多出来的框，其他表示每个类别的总数 可能
    return freq

def class_equal_weights(imdb):
    num_classes = imdb['num_classes']
    posweight = cfg.train.pos_weight
    # what we want the expectation of each class weight to be
    expected_class_weight = np.array(
            [1 - posweight] + [posweight / num_classes] * num_classes,
            dtype=np.float32)
    class_counts = get_class_counts(imdb)
    num_samples = np.sum(class_counts)

    class_weights = num_samples * expected_class_weight / class_counts
    #返回一个权重向量，表示每个类别站的比重。
    return class_weights