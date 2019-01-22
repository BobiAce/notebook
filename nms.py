###=========================
##        NMS实现
import numpy as np
def py_cpu_nms(x1,y1,x2,y2,scores, thresh):
    """Pure Python NMS baseline."""
    # 所有box的坐标信息
    # x1 = dets[:, 0]
    # y1 = dets[:, 1]
    # x2 = dets[:, 2]
    # y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算出所有box的面积
    order = scores.argsort()[::-1]  # 图片评分按降序排序

    keep = []  # 用来存放保留的box的索引
    while order.size > 0:
        i = order[0]  # i 是还未处理的图片中的最大评分索引
        keep.append(i)  # 保留改图片的值
        # 矩阵操作，下面计算的是box_i分别与其余box相交的矩形的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算出各个相交矩形的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 只保留比例小于阙值的box，然后继续处理,因为这可能是另外一个目标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


#自己仿照的实现：
def nms_self(boxes,thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]

    aera = (x2-x1+1)*(y2-y1+1)
    #按照评分的升序排列
    index = np.argsort(score)[::-1]

    while len(index) > 0:
        i = index[0]
        pick.append(i)
        xx1 = np.maximum(x1[i],x1[index[1:]])
        yy1 = np.maximum(y1[i],y1[index[1:]])
        xx2 = np.minimum(x2[i],x2[index[1:]])
        yy2 = np.minimum(y2[i],y2[index[1:]])

        w = np.maximum(0,xx2-xx1+1)
        h = np.maximum(0,yy2-yy1+1)

        if w>0 and h>0:
            overlap = (w*h)/(aera[i] + aera[1:] - (w*h))
        inds = np.where(overlap <= thresh)[0]
        index = index[inds + 1]
#不得不说，np.where这个操作还是很经典的，我自己最开始是采用的np.delete的方式，但是后来又看了原版实现
if __name__ == '__main__':
    loc = np.random.randint(1, 50, [6, 2])
    size = np.random.randint(50, 100, [6, 2])
    score = [(x+0.4)/1.4 for x in np.random.rand(6)]
    score = np.array(score)
    x1 = loc[:,0]
    y1 = loc[:,1]
    x2 = size[:,0]
    y2 = size[:,1]
    keep = py_cpu_nms(x1,y1,x2,y2,score, 0.18)