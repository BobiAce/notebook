# coding: utf-8
import numpy as np


class solutionNMS():
    def rere_nms(self, boxes, thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        score = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = score.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            # j = order[1:]
            keep.append(i)
            # 计算得分最高的框和剩余框的相交面积
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算相交面积百分比
            overlap = inter / (area[i] + area[order[1:]] - inter)
            inds = np.where(overlap <= thresh)[0]
            order = order[inds + 1]
        return keep


    def py_nms(self, bbox, thresh):
        """Pure Python NMS baseline."""
        # tl_x,tl_y,br_x,br_y及score
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        scores = bbox[:, 4]
        # 计算每个检测框的面积，并对目标检测得分进行降序排序
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []   # 保留框的结果集合
        while order.size > 0:
            i = order[0]
            keep.append(i)   # 保留该类剩余box中得分最高的一个
            # 计算最高得分矩形框与剩余矩形框的相交区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算相交的面积,不重叠时面积为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 计算IoU：重叠面积 /（面积1+-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            print(ovr)
            print('aa')
            # 保留IoU小于阈值的box
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]   # 注意这里索引加了1,因为ovr数组的长度比order数组的长度少一个
        return keep

    def re_nms(self, boxes, thresh, sigma):
        """
        repeat nms bbox = x1,y1,x2,y2,score
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        score = boxes[:, 4]
        # 计算每个矩形框的面积
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 根据score 降序排序 boxes的索引
        order = score.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 就按最高得分的框与剩下的框的相交区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            IOU = inter / (area[i] + area[order[1:]] - inter)

            # inds = np.where(IOU <= thresh)[0]
            # order = order[inds + 1]

            ids_t = np.where(IOU > thresh)[0]
            score[[order[ids_t+1]]] *= np.exp(-(IOU[ids_t] * IOU[ids_t]) / sigma)
            score = score[order[1:]]

            c_boxes = boxes[order[1:]]
            order = score.argsort()[::-1]

            x1 = c_boxes[:, 0]   # 因为bbox已经做了筛选了，areas需要重新计算一遍，抑制的bbox剔除掉
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        return keep

    def box_soft_nms(bboxes, scores, labels, nms_threshold=0.3, soft_threshold=0.3, sigma=0.5, mode='union'):
        """
        soft-nms implentation according the soft-nms paper
        :param bboxes: all pred bbox
        :param scores: all pred cls
        :param labels: all detect class label，注：scores只是单纯的得分，需配合label才知道具体对应的是哪个类
        :param nms_threshold: origin nms thres, for judging to reduce the cls score of high IoU pred bbox
        :param soft_threshold: after cls score of high IoU pred bbox been reduced, soft_thres further filtering low score pred bbox
        :return:
        """
        unique_labels = labels.cpu().unique().cuda()    # 获取pascal voc 20类标签

        box_keep = []
        labels_keep = []
        scores_keep = []
        for c in unique_labels:             # 相当于NMS中对每一类的操作，对应step-1
            c_boxes = bboxes[labels == c]   # bboxes、scores、labels一一对应，按照label == c就可以取出对应类别 c 的c_boxes、c_scores
            c_scores = scores[labels == c]
            weights = c_scores.clone()
            x1 = c_boxes[:, 0]
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)         # bbox面积
            _, order = weights.sort(0, descending=True)   # bbox根据score降序排序，对应NMS中step-2
            while order.numel() > 0:                      # 对应NMS中step-5
                i = order[0]                              # 当前order中的top-1，保存之
                box_keep.append(c_boxes[i])               # 保存bbox
                labels_keep.append(c)                     # 保存cls_id
                scores_keep.append(c_scores[i])           # 保存cls_score

                if order.numel() == 1:  # 当前order就这么一个bbox了，那不玩了，下一个类的bbox操作吧
                    break

                xx1 = x1[order[1:]].clamp(min=x1[i])      # 别忘了x1[i]对应x1[order[0]]，也即top-1，寻找Insection区域的坐标
                yy1 = y1[order[1:]].clamp(min=y1[i])
                xx2 = x2[order[1:]].clamp(max=x2[i])
                yy2 = y2[order[1:]].clamp(max=y2[i])

                w = (xx2 - xx1 + 1).clamp(min=0)          # Insection区域的宽、高、面积
                h = (yy2 - yy1 + 1).clamp(min=0)
                inter = w * h

                # IoU中U的计算模式，两种union、min，比较容易理解
                if mode == 'union':
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)
                elif mode == 'min':
                    ovr = inter / areas[order[1:]].clamp(max=areas[i])
                else:
                    raise TypeError('Unknown nms mode: %s.' % mode)

                # 经过origin NMS thres，得到高IoU的bboxes index，
                # origin NMS操作就直接剔除掉这些bbox了，soft-NMS就是对这些bbox对应的score做权重降低
                ids_t= (ovr>=nms_threshold).nonzero().squeeze()   # 高IoU的bbox，与inds = np.where(ovr >= nms_threshold)[0]功能类似

                # torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)：这个比较好理解，对score做权重降低的参数，从fig 2、公式中都可以参考
                # order[ids_t+1]：+1对应x1[order[0]]，也即top-1，若需映射回order中各个bbox，就必须+1
                # 这样整体上就容易理解了，就是soft-nms的score抑制方式，未使用NMS中粗暴的直接score = 0的抑制方式
                weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

                # soft-nms对高IoU pred bbox的score调整了一次，soft_threshold仅用于对score抑制，score太小就不考虑了
                ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()   # 这一轮未被抑制的bbox
                if ids.numel() == 0:   # 竟然全被干掉了，下一个类的bbox操作吧
                    break

                c_boxes = c_boxes[order[1:]][ids]   # 先取得c_boxes[order[1:]]，再在其基础之上操作[ids]，获得这一轮未被抑制的bbox
                c_scores = weights[order[1:]][ids]
                _, order = c_scores.sort(0, descending=True)
                if c_boxes.dim()==1:
                    c_boxes=c_boxes.unsqueeze(0)
                    c_scores=c_scores.unsqueeze(0)

                x1 = c_boxes[:, 0]   # 因为bbox已经做了筛选了，areas需要重新计算一遍，抑制的bbox剔除掉
                y1 = c_boxes[:, 1]
                x2 = c_boxes[:, 2]
                y2 = c_boxes[:, 3]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # scores_keep保存的是未做权重降低的score，降低权重的score仅用于soft-nms操作
        return box_keep, labels_keep, scores_keep


# if __name__ == '__main__':
#     bbox = np.array([[100, 120, 170, 200, 0.98],
#                      [20, 40, 80, 90, 0.99],
#                      [20, 38, 82, 88, 0.96],
#                      [200, 380, 282, 488, 0.9],
#                      [19, 38, 75, 91, 0.8]])
#     print(solutionNMS().re_nms(bbox, 0.5))

# #===================================
# # EAST 中Locality-Aware NMS的实现
# #===================================
import numpy as np
from shapely.geometry import Polygon

class solutionLNMS():
    def intersection(self, g, p):
        #取g,p中的几何体信息组成多边形
        g = Polygon(g[:8].reshape((4, 2)))
        p = Polygon(p[:8].reshape((4, 2)))

        # 判断g,p是否为有效的多边形几何体
        if not g.is_valid or not p.is_valid:
            return 0

        # 取两个几何体的交集和并集
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter/union

    def weighted_merge(self, g, p):
        # 取g,p两个几何体的加权（权重根据对应的检测得分计算得到）
        g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])

        #合并后的几何体的得分为两个几何体得分的总和
        g[8] = (g[8] + p[8])
        return g

    def standard_nms(self, S, thres):
        #标准NMS
        order = np.argsort(S[:, 8])[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ovr = np.array([self.intersection(S[i], S[t]) for t in order[1:]])
            inds = np.where(ovr <= thres)[0]
            order = order[inds+1]

        return S[keep]

    def nms_locality(self, polys, thres=0.3):
        '''
        locality aware nms of EAST
        :param polys: a N*9 numpy array. first 8 coordinates, then prob
        :return: boxes after nms
        '''
        S = []    #合并后的几何体集合
        p = None   #合并后的几何体
        for g in polys:
            if p is not None and self.intersection(g, p) > thres:    #若两个几何体的相交面积大于指定的阈值，则进行合并
                p = self.weighted_merge(g, p)
            else:    #反之，则保留当前的几何体
                if p is not None:
                    S.append(p)
                p = g
        if p is not None:
            S.append(p)
        if len(S) == 0:
            return np.array([])
        return self.standard_nms(np.array(S), thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    box = Polygon(np.array([[343, 350], [448, 135],
                      [474, 143], [369, 359]])).area
    print(box)
    g = [114.08222961, 29.94154549, 270.02160645, 28.1983242,

         270.58172607, 78.30197144, 114.64233398, 80.04519653, 0.87047273]

    P = [110.07213085, 29.98007349, 267.0800416, 27.57254931,

         267.85499947, 78.08751085, 110.84708252, 80.49503197, 0.85117343]
