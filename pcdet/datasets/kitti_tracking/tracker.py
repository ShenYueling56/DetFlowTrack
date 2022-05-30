SCORE_DECAY = 0.2


class tracker():
    total_num = 0
    def __init__(self, det, track_id):
        self.track_id = track_id
        self.label = det.label
        self.score = det.score
        self.boxes_lidar = det.box3d
        self.appear_life = 1
        self.miss_time = 0

    # 利用预测的场景流得到box3d, 并用来更新track
    def predict(self, box3d):
        # self.score = self.score - SCORE_DECAY
        self.boxes_lidar = box3d
        # print("predict box3d", self.boxes_lidar)

    # 如果在下一帧找到了匹配,就用新的det更新tracker
    def update(self, det):
        # self.score = self.score + SCORE_DECAY
        # self.score = 1 - ((1 - self.score)*(1 - det.score))/((1-self.score)+(1-det.score))
        # self.score = 1 - ((1-self.score)*(1-det.score))
        self.score = det.score
        # self.score = max(self.score, det.score)
        self.boxes_lidar = det.box3d
        self.appear_life = self.appear_life+1
        assert self.label == det.label
        # print("update box3d ", self.boxes_lidar)

class det():
    def __init__(self, box3d, label, score):
        self.box3d = box3d
        self.label = label
        self.score = score
