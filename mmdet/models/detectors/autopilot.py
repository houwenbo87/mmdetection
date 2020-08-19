from ..builder import DETECTORS
from .autopilot_detector import AutopilotDetector


@DETECTORS.register_module()
class Autopilot(AutopilotDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 vehicle_head=None,
                 parkspace_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Autopilot, self).__init__(backbone, neck, bbox_head, vehicle_head, parkspace_head, train_cfg,
                                   test_cfg, pretrained)
