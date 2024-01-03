# Copyright (c) OpenMMLab. All rights reserved.
# from .ppyoloe_head import PPYOLOEHead, PPYOLOEHeadModule
from .ppyoloe_head import PPYOLOEHead, PPYOLOEHeadModule
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from .rtmdet_ins_head import RTMDetInsSepBNHead, RTMDetInsSepBNHeadModule
from .rtmdet_rotated_head import (RTMDetRotatedHead,
                                  RTMDetRotatedSepBNHeadModule)
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule
from .yolov6_head import YOLOv6Head, YOLOv6HeadModule
from .yolov7_head import YOLOv7Head, YOLOv7HeadModule, YOLOv7p6HeadModule
from .yolov8_head import YOLOv8Head, YOLOv8HeadModule
from .yolox_head import YOLOXHead, YOLOXHeadModule
from .ppyoloe_head1 import PPYOLOEHeadMY, PPYOLOEHeadModuleMY

from .ppyoloe_head2 import PPYOLOEHeadMY2, PPYOLOEHeadModuleMY2
# from .ppyoloe_head_final import PPYOLOEHeadFinal, PPYOLOEHeadModuleFinal
# from .ppyoloe_head_final import PPYOLOEHead, PPYOLOEHeadModule

__all__ = [
    'YOLOv5Head', 'YOLOv6Head', 'YOLOXHead', 'YOLOv5HeadModule',
    'YOLOv6HeadModule', 'YOLOXHeadModule', 'RTMDetHead',
    'RTMDetSepBNHeadModule', 'YOLOv7Head',
    'PPYOLOEHead', 'PPYOLOEHeadModule',
    # 'PPYOLOEHead1', 'PPYOLOEHeadModule1'
    'YOLOv7HeadModule', 'YOLOv7p6HeadModule', 'YOLOv8Head', 'YOLOv8HeadModule',
    'RTMDetRotatedHead', 'RTMDetRotatedSepBNHeadModule', 'RTMDetInsSepBNHead',
    'RTMDetInsSepBNHeadModule',
    'PPYOLOEHeadMY', 'PPYOLOEHeadModuleMY',
    'PPYOLOEHeadMY2', 'PPYOLOEHeadModuleMY2',
    # 'PPYOLOEHeadFinal', 'PPYOLOEHeadModuleFinal',
    # 'PPYOLOEHead', 'PPYOLOEHeadModule',
]
