# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'


class DINO(object):
    def __init__(self, device='cpu'):
        # Grounding DINO
        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util import box_ops
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
        from huggingface_hub import hf_hub_download
        
        self.device = device
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        self.ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        
        cache_config_file = hf_hub_download(repo_id=self.ckpt_repo_id, filename=self.ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=self.ckpt_repo_id, filename=self.ckpt_filenmae)
        checkpoint = torch.load(cache_file)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        _ = model.eval()
        self.groundingdino_model = model #load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename).to(device)
        self.detect = predict
        self.types = ['beauty products', 'makeup products', 'perfume', 'gift boxes']
        self.transform = T.Compose([T.RandomResize([800], max_size=1333),
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
    
    def run(self, input):
        from torchvision.ops import box_convert
        from decalib.utils.util import convert_cv2pil
        
        print(f'{input.shape}')
        w, h, _ = input.shape
        input = convert_cv2pil(input, resize=768)
        image, _ = self.transform(input, None) #load_image(img_path, 768)
        for i in range(len(self.types)):
            detected_boxes, logits, phrases = self.detect(model=self.groundingdino_model,
                                                          image=image, 
                                                          caption=self.types[i], 
                                                          box_threshold=0.3,
                                                          text_threshold=0.25,
                                                          device=self.device)
            print(detected_boxes[0])
            if detected_boxes.size(0) != 0:
                break
        # TBC
        boxes_unnorm = detected_boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        bbox = boxes_xyxy[0].squeeze()
        return bbox, 'bbox'
        

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'



