import torch
from fvcore.common.file_io import PathManager
from collections import OrderedDict
import copy
import pickle
model_file = "/home/wangxuanhan/research/project/detectron2-master/coco_dp_exps/ResNet50_fpn_MID_Kpt_Net_1lx/model_final.pth"
# model_file = "/home/wangxuanhan/research/project/detectron2-master/coco_dp_exps/ResNet50_fpn_MID_Kpt_Net_1lx/model_final_162be9.pkl"
model = torch.load(model_file, map_location=torch.device("cpu"))
# model = pickle.load(open(model_file,'rb'))
new_model = {}
new_model['model'] = model['model']
# new_model = copy.deepcopy(model)
# param = OrderedDict()
# print(new_model['model'].keys())
# for k in list(new_model['model'].keys()):
#     target_k = k
#     if 'roi_heads' in k :
#         print(k)
#     if 'roi_heads.decoder' in k:
#         target_k = k.replace('roi_heads.decoder','roi_heads.mid_decoder')
#     if 'ann_index' in k:
#         target_k = k.replace('ann_index','m')
#     new_model['model'][target_k] = new_model['model'][k]
# pickle.dump(new_model,open(model_file,'wb'))
with PathManager.open(model_file, "wb") as f:
    torch.save(new_model, f)