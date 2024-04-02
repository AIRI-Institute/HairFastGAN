import sys
sys.path.append('.')

from external_code.landmarks_util import detect_landmarks, predictor_dict
from global_value_utils import GLOBAL_DATA_ROOT, DATASET_NAME


data_name = DATASET_NAME

root_dir = GLOBAL_DATA_ROOT

for landmark_num in [81, 68]:
    print('detect %d landmarks' % landmark_num)
    detect_landmarks(root_dir, data_name,
                     landmark_output_file_path=root_dir + '/landmark%d.pkl' % landmark_num,
                     output_dir=None, predictor=predictor_dict[landmark_num])
