import os

def check_dataset(data_dir):
    required_files = ['calib', 'image_2', 'label', 'planes', 'ImageSets']
    for folder in ['training', 'testing']:
        for req_file in required_files:
            path = os.path.join(data_dir, folder, req_file)
            if not os.path.exists(path):
                print(f"Missing {path}")

data_dir = '/home/gao/MonoCD/data/kitti'
check_dataset(data_dir)
