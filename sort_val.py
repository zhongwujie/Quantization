'''
@brief: sort the valiation data. The dataset is downloader from the kaggle website. 
@reference: https://github.com/williamFalcon/pytorch-imagenet-dataset
'''
import os
import subprocess

def run(folder_path):
    # load up the relevant files
    src_images_path = os.path.join(folder_path, 'ILSVRC/Data/CLS-LOC/val')
    des_images_path = os.path.join(folder_path, 'ILSVRC/Data/CLS-LOC/val_sorted')
    if not os.path.exists(des_images_path):
        os.makedirs(des_images_path)
    annotations_file_path = os.path.join(folder_path, "LOC_val_solution.csv")

    with open(file=annotations_file_path, mode='r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            img_name, predic_str = line.split(',')
            img_name = img_name + ".JPEG"
            class_id = predic_str.split(' ', 1)[0]
            print(i, " ", class_id, img_name)
            # make class id folder
            from_path = os.path.join(src_images_path, img_name)
            if os.path.exists(from_path):
                class_path = os.path.join(des_images_path, class_id)
                to_path = os.path.join(class_path, img_name)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                subprocess.run(["cp", from_path, to_path])
            else:
                print(f'skipping... {from_path}')
                

if __name__ == "__main__":
    folder_path = "../../dataset/imagenet"
    run(folder_path)