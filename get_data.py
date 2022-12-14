import glob
import numpy as np
import cv2
import torch

import time
def simple_progress_bar(i: int, n: int, init_time: float):
    avg_time = (time.time()-init_time)/(i+1)
    percent = ((i+1)/(n))*100
    print(
        end=f"\r|{'='*(int(percent))+'>'+'.'*int(100-int(percent))}|| " + \
        f"||Completion: {percent : 4.3f}% || \t "+ \
        f"||Time elapsed: {avg_time*(i+1):4.3f} seconds || \t " + \
        f"||Remaining time: {(avg_time*(n-(i+1))): 4.3f} seconds."
    )
    return


def load_data(dir_name):
    # dir_name can be rgb_data or depth_data
    t0 = time.time()
    count = 0
    file_type = 'png'
    img_list = glob.glob('./' + dir_name + '/*.' + file_type)
    temp_img_array_list = []
    temp_depth_img_array_list = []
    label_list = []
    for img in img_list:
        count += 1
        # if count > 5000:
        #     break
        if "depth" in img:
            temp_depth_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)/255
            temp_depth_img_array_list.append(temp_depth_img)
            del temp_depth_img
        else:
            temp_img = cv2.imread(img)/255
            temp_img_array_list.append(temp_img)
            del temp_img

        if "apple" in img:
            label = 0
        if "ball" in img:
            label = 1
        if "banana" in img:
            label = 2
        if "bell_pepper" in img:
            label = 3
        if "binder" in img:
            label = 4
        if "bowl" in img:
            label = 5
        if "calculator" in img:
            label = 6
        if "camera" in img:
            label = 7
        if "cap" in img:
            label = 8
        if "cell_phone" in img:
            label = 9
        if "cereal_box" in img:
            label = 10
        if "coffee_mug" in img:
            label = 11
        if "comb" in img:
            label = 12
        if "dry_battery" in img:
            label = 13
        if "flashlight" in img:
            label = 14
        if "food_bag" in img:
            label = 15
        if "food_box" in img:
            label = 16
        if "food_can" in img:
            label = 17
        if "food_cup" in img:
            label = 18
        if "food_jar" in img:
            label = 19
        if "garlic" in img:
            label = 20
        if "glue_stick" in img:
            label = 21
        if "greens" in img:
            label = 22
        if "hand_towel" in img:
            label = 23
        if "instant_noodles" in img:
            label = 24
        if "keyboard" in img:
            label = 25
        if "kleenex" in img:
            label = 26
        if "lemon" in img:
            label = 27
        if "lightbulb" in img:
            label = 28
        if "lime" in img:
            label = 29
        if "marker" in img:
            label = 30
        if "mushroom" in img:
            label = 31
        if "notebook" in img:
            label = 32
        if "onion" in img:
            label = 33
        if "orange" in img:
            label = 34
        if "peach" in img:
            label = 35
        if "pear" in img:
            label = 36
        if "pitcher" in img:
            label = 37
        if "plate" in img:
            label = 38
        if "pliers" in img:
            label = 39
        if "potato" in img:
            label = 40
        if "rubber_eraser" in img:
            label = 41
        if "scissors" in img:
            label = 42
        if "shampoo" in img:
            label = 43
        if "soda_can" in img:
            label = 44
        if "sponge" in img:
            label = 45
        if "stapler" in img:
            label = 46
        if "tomato" in img:
            label = 47
        if "toothbrush" in img:
            label = 48
        if "toothpaste" in img:
            label = 49
        if "water_bottle" in img:
            label = 50

        label_list.append(label)
        # simple_progress_bar(count-1, 136091, t0)

    temp_depth_img_array_list = np.array(temp_depth_img_array_list)
    temp_img_array_list = np.array(temp_img_array_list)
    label_list = np.array(label_list)
    t1 = time.time()
    print(t1-t0)
    return temp_img_array_list, temp_depth_img_array_list, label_list


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    # return 'cpu'


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

if __name__=="__main__":
    print("collecting rgb_images")
    a, _, c = load_data("rgb_data")
    # np.savetxt("rgb_data.txt", a)

    print("collecting depth_images")
    _, b, c = load_data("depth_data")
    # np.savetxt("depth_data.txt", b)
    depth_data = np.expand_dims(b, axis=3)
    print(depth_data.shape)
    print(c.shape)
    print(a[-1])
    print(depth_data[-1])
    X = np.concatenate(a, depth_data, 3)
    print(X.shape)


