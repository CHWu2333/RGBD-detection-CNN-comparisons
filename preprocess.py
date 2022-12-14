from PIL import Image
import os
from args import Args

file_dir = r"C:\Users\Articial_Idiot\Desktop\perception_roject\rgbd-dataset"
save_dir = r"C:\Users\Articial_Idiot\Desktop\perception_roject\rgb_data"
for i in range(len(Args.class_list)):
    for j in range(15):
        for k in range(4):
            for p in range(300):
                file_name = str(Args.class_list[i]) + "_" +str(j)+ "_" +str(k)+ "_" +str(p)+"_crop.png"

                file = os.path.join(file_dir, str(Args.class_list[i]), str(Args.class_list[i]) + "_" + str(j), file_name)

                try:
                    im = Image.open(file)
                    im1 = im.resize(Args.picture_size)
                    im1.save(os.path.join(save_dir, file_name))
                    # print("success")
                except:
                    print("no such directory "+str(file_name))
                    pass


