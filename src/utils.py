import scipy.misc, numpy as np, os, sys#!!!

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):#默认False,不对输入的图片进行缩放
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(arr, (256, 256, 3))#imresize用于对图像做缩放处理。arr: 需要被调整尺寸的图像数组
   if not (len(img.shape) == 3 and img.shape[2] == 3):#!!!在矩阵中，[0]就表示行数，[1]则表示列数。?
       img = np.dstack((img,img,img))#堆栈数组按顺序深入（沿第三维）。使图片变为三通道
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

