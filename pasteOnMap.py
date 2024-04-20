import argparse, os, time, glob, cv2
import numpy as np
from scipy.misc import imsave, imread, imresize
from keras.preprocessing import image
from keras.layers import Multiply, Input, BatchNormalization, Activation, Add, concatenate
from keras.layers import Dropout, Flatten, Dense, UpSampling2D, Conv2D, MaxPooling2D, Deconvolution2D, Reshape, Activation, Maximum
from keras.layers import Conv2DTranspose, GlobalAveragePooling2D, Deconv2D
from keras.models import Model
from keras import regularizers
from keras import backend as K

imgsize = (512, 512)
l2_w = 2.5e-4
parser = argparse.ArgumentParser()

parser.add_argument("--mode", dest='mode', 
                    help="single or video")
parser.add_argument("--map", dest='map',  nargs='?',
                    help="Path to the content image")
parser.add_argument("--front", dest='front',  nargs='?',
                    help="Path to the video content")
parser.add_argument("--map_folder", dest='map_folder')

args = parser.parse_args()

def get_model_8():
    inputs = Input((None, None, 3))
    seg = Input((None, None, 3))



    x0 = rb(inputs, (16, 4, 4),   True)
    x = rb(x0, (16, 4, 4))
    x = rb(x, (16, 4, 4))
    d0 = rb(x, (16, 4, 4))


    """"""""""""""""""""
    se = rb(seg, (32, 8, 8), True)

    x1 = rb(d0, (32, 8, 8), True)
    x1= concatenate([x1, se])


    x = rb(x1, (32, 8, 8), True)
    x = rb(x, (32, 8, 8))
    x = rb(x, (32, 8, 8))
    d1 = rb(x, (32, 8, 8))
    """"""""""""""""""""

    """"""""""""""""""""

    se1 = rb_down(se, (64, 16, 16))

    x2 = rb_down(d1, (64, 16, 16))
    x2= concatenate([x2, se1])

    x = rb(x2, (64, 16, 16), True)
    x = rb(x, (64, 16, 16))
    x = rb(x, (64, 16, 16))
    x = rb(x, (64, 16, 16))
    d2 = rb(x, (64, 16, 16))

    """"""""""""""""""""

    """"""""""""""""""""

    se2 = rb_down(se1, (128, 32, 32))

    x3 = rb_down(d2, (128, 32, 32))
    x3 = concatenate([x3, se2])

    x = rb(x3, (128, 32, 32), True)
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    d3 = rb(x, (128, 32, 32))

    """"""""""""""""""""

    """"""""""""""""""""

    se3 = rb_down(se2, (256, 64, 64))

    x4 = rb_down(d3, (256, 64, 64))
    x4 = concatenate([x4, se3])

    x = rb(x4, (256, 64, 64), True)
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    x = rb(x, (256, 64, 64))
    d4 = rb(x, (256, 64, 64))


    """"""""""""""""""""


    """"""""""""""""""""

    se4 = rb_up(se3, (128, 32, 32))

    x5 = rb_up(d4, (128, 32, 32))
    x5 = concatenate([x5, se4])

    x = rb(x5, (128, 32, 32), True)
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    x = rb(x, (128, 32, 32))
    u1 = rb(x, (128, 32, 32))

    s1 = concatenate([u1, d3])



    """"""""""""""""""""

    """"""""""""""""""""
    se5 = rb_up(se4, (64, 16, 16))

    x6 = rb_up(s1, (64, 16, 16))
    x6 = concatenate([x6, se5])

    x = rb(x6, (64, 16, 16), True)
    x = rb(x, (64, 16, 16))
    x = rb(x, (64, 16, 16))
    x = rb(x, (64, 16, 16))
    u2 = rb(x, (64, 16, 16))

    s2 = concatenate([u2, d2])

    """"""""""""""""""""

    """"""""""""""""""""


    x7 = rb_up(s2, (32, 8, 8))

    x = rb(x7, (32, 8, 8), True)

    x = rb(x, (32, 8, 8))
    x = rb(x, (32, 8, 8))
    u3 = rb(x, (32, 8, 8))

    s3 = concatenate([u3, d1])


    """"""""""""""""""""

    x = rb(s3, (16, 4, 4), True)

    x = rb(x, (16, 4, 4))
    x = rb(x, (16, 4, 4))
    x = rb(x, (16, 4, 4))

    x = concatenate([x, d0])

    mapoutput = rb(s3, (3, 3, 3), True)


    model = Model(inputs=[inputs,seg], outputs=mapoutput)

    def loss_fun(y_true, y_pred):
        pred_soft = K.softmax(y_pred)
        true_soft = K.softmax(y_true)
        cy = K.std(K.batch_flatten(y_true), axis=-1, keepdims=True)

        cy_s = K.square(cy)
        ck = K.std(K.batch_flatten(pred_soft), axis=-1, keepdims=True)
        ck_s = K.square(ck)
        center_invariant_loss = 0.25*K.square(cy_s - ck_s)
        center_loss = 0.5*K.square(K.batch_flatten(pred_soft) - cy)
        cross = K.categorical_crossentropy(true_soft, pred_soft)
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        true_std = K.std(y_true, axis=-1)
        pred_std = K.std(y_pred, axis=-1)
        std_loss = K.mean(K.square(pred_std - true_std), axis=-1)
        u_true = K.mean(y_true, axis=-1)
        u_pred = K.mean(y_pred, axis=-1)
        var_true = K.var(y_true, axis=-1)
        var_pred = K.var(y_pred, axis=-1)
        covar_true_pred = K.mean(y_true * y_pred, axis=-1) - u_true * u_pred
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= K.clip(denom, K.epsilon(), np.inf)
        dssim_loss = K.mean((1.0 - ssim) / 2.0)

        base_loss = K.sum(cross) + K.sum(std_loss)
        base_loss += K.sum(mse) + K.sum(dssim_loss)

        mae = K.mean(K.abs(y_pred - y_true), axis=-1)
        loss = 10*K.sum(mae)
        # loss += (K.sum(cross)+K.sum(center_invariant_loss)+K.sum(center_loss))*content_loss
        # loss += (K.sum(cross)+K.sum(center_invariant_loss)+K.sum(center_loss))*style
        #loss += style

        tv = total_variation_loss(y_pred)
        loss += 0.01*tv

        return loss
    model.compile(optimizer='Adam',
                      loss= loss_fun,
                 metrics= ['acc'])
    return model
def total_variation_loss(x):
    height=256
    width=256
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, 1:, :width - 1])
        b = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, :height - 1, 1:])
    else:
        a = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
        b = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
def rb_down(input_tensor, filters, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    x = BatchNormalization(momentum=0.99)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filter1, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(l2_w))(x)


    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_w))(x)


    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1), kernel_regularizer=regularizers.l2(l2_w))(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(l2_w))(input_tensor)

    x = Add()([x, shortcut])

    return x
def rb_up(input_tensor, filters, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    x = BatchNormalization(momentum=0.99)(input_tensor)
    x = Activation('relu')(x)
    x = Deconv2D(filter1, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(l2_w))(x)



    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_w))(x)


    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1), kernel_regularizer=regularizers.l2(l2_w))(x)

    shortcut = Deconv2D(filter3, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(l2_w))(input_tensor)

    x = Add()([x, shortcut])


    return x
def rb(input_tensor, filters, shortcut_conv=False):
    filter1, filter2, filter3 = filters
    x = BatchNormalization(momentum=0.99)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filter1, (1, 1), kernel_regularizer=regularizers.l2(l2_w))(x)

    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_w))(x)


    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1), kernel_regularizer=regularizers.l2(l2_w))(x)

    if shortcut_conv:

        shortcut = Conv2D(filter3, (1, 1), kernel_regularizer=regularizers.l2(l2_w))(input_tensor)
        x = Add()([x, shortcut])
    else:

        x = Add()([x, input_tensor])
    return x
def main():
    print('Loading model...')
    model = get_model_8()
    model.load_weights('./weight/0630_addphoto_home_cccweightEcEsd4.h5')
    print('Load model successfully')
    start_time = time.time()
    map = os.path.join("./data/artmap/", args.map+".jpg")

    map = imread(map, mode='RGB')
    height, width, _ = map.shape
    h = (height//16)*16
    w = (width//16)*16
    
    map = imresize(map, (h, w))
    map = image.img_to_array(map)
    
    frame_dir = glob.glob(os.path.join("data/content/", args.front, '*.jpg'))
    frame_dir.sort()
    outputdir = "data/withmap/"+  args.front + "_" + args.map.split('.')[0] + "/"
    if not os.path.exists(outputdir):
        print(outputdir)
        os.makedirs(outputdir)
    for front in frame_dir:
        output = outputdir + front.split("/")[-1]
        front = imread(front, mode='RGB')
        front = imresize(front, (h, w))
        front = image.img_to_array(front)
        pimg = cv2.addWeighted(front, 0.6, map, 0.4, -100)
        end_time = time.time()
        print('addphoto model predict time: %ds' % (end_time - start_time))
        cv2.imwrite(output, pimg)
    


def blendimage(alpha = 0.9):
       if args.mode == "single":
        if args.map_folder=='artmap':
            map = os.path.join("./data/artmap/", args.map)
        else:
            map = os.path.join("./data/ed_map/", args.map)
        map = imread(map, mode='RGB')
        height, width, _ = map.shape
        h = (height//16)*16
        w = (width//16)*16
        print(h, w)
        map = imresize(map, (h, w))
        map = image.img_to_array(map)
        # map = np.expand_dims(map, axis=0)
        front = "data/test/single/" + args.front + ".jpg"
        
        output = "data/test/single_map/" + args.front + ".jpg"
        front = imread(front, mode='RGB')
        front = imresize(front, (h, w))
        front = image.img_to_array(front)
        # front = np.expand_dims(front, axis=0)
        merge = map
        for i in range(h):
            for j in range(w):
                # print(front[i][j])
                if front[i][j][0]<150:
                    merge[i][j][:] = front[i][j][:] * alpha + map[i][j][:] * (1-alpha)
        print(merge.shape)
        imsave(output, merge)
if __name__ == "__main__":
    main()
    