import cv2
import numpy as np
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from argparse import ArgumentParser

def get_vid_frames(vid_path, frame_rate=4):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    frame_count = 0
    fr_old = int(math.ceil(vid.get(cv2.CAP_PROP_FPS)))
    while vid.isOpened():
        s, i = vid.read()
        if s and frame_count%(fr_old//frame_rate)==0:
            frames.append(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY))
        elif not s:
            break
        frame_count += 1
    vid.release()
    return np.array(frames)

def downsample(i):
    new_size = (i.shape[1]//2, i.shape[0]//2)

    return cv2.resize(i, new_size, interpolation=cv2.INTER_NEAREST)

def get_ss_gauss_pyr(i, n_levels, sigma=1.6):
    'Gaussian pyramid with scale space 5 for each level.'

    gau_pyr = []
    level = [i]

    k = (2**(.2))

    sigs = [1.6, math.sqrt((k**2) - 1)]
    for i in range(2, 8):
        sigs.append(sigs[i-1] * k)

    for n in range(n_levels):
        for s in range(8):
            if s == 0:
                if n > 0:
                    level.append(downsample(gau_pyr[n-1][5,:,:]))
            else:
                level.append(np.float32(cv2.GaussianBlur(level[s-1],(0,0), sigs[s], sigmaY=sigs[s])))
        gau_pyr.append(np.array(level))
        level = []

    return gau_pyr

def diff_of_gauss(ss_pyr):

    DoGs = []

    for o in ss_pyr:
        DoG = []
        print('SHAPE',(o[2,:,:] - o[1,:,:]).shape)

        for n in range(7):
            DoG.append((o[n+1,:,:] - o[n,:,:]))

        DoGs.append(np.array(DoG))

    return DoGs


def is_local_max(DoG, le):

    for s in range(-1,1):
        for y in range(-1,1):
            for x in range(-1,1):
                if DoG[le[0]+s,le[1]+y,le[2]+x] > DoG[le]:
                    return False
    return True

def is_local_min(DoG, le):

    for s in range(-1,1):
        for y in range(-1,1):
            for x in range(-1,1):
                if DoG[le[0]+s,le[1]+y,le[2]+x] < DoG[le]:
                    return False
    return True


def local_extrema(DoG, threshold):
    le = []

    pos_DoG = np.abs(DoG)

    for y in range(8, DoG.shape[1]-8):
        for x in range(8, DoG.shape[2]-8):
            for s in range(1,6):
                if pos_DoG[s,y,x] > threshold:
                    if is_local_max(DoG,(s,y,x)):
                        le.append((s,y,x))
                    elif is_local_min(DoG, (s,y,x)):
                        le.append((s,y,x))

    return le


def derivs(DoG, s, y, x):
    #computes partial derivatives w.r.t. s, y ,x
    dx = (DoG[s, y, x+1] - DoG[s, y, x-1])/2.0
    dy = (DoG[s, y+1, x] - DoG[s, y-1, x])/2.0
    ds = (DoG[s+1, y, x] - DoG[s-1, y, x])/2.0

    deriv = np.array([dx,dy,ds])

    return deriv


def localize(DoG, s, y, x):
    #Hessian Matrix

    v = DoG[s,y,x]
    dxx = DoG[s,y,x+1] + DoG[s,y,x-1] - 2*v
    dyy = DoG[s,y+1,x] + DoG[s,y-1,x] - 2*v
    dss = DoG[s+1,y,x] + DoG[s-1,y,x] - 2*v

    dxy = (DoG[s,y+1,x+1] - DoG[s,y+1,x-1] - DoG[s,y-1,x+1] + DoG[s,y-1,x-1])/4.0
    dxs = (DoG[s+1, y, x+1] - DoG[s+1, y, x-1] - DoG[s-1,y,x+1] + DoG[s-1,y,x-1])/4.0
    dys = (DoG[s+1, y+1, x] - DoG[s+1, y-1, x] - DoG[s-1,y+1,x] + DoG[s-1,y-1,x])/4.0

    H = np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
    return H


def subpx_step(DoG, s, y, x, deriv, H):

    H_inv = cv2.invert(H, flags=cv2.DECOMP_SVD)[1]

    return -H_inv.dot(deriv)


def contrast(DoG, s, y, x, H):

    deriv = derivs(DoG, s, y, x)

    return DoG[s,y,x] + deriv.dot(subpx_step(DoG, s, y, x, deriv, H))


def edge_contrast_kp(DoG, kps):

    ec_kps = []
    e_count = 0
    c_count = 0

    for kp in kps:
        H = localize(DoG,kp[0],kp[1],kp[2])

        if contrast(DoG,kp[0], kp[1], kp[2], H) >= 0.03: # Contrast Threshold from Lowe
            tr = H[0,0] + H[1,1]
            det = H[0,0] * H[1,1] - (H[1,0]**2)
            if (det > 0 and
                tr**2/det < (121/10)): # From Lowe Paper, curve threshold of 10
                ec_kps.append(np.array(kp))

    return np.array(ec_kps)

def calc_grad_mo(ss, y, x):

    dx = ss[y, x+1] - ss[y, x-1]
    dy = ss[y-1, x] - ss[y+1, x]

    o = (math.atan2(dy, dx)+math.pi - 0.001) * 180 / math.pi
    if(o < 0):
        o = o+360
    m = math.sqrt(dx**2 + dy**2)

    return o, m

def orientation(kps, DoG):

    kps_ori = []

    for kp in kps:
        sigma = 1.5*(kp[0]+1)
        size = int((kp[0]+1) * 2)
        hist = np.zeros(10, dtype=np.float32) #orientation histogram

        for py in range(-size//2, size//2):
            for px in range(-size//2, size//2): # orientation patch
                x = kp[2] + px
                y = kp[1] + py

                o, m = calc_grad_mo(DoG[kp[0],:,:], y, x)
                weights = cv2.getGaussianKernel(size*2,sigma)*(cv2.getGaussianKernel(size*2,sigma).T)
                bin = int(np.floor(o)//36)

                hist[bin] += m * weights[py+size,px+size]
        kp_bin = np.argmax(hist)
        kps_ori.append(np.array([kp[0], kp[1], kp[2], kp_bin*36]))
    return kps_ori


def descriptor_hist(ori, ms, os):

    hist = np.zeros(8, dtype=np.float32)
    c = 3.5

    for i, (m, o) in enumerate(zip(ms, os)):
        o = (o - ori) %360
        n = int(np.floor(o)//45)

        weight = 1 - abs(o - 22.5 - (n*45))/22.5
        m *= weight

        x_weight = 1 - max(abs(i%4 - c)/c, 0.001)
        m *= x_weight

        y_weight = 1 - max(abs(i//4 - c)/c, 0.001)
        m *= y_weight

        hist[n] += m

    # hist *= 1/max(la.norm(hist), min_weight)
    # hist *= 1/max(la.norm(hist), min_weight)

    return np.float32(hist)

def descriptor(kps_o, DoG):
    # Per Lowe, 4x4 size subregions
    d_list = []

    for kp in kps_o:
        patch = DoG[int(kp[0]), int(kp[1])-8: int(kp[1])+8, int(kp[2])-8: int(kp[2])+ 8]

        shifted1 = np.zeros((16,16))
        shifted2 = np.zeros((16,16))

        shifted1[0,:] = patch[0,:]
        shifted1[1:,:] = patch[:15,:]
        shifted2[15,:] = patch[15,:]
        shifted2[:15,:] = patch[1:,:]

        dy = np.array(shifted2 - shifted1)

        shifted1[:,0] = patch[:,0]
        shifted1[:,1:] = patch[:,:15]
        shifted2[:,15] = patch[:,15]
        shifted2[:,:15] = patch[:,1:]

        dx = shifted2 - shifted1

        os = np.array(np.arctan2(dy, dx) + math.pi)* 180 / math.pi
        os[os < 0] +=360
        ms = np.array(np.sqrt(dx**2 + dy**2))

        des = np.zeros((8,4,4)).flatten() # 8 bins, 4x4 vector in each

        for i in range(0, 4):
            for j in range(0, 4):
                hist = descriptor_hist(kp[3],
                                       ms[i*4:j*4].flatten(),
                                       os[i*4:j*4].flatten())
                des[i*32 + j*8:i*32 + j*8 + 8] = hist

        # des *= 1/max(la.norm(des), min_weight)
        # des[des>0.2] = 0.2
        # des *= 100/max(la.norm(des), min_weight)

        d_list.append(np.float32(des*100))

    return np.array(d_list)


def format_kps(raw_kps, oct):

    formatted = []

    for kp in raw_kps:
        scale = 2**oct
        formatted.append([kp[2]*scale,
                           kp[1]*scale])

    return formatted


def my_sift(i, threshold):

    img = cv2.GaussianBlur(i,(0,0),1.6,sigmaY=1.6)

    # Scale Space Gaussian Pyramid
    ss_gauss_pyr = get_ss_gauss_pyr(img, 5)

    DoGs = diff_of_gauss(ss_gauss_pyr)

    kp = []
    des = []

    for lvl, DoG in enumerate(DoGs): # lvl = pyramid level

        potential_kps = local_extrema(DoG,threshold) #get positions of local extrema of DoG
        kp_set = edge_contrast_kp(DoG,potential_kps)

        kp_o = orientation(kp_set, DoG)
        kp.extend(format_kps(kp_o, lvl))
        des.extend(descriptor(kp_o, DoG))

    return kp,np.array(des)

def cut_and_save_clips(vid_path, cut_times, vidlen, fr):

    last = 0
    final = 1
    ext = vid_path[vid_path.rindex('.')+1:]
    for n in range(len(cut_times)):
        ffmpeg_extract_subclip(vid_path, last, cut_times[n] - 1/fr, targetname='clip_{0}.{1}'.format(n+1,ext))
        final = n
        last = cut_times[n]

    ffmpeg_extract_subclip(vid_path, last, vidlen, targetname='clip_{0}.{1}'.format(final+1,ext))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--file", dest="filename",
                        help="path to video file",
                        metavar="VIDFILE")
    parser.add_argument("--framerate", dest="fr",
                        help="desired frame rate to process at",
                        metavar="FRAMERATE")
    parser.add_argument("--mincutratio", dest="thresh",
                        help="minimum ratio to cut at, decrease if missing cuts",
                        metavar="MINCUTRATIO")

    args = parser.parse_args()


    filename = args.filename
    fr = int(args.fr)
    thresh = float(args.thresh)

    print('Reading video file...')
    frames = get_vid_frames(filename, fr)
    print('Getting cuts (may a few minutes depending on video length, frame rate)...')
    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    cut_times = []

    kp_old, des_old = sift.detectAndCompute(frames[0],None)

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    for i in range(1, frames.shape[0]):
        kp, des = sift.detectAndCompute(frames[i],None)
        matches = flann.knnMatch(des_old,des,k=2)
        good = 0

        for (m,n) in matches:
            if m.distance < 0.75*n.distance:
                good += 1

        if(len(kp_old)/(good+1) > thresh):
            cut_times.append(i/fr)
        kp_old, des_old = kp, des
    cut_times.append(len(frames)/fr)
    print('saving clips...')
    cut_and_save_clips(filename, cut_times, len(frames)*fr, fr)
