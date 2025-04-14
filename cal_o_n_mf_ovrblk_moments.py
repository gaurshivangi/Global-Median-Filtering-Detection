import numpy as np
from scipy import stats
from skimage.util import view_as_windows

def cal_o_n_mf_ovrblk_moments(X, Xmf, win_sz, px_ol):
   
    def pad_to_odd(im):
        h, w = im.shape
        if h % 2 == 0:
            im = np.vstack([im, np.zeros((1, w))])
        if w % 2 == 0:
            im = np.hstack([im, np.zeros((im.shape[0], 1))])
        return im

    if px_ol == 1:
        X = pad_to_odd(X)
        Xmf = pad_to_odd(Xmf)
        step = win_sz - 1
    elif px_ol == 2:
        step = win_sz - 2
        # don't pad for 2-px overlap

    #overlapping blocks using view_as_windows
    blocks_o = view_as_windows(X, (win_sz, win_sz), step)
    blocks_mf = view_as_windows(Xmf, (win_sz, win_sz), step)

    n_blocks = blocks_o.shape[0] * blocks_o.shape[1]
    blocks_o_flat = blocks_o.reshape(n_blocks, -1)
    blocks_mf_flat = blocks_mf.reshape(n_blocks, -1)

    #skewness and kurtosis in batch
    skew_o = stats.skew(blocks_o_flat, axis=1, nan_policy='omit')
    skew_mf = stats.skew(blocks_mf_flat, axis=1, nan_policy='omit')
    kurt_o = stats.kurtosis(blocks_o_flat, axis=1, nan_policy='omit')
    kurt_mf = stats.kurtosis(blocks_mf_flat, axis=1, nan_policy='omit')

    valid_skew = ~np.isnan(skew_o) & ~np.isnan(skew_mf)
    valid_kurt = ~np.isnan(kurt_o) & ~np.isnan(kurt_mf)

    #count NaNs
    n_nanskewo = np.isnan(skew_o).sum()
    n_nanskewmf = np.isnan(skew_mf).sum()
    n_nankurto = np.isnan(kurt_o).sum()
    n_nankurtmf = np.isnan(kurt_mf).sum()

    if np.all(np.isnan(skew_o)) or np.all(np.isnan(skew_mf)):
        skeworemnan_ovblk = skewmfremnan_ovblk = np.ones(n_blocks)
        n_ovblk_remnanskew = 0
        chk_skew = 1
    else:
        skeworemnan_ovblk = skew_o[valid_skew]
        skewmfremnan_ovblk = skew_mf[valid_skew]
        n_ovblk_remnanskew = len(skeworemnan_ovblk)
        chk_skew = 0

    if np.all(np.isnan(kurt_o)) or np.all(np.isnan(kurt_mf)):
        kurtoremnan_ovblk = kurtmfremnan_ovblk = np.ones(n_blocks)
        n_ovblk_remnankurt = 0
        chk_kurt = 1
    else:
        kurtoremnan_ovblk = kurt_o[valid_kurt]
        kurtmfremnan_ovblk = kurt_mf[valid_kurt]
        n_ovblk_remnankurt = len(kurtoremnan_ovblk)
        chk_kurt = 0

    return (skeworemnan_ovblk, skewmfremnan_ovblk,
            kurtoremnan_ovblk, kurtmfremnan_ovblk,
            n_ovblk_remnanskew, n_ovblk_remnankurt,
            n_nanskewo, n_nanskewmf, n_nankurto, n_nankurtmf,
            chk_skew, chk_kurt)

