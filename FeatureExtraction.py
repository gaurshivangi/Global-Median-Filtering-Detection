import os
import numpy as np
import cv2
from scipy import stats
from tqdm import tqdm
from cal_o_n_mf_ovrblk_moments import cal_o_n_mf_ovrblk_moments 

#params
win_sz = 3
px_ol = 2
r_req = 64
c_req = 64
h = np.ones((win_sz, win_sz)) / (win_sz * win_sz)

org_directory = './'
mf_directory = './ucid_mf3/'

org_files = [f for f in os.listdir(org_directory) if f.endswith('.tif')]
mf_files = [f for f in os.listdir(mf_directory) if f.endswith('.tif')]

count = len(org_files)
label = np.concatenate([np.ones(count), 2 * np.ones(count)])

skr_1pko = np.zeros(count)
skl_1pko = np.zeros(count)
skr_2pko = np.zeros(count)
skl_2pko = np.zeros(count)
skr_3pko = np.zeros(count)
skl_3pko = np.zeros(count)
skr_1pkmf = np.zeros(count)
skl_1pkmf = np.zeros(count)
skr_2pkmf = np.zeros(count)
skl_2pkmf = np.zeros(count)
skr_3pkmf = np.zeros(count)
skl_3pkmf = np.zeros(count)
skl_4pko = np.zeros(count)
skr_4pko = np.zeros(count)
skl_4pkmf = np.zeros(count)
skr_4pkmf = np.zeros(count)
mid_pko = np.zeros(count)
mid_pkmf = np.zeros(count)
ku_1pko = np.zeros(count)
ku_2pko = np.zeros(count)
ku_3pko = np.zeros(count)
ku_4pko = np.zeros(count)
ku_1pkmf = np.zeros(count)
ku_2pkmf = np.zeros(count)
ku_3pkmf = np.zeros(count)
ku_4pkmf = np.zeros(count)

#arrays for result storage
mean_sko = np.zeros(count)
var_sko = np.zeros(count)
mean_skmf = np.zeros(count)
var_skmf = np.zeros(count)
kurto = np.zeros(count)
kurtmf = np.zeros(count)
n_nansko = np.zeros(count)
n_nanskmf = np.zeros(count)
n_binskew = np.zeros(count)
n_binkurt = np.zeros(count)
mean_kuo = np.zeros(count)
mean_kumf = np.zeros(count)
var_kuo = np.zeros(count)
var_kumf = np.zeros(count)

for t in tqdm(range(count), desc="Processing"):
    print(t)
    p = 1
    q = 1
    l = 1
    
    #original image
    Imo = os.path.join(org_directory, org_files[t])
    Imo_bn = cv2.imread(Imo, cv2.IMREAD_UNCHANGED)
    
    # Convert to grayscale if image is RGB
    if len(Imo_bn.shape) == 3:
        Imo_bn = cv2.cvtColor(Imo_bn, cv2.COLOR_BGR2GRAY)
    
    Imo_b = Imo_bn.astype(float)
    
    #median filtered image
    Immf = os.path.join(mf_directory, mf_files[t])
    Immf_bn = cv2.imread(Immf, cv2.IMREAD_UNCHANGED)
    Immf_b = Immf_bn.astype(float)
    
    (skeworemnan3x3o, skewmf3x3remnan3x3o, kurtoremnan3x3o, kurtmf3x3remnan3x3o, 
     novblk_remnanskew, novblk_remnankurt, n_nanskewo, n_nanskewmf, n_nankurto, 
     n_nankurtmf, chk_skew, chk_kurt) = cal_o_n_mf_ovrblk_moments(Imo_b, Immf_b, win_sz, px_ol)
    
    if chk_skew == 1:
        mean_sko[t] = 0
        var_sko[t] = 0
        mean_skmf[t] = 0
        var_skmf[t] = 0
        kurto[t] = 0
        kurtmf[t] = 0
        mid_pko[t] = 0
        mid_pkmf[t] = 0
        skr_1pko[t] = 0
        skl_1pko[t] = 0
        skr_2pko[t] = 0
        skl_2pko[t] = 0
        skr_3pko[t] = 0
        skl_3pko[t] = 0
        skr_1pkmf[t] = 0
        skl_1pkmf[t] = 0
        skr_2pkmf[t] = 0
        skl_2pkmf[t] = 0
        skr_3pkmf[t] = 0
        skl_3pkmf[t] = 0
        skl_4pko[t] = 0
        skr_4pko[t] = 0
        skl_4pkmf[t] = 0
        skr_4pkmf[t] = 0
        n_nansko[t] = n_nanskewo
    
    if chk_kurt == 1:
        ku_1pko[t] = 0
        ku_2pko[t] = 0
        ku_3pko[t] = 0
        ku_4pko[t] = 0
        ku_1pkmf[t] = 0
        ku_2pkmf[t] = 0
        ku_3pkmf[t] = 0
        ku_4pkmf[t] = 0
        mean_kuo[t] = 0
        mean_kumf[t] = 0
        var_kuo[t] = 0
        var_kumf[t] = 0
    
    if chk_skew == 0 and chk_kurt == 0:
        n_binskew[t] = 1 + np.ceil(np.log2(novblk_remnanskew))
        
        sigma = np.sqrt((6 * (novblk_remnankurt - 2) / (novblk_remnankurt + 1) * (novblk_remnankurt + 3)))
        n_binkurt[t] = 1 + np.ceil(np.log2(novblk_remnankurt) + np.log2(1 + (abs(stats.skew(kurtoremnan3x3o))) / sigma))
        
        if np.isnan(n_binkurt[t]):
            n_binkurt[t] = n_binskew[t]
        
        # Calculate histograms
        n1, x1 = np.histogram(skeworemnan3x3o, int(n_binskew[t]))
        n2, x2 = np.histogram(skewmf3x3remnan3x3o, int(n_binskew[t]))
        n3, x3 = np.histogram(kurtoremnan3x3o, int(n_binkurt[t]))
        n4, x4 = np.histogram(kurtmf3x3remnan3x3o, int(n_binkurt[t]))
        
        # Convert bin edges to bin centers
        x1 = (x1[:-1] + x1[1:]) / 2
        x2 = (x2[:-1] + x2[1:]) / 2
        x3 = (x3[:-1] + x3[1:]) / 2
        x4 = (x4[:-1] + x4[1:]) / 2
        
        if n_binskew[t] == 1:
            N1, edges1 = np.histogram(skeworemnan3x3o, 1)
            N2, edges2 = np.histogram(skewmf3x3remnan3x3o, 1)
            bin_wsko = edges1[1] - edges1[0]
            bin_wskmf = edges2[1] - edges2[0]
        else:
            bin_wsko = x1[-1] - x1[-2] if len(x1) > 1 else 0
            bin_wskmf = x2[-1] - x2[-2] if len(x2) > 1 else 0
            
        if n_binkurt[t] == 1:
            N3, edges3 = np.histogram(kurtoremnan3x3o, 1)
            N4, edges4 = np.histogram(kurtmf3x3remnan3x3o, 1)
            bin_wkuo = edges3[1] - edges3[0]
            bin_wkumf = edges4[1] - edges4[0]
        else:
            bin_wkuo = x3[1] - x3[0] if len(x3) > 1 else 0
            bin_wkumf = x4[1] - x4[0] if len(x4) > 1 else 0
        
        #feature calculations
        mean_sko[t] = np.mean(skeworemnan3x3o)
        mean_skmf[t] = np.mean(skewmf3x3remnan3x3o)
        
        var_sko[t] = np.var(skeworemnan3x3o)
        var_skmf[t] = np.var(skewmf3x3remnan3x3o)
        
        kurto[t] = stats.kurtosis(skeworemnan3x3o)
        kurtmf[t] = stats.kurtosis(skewmf3x3remnan3x3o)
        
        #feature 1 - x2 values check for skewness
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= 2.4748 and x2_lower <= 2.4748:
                skr_4pkmf[t] = n2[i]
        
        #feature 1 - x1 values check for skewness
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= 2.4748 and x1_lower <= 2.4748:
                skr_4pko[t] = n1[i]
        
        #feature 2 - x2 values check
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= -2.4748 and x2_lower <= -2.4748:
                skl_4pkmf[t] = n2[i]
        
        #feature 2 - x1 values check
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= -2.4748 and x1_lower <= -2.4748:
                skl_4pko[t] = n1[i]
        
        #feature 6 - mid point check for x1
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= 0.0000 and x1_lower <= 0.0000:
                mid_pko[t] = n1[i]
        
        #feature 6 - mid point check for x2
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= 0.0000 and x2_lower <= 0.0000:
                mid_pkmf[t] = n2[i]
        
        #feature 7 - check for specific skewness value (1.3363)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= 1.3363 and x2_lower <= 1.3363:
                skr_1pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= 1.3363 and x1_lower <= 1.3363:
                skr_1pko[t] = n1[i]
        
        #feature 8 - check for specific skewness value (-1.3363)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= -1.3363 and x2_lower <= -1.3363:
                skl_1pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= -1.3363 and x1_lower <= -1.3363:
                skl_1pko[t] = n1[i]
        
        #feature 9 - check for specific skewness value (0.7071)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= 0.7071 and x2_lower <= 0.7071:
                skr_2pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= 0.7071 and x1_lower <= 0.7071:
                skr_2pko[t] = n1[i]
        
        #feature 10 - check for specific skewness value (-0.7071)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= -0.7071 and x2_lower <= -0.7071:
                skl_2pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= -0.7071 and x1_lower <= -0.7071:
                skl_2pko[t] = n1[i]
        
        #feature 11 - check for specific skewness value (0.2236)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= 0.2236 and x2_lower <= 0.2236:
                skr_3pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= 0.2236 and x1_lower <= 0.2236:
                skr_3pko[t] = n1[i]
        
        #feature 12 - check for specific skewness value (-0.2236)
        for i in range(len(x2)):
            x2_upper = np.fix((x2[i] + bin_wskmf / 2) * (10**4)) / (10**4)
            x2_lower = np.fix((x2[i] - bin_wskmf / 2) * (10**4)) / (10**4)
            if x2_upper >= -0.2236 and x2_lower <= -0.2236:
                skl_3pkmf[t] = n2[i]
        
        for i in range(len(x1)):
            x1_upper = np.fix((x1[i] + bin_wsko / 2) * (10**4)) / (10**4)
            x1_lower = np.fix((x1[i] - bin_wsko / 2) * (10**4)) / (10**4)
            if x1_upper >= -0.2236 and x1_lower <= -0.2236:
                skl_3pko[t] = n1[i]
        
        #feature 13 - nan counts
        n_nansko[t] = n_nanskewo
        n_nanskmf[t] = n_nanskewmf
        
        #feature 16 - check for specific kurtosis value (2.7857)
        for i in range(len(x4)):
            x4_upper = np.fix((x4[i] + bin_wkumf / 2) * (10**4)) / (10**4)
            x4_lower = np.fix((x4[i] - bin_wkumf / 2) * (10**4)) / (10**4)
            if x4_upper >= 2.7857 and x4_lower <= 2.7857:
                ku_1pkmf[t] = n4[i]
        
        for i in range(len(x3)):
            x3_upper = np.fix((x3[i] + bin_wkuo / 2) * (10**4)) / (10**4)
            x3_lower = np.fix((x3[i] - bin_wkuo / 2) * (10**4)) / (10**4)
            if x3_upper >= 2.7857 and x3_lower <= 2.7857:
                ku_1pko[t] = n3[i]
        
        #feature 15 - check for specific kurtosis value (1.5000)
        for i in range(len(x4)):
            x4_upper = np.fix((x4[i] + bin_wkumf / 2) * (10**4)) / (10**4)
            x4_lower = np.fix((x4[i] - bin_wkumf / 2) * (10**4)) / (10**4)
            if x4_upper >= 1.5000 and x4_lower <= 1.5000:
                ku_2pkmf[t] = n4[i]
        
        for i in range(len(x3)):
            x3_upper = np.fix((x3[i] + bin_wkuo / 2) * (10**4)) / (10**4)
            x3_lower = np.fix((x3[i] - bin_wkuo / 2) * (10**4)) / (10**4)
            if x3_upper >= 1.5000 and x3_lower <= 1.5000:
                ku_2pko[t] = n3[i]
        
        #feature 14 - check for specific kurtosis value (1.0500)
        for i in range(len(x4)):
            x4_upper = np.fix((x4[i] + bin_wkumf / 2) * (10**4)) / (10**4)
            x4_lower = np.fix((x4[i] - bin_wkumf / 2) * (10**4)) / (10**4)
            if x4_upper >= 1.0500 and x4_lower <= 1.0500:
                ku_3pkmf[t] = n4[i]
        
        for i in range(len(x3)):
            x3_upper = np.fix((x3[i] + bin_wkuo / 2) * (10**4)) / (10**4)
            x3_lower = np.fix((x3[i] - bin_wkuo / 2) * (10**4)) / (10**4)
            if x3_upper >= 1.0500 and x3_lower <= 1.0500:
                ku_3pko[t] = n3[i]
        
        #feature 17 - check for specific kurtosis value (7.1249)
        for i in range(len(x4)):
            x4_upper = np.fix((x4[i] + bin_wkumf / 2) * (10**4)) / (10**4)
            x4_lower = np.fix((x4[i] - bin_wkumf / 2) * (10**4)) / (10**4)
            if x4_upper >= 7.1249 and x4_lower <= 7.1249:
                ku_4pkmf[t] = n4[i]
        
        for i in range(len(x3)):
            x3_upper = np.fix((x3[i] + bin_wkuo / 2) * (10**4)) / (10**4)
            x3_lower = np.fix((x3[i] - bin_wkuo / 2) * (10**4)) / (10**4)
            if x3_upper >= 7.1249 and x3_lower <= 7.1249:
                ku_4pko[t] = n3[i]
        
        #feature 18 - mean kurtosis
        mean_kuo[t] = np.mean(kurtoremnan3x3o)
        mean_kumf[t] = np.mean(kurtmf3x3remnan3x3o)
        
        #feature 19 - variance of kurtosis
        var_kuo[t] = np.var(kurtoremnan3x3o)
        var_kumf[t] = np.var(kurtmf3x3remnan3x3o)

#feature matrices
DU32_allvsmf35J = np.vstack([
    np.column_stack([skl_4pko, skr_4pko, mean_sko, var_sko, kurto, mid_pko, skr_1pko, skl_1pko, 
                     skr_2pko, skl_2pko, skr_3pko, skl_3pko, n_nansko, ku_1pko, ku_2pko, 
                     ku_3pko, ku_4pko, mean_kuo, var_kuo]),
    np.column_stack([skl_4pkmf, skr_4pkmf, mean_skmf, var_skmf, kurtmf, mid_pkmf, skr_1pkmf, skl_1pkmf, 
                     skr_2pkmf, skl_2pkmf, skr_3pkmf, skl_3pkmf, n_nanskmf, ku_1pkmf, ku_2pkmf, 
                     ku_3pkmf, ku_4pkmf, mean_kumf, var_kumf])
])

DUL32_allvsmf35J = np.column_stack([DU32_allvsmf35J, label])

#min and max for normalization
min_DU32_allvsmf35J = np.min(DU32_allvsmf35J, axis=0)
max_DUC32_allvsmf35J = np.max(DU32_allvsmf35J, axis=0)

#normalize data to range [-1, 1]
for j in range(2 * count):
    for k in range(DU32_allvsmf35J.shape[1]):
        if max_DUC32_allvsmf35J[k] != min_DU32_allvsmf35J[k]:  
            DU32_allvsmf35J[j, k] = ((2 * (DU32_allvsmf35J[j, k] - min_DU32_allvsmf35J[k])) / 
                                     (max_DUC32_allvsmf35J[k] - min_DU32_allvsmf35J[k])) - 1

# Create final data with labels
DNU32_allvsmf35J = np.column_stack([DU32_allvsmf35J, label])

# Save results
np.save('DNU32_allvsmf35J.npy', DNU32_allvsmf35J)
np.save('DUL32_allvsmf35J.npy', DUL32_allvsmf35J)
np.save('min_DU32_allvsmf35J.npy', min_DU32_allvsmf35J)
np.save('max_DU32_allvsmf35J.npy', max_DUC32_allvsmf35J)
