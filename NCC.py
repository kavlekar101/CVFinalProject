import numpy as np
import ImagePyramid as ip
import cv2

def compute_best_point(template,
                       search,
                       template_pixels,
                       r_std_template,
                       g_std_template,
                       b_std_template,
                       r_template_minus_mean,
                       g_template_minus_mean,
                       b_template_minus_mean):
    
    patches = []
    print(template.shape, search.shape)
    for y in range(0, search.shape[0]-template.shape[0]):
        for x in range(0, search.shape[1]-template.shape[1]):
            if 264 == y and 886 == x:
                print("here")
            patch = search[y:y+template.shape[0], x:x+template.shape[1]]
            
            r_mean_patch = np.mean(patch[:,:,0])
            g_mean_patch = np.mean(patch[:,:,1])
            b_mean_patch = np.mean(patch[:,:,2])

            r_std_patch = np.std(patch[:,:,0])
            g_std_patch = np.std(patch[:,:,1])
            b_std_patch = np.std(patch[:,:,2])
            
            r_patch_minus_mean = patch[:,:,0] - r_mean_patch
            g_patch_minus_mean = patch[:,:,1] - g_mean_patch
            b_patch_minus_mean = patch[:,:,2] - b_mean_patch
            
            normalization = 1/(template_pixels-1)
            
            r_corr = np.sum(r_template_minus_mean*r_patch_minus_mean)/(r_std_template*r_std_patch) * normalization
            g_corr = np.sum(g_template_minus_mean*g_patch_minus_mean)/(g_std_template*g_std_patch) * normalization
            b_corr = np.sum(b_template_minus_mean*b_patch_minus_mean)/(b_std_template*b_std_patch) * normalization
            
            corr = r_corr + g_corr + b_corr
            
            patches.append((corr, y, x, patch))
            
    sorted_patches = sorted(patches, key=lambda x: x[0], reverse=True)
    if len(sorted_patches) < 1:
        return 3, 0, 0, template
    return sorted_patches[0]

def compute_best_point_pyramid(template, search):
    template_pixels = template.shape[0]*template.shape[1]
    
    r_mean_template = np.mean(template[:,:,0])
    g_mean_template = np.mean(template[:,:,1])
    b_mean_template = np.mean(template[:,:,2])

    r_std_template = np.std(template[:,:,0])
    g_std_template = np.std(template[:,:,1])
    b_std_template = np.std(template[:,:,2])
    
    r_template_minus_mean = template[:,:,0] - r_mean_template
    g_template_minus_mean = template[:,:,1] - g_mean_template
    b_template_minus_mean = template[:,:,2] - b_mean_template
    
    best_point = compute_best_point(template,
                                    search,
                                    template_pixels,
                                    r_std_template,
                                    g_std_template,
                                    b_std_template,
                                    r_template_minus_mean,
                                    g_template_minus_mean,
                                    b_template_minus_mean)
    
    return best_point

def pyramid_calculations(template, search):
    pyramid = ip.create_pyramid(search, levels=4)[::-1]
    search_window = [0, 0, pyramid[0].shape[0], pyramid[0].shape[1]]
    
    for i, level in enumerate(pyramid):
        _, y, x, _ = compute_best_point_pyramid(template, level[search_window[0]:search_window[0]+search_window[2], search_window[1]:search_window[1]+search_window[3]])
        
        search = level.copy()
        search[search_window[0]:search_window[0]+search_window[2], search_window[1]:search_window[1]+search_window[3]] = [0, 0, 0]
        cv2.imwrite('level' + str(i) + '.jpg', search)
        
        
        search_window[0] = y * 2
        search_window[1] = x * 2
        search_window[2] = template.shape[0] * 2
        search_window[3] = template.shape[1] * 2
        
    return y, x