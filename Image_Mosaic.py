#%%

import numpy as np 
import matplotlib.pyplot as plt
import cv2

def change_to_homogenous_coordinates(p):
    homogenous_points = []

    ones_vec = np.ones((p.shape[0], 1))
    homogenous_points = np.hstack((p, ones_vec))
    
    return homogenous_points

def getCorrespondences(img1_url, img2_url, n=4):
    if n < 4:
        raise Exception("n must be >= 4")
        
    img=plt.imread(img1_url)
    imgplot = plt.imshow(img, cmap='gray')
    points1 = plt.ginput(n)
    # plt.show()
    img=plt.imread(img2_url)
    imgplot = plt.imshow(img, cmap='gray')
    points2 = plt.ginput(n)
    # plt.show()
    points1 = np.array(points1)
    points2 = np.array(points2)
    return points1, points2


def create_H_matrix(points1, points2):
    A = []
    B = []
    for a, b in zip(points1, points2):
        # A = 
        #[ [x, y, 1, 0, 0, 0, -xx',-yx'],
        # [0, 0, 0, x, y, 1, -xy', -yy'] ]
        A.append(np.array([a[0],a[1],1,0,0,0,-a[0]*b[0], -a[1]*b[0]]))
        A.append(np.array([0,0,0,a[0],a[1],1,-a[0]*b[1], -a[1]*b[1]]))
        
        B.append(np.array(b[0]))
        B.append(np.array(b[1]))

    A = np.array(A)
    B = np.array(B)
    H = np.linalg.lstsq(A, B, rcond=None)[0]
    H = np.hstack((H, [1.0]))
    H = np.reshape(H, (3,3))

    return H


def transform_points(points1, H):
    points1_homogenous = change_to_homogenous_coordinates(points1)

    points2_transformed = H @ points1_homogenous.T

    heterogenous_coordinates = []
    for row in points2_transformed.T:
        heterogenous_coordinates.append([row[0]/row[2], row[1]/row[2]])
        
    heterogenous_coordinates = np.array(heterogenous_coordinates)
    return heterogenous_coordinates

def verify_H(img2_url, points1, H):
    img = plt.imread(img2_url)
    plt.imshow(img, cmap='gray')
    transformed_points2 = transform_points(points1, H)
    for point in transformed_points2:
        plt.scatter(point[0], point[1], s=20)
    plt.show()
    


def getCornerPositinos(image, H):
    corner1 = H*np.mat([[0], [0], [1]])
    corner1 = corner1/corner1[-1]
    corner2 = H*np.mat([image.shape[0], image.shape[1], 1]).T
    corner2 = corner2/corner2[-1]
    corner3 = np.matmul(H,np.mat([0, image.shape[1], 1]).T)
    corner3 = corner3/corner3[-1]
    corner4 = np.matmul(H,np.mat([image.shape[0], 0, 1]).T)
    corner4 = corner4/corner4[-1]
    return corner1[:-1], corner2[:-1], corner3[:-1], corner4[:-1]

def warp_image(image_path, H, corners):
    image = cv2.imread(image_path)
    image = np.transpose(image, [1,0,2])
    corners = getCornerPositinos(image, H)
    maxs = np.max(corners, axis=0)
    mins = np.min(corners, axis=0)
    dims = np.round(maxs-mins).T[0]
    new_image = np.zeros((image.shape[0]*2, image.shape[1]*2, 3),dtype=np.uint8)
    new_image_inter = warp_image_backward(H, image, mins, dims)
    # # out = cv2.warpPerspective(image,H,(int(dims[1]), int(dims[1])),flags=cv2.INTER_LINEAR)
    # # plt.imshow(new_image)
    # # plt.show()
    # # new_image=to_img(new_image)
    # new_image = np.transpose(new_image, [1,0,2])

    negative_mins = np.where(mins < 0)
    offsets = np.zeros_like(mins)
    offsets[negative_mins] = mins[negative_mins] * -1
    mins[negative_mins] = 0
    new_image[int(mins[0]):int(mins[0]) + int(dims[0]), int(mins[1]):int(mins[1]) + int(dims[1])] = new_image_inter
    new_image = np.transpose(new_image, [1,0,2])

    cv2.imshow("warp", new_image)
    cv2.waitKey(0)
    cv2.imwrite("image3.jpg", new_image)
    return new_image, offsets

def is_valid_index(image_shape, idx):
    condition_1 = idx[0] < image_shape[0]
    condition_2 = idx[1] < image_shape[1]
    condition_3 = idx[0] >= 0
    condition_4 = idx[1] >= 0
    return condition_1 and condition_2 and condition_3 and condition_4

def check_4_points_are_valid(image_shape, x1, x2, y1, y2):
    condition_1 = is_valid_index(image_shape, [x1, y1])
    condition_2 = is_valid_index(image_shape, [x2, y1])
    condition_3 = is_valid_index(image_shape, [x1, y2])
    condition_4 = is_valid_index(image_shape, [x2, y2])
    return condition_1 and condition_2 and condition_3 and condition_4

def calculate_interpolation_weights(x, y, x1, x2, y1, y2):
    w1 = (x2 - x) * (y2 - y)
    w1 = np.repeat(w1, 3)
    w2 = (x - x1) * (y2 - y)
    w2 = np.repeat(w2, 3)
    w3 = (x2 - x) * (y - y1)
    w3 = np.repeat(w3, 3)
    w4 = (x - x1) * (y - y1)
    w4 = np.repeat(w4, 3)
    return w1, w2, w3, w4

def interpolate(image, idxs_old):
    x, y = idxs_old[0], idxs_old[1]
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1
    if check_4_points_are_valid(image.shape, x1, x2, y1, y2):
        w1, w2, w3, w4 = calculate_interpolation_weights(x, y, x1, x2, y1, y2)
        new_pixel_intensity = w1 * image[x1, y1] + w2 * image[x2, y1] + w3 * image[x1, y2] + w4 * image[x2, y2]
        return (new_pixel_intensity).astype(int)
    return 0

def warp_image_backward(H:np.mat, image:np.array, mins, dims):
    new_image = np.zeros((int(dims[0]), int(dims[1]), 3), dtype=np.uint8)
    Hinv = np.linalg.inv(H)
    x = np.array([mins[0], mins[1], [0]])

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            idxs_new = np.array([i,j,1])
            uu = idxs_new+x.T[0]
            idxs_old = np.dot(Hinv, idxs_new+x.T[0])
            idxs_old = idxs_old/idxs_old[-1]
            if idxs_old[0] % 1 == 0 and idxs_old[1] % 1 == 0: # if not a decimal
                if is_valid_index(image.shape, idxs_old[:-1]):
                    new_image_inter[i, j] = image[idxs_old[0], idxs]
            else:
                new_intensity = interpolate(image, idxs_old/idxs_old[-1])
                new_image[i, j] = new_intensity
            # idxs_old=(idxs_old/idxs_old[-1]).astype(int)#should interpolate not just round
            # # print(idxs_old)
            # if(is_valid_index(image.shape, idxs_old)):
                # new_image[i, j] = image[idxs_old[0], idxs_old[1]]
    return new_image


def stitch(warped_img, image2_url, offsets):
    img2 = cv2.imread(image2_url)
    warped_img[int(offsets[1]):int(offsets[1]) + img2.shape[0], int(offsets[0]):int(offsets[0]) + img2.shape[1]] = img2
    return warped_img

def crop(mosaic):
    rows, cols = np.where(mosaic[:,:,1] > 0)
    xmin = min(rows)
    ymin = min(cols)
    xmax = max(rows)
    ymax = max(cols)
    mosaic = mosaic[xmin:xmax, ymin:ymax]
    return mosaic


def get_mosaic(image1_url, image2_url, out_url, n=4):
    # Assignment 1.1
    p, p_dash = getCorrespondences(image1_url, image2_url, n=n)
    # print(p)
    # print(p_dash)
    

    # Assignment 1.2
    H = create_H_matrix(p, p_dash)
    verify_H(image2_url, p, H)


    # Assignment 1.3
    warped_img, offsets = warp_image(image1_url, H, p)

  
    # Assignment 1.4
    mosaic = stitch(warped_img, image2_url, offsets)
    mosaic = crop(mosaic)
    cv2.imshow("result", mosaic)
    cv2.waitKey(0)
    cv2.imwrite(out_url, mosaic)
    return mosaic


if __name__ == '__main__':
    image1_url = './images/image1.jpg'
    image2_url = './images/image2.jpg'
    out_url = './outputs/result1.jpg'
    mosaic = get_mosaic(image1_url, image2_url, out_url, n=6)
    

    # Bonus
    image1_url = './images/shanghai-23.png'
    image2_url = './images/shanghai-21.png'
    out_url = './outputs/result21.jpg'
    first_mosaic = get_mosaic(image1_url, image2_url, out_url, n=7)


    image1_url = './outputs/result21.jpg'
    image2_url = './images/shanghai-22.png'
    out_url = './outputs/result22.jpg'
    final_mosaic = get_mosaic(image1_url, image2_url, out_url, n=7)





# %%

    ## GOOD POINTS TO TRY ##

    ## Main Example

    # p = np.array([[375.60923809, 291.42314087],
    #               [527.32289694, 309.10755509],
    #               [477.0619302,  359.36852182],
    #               [278.81033919, 228.13155313],
    #               [324.41751271, 504.56687017],
    #               [451.00068819, 468.26728308]])

    # p_dash = np.array([[814.9273177,  319.34590016],
    #                    [981.53311484, 325.86121067],
    #                    [930.34138946, 385.42976384],
    #                    [717.19766016, 261.63886428],
    #                    [780.4892479,  537.14342268],
    #                    [909.86469931, 494.3285251 ]])



    ## Bonus First Mosaic

    # p = np.array([[650.18303784, 182.52437961],
    #               [594.33751925, 308.17679645],
    #               [545.9380698,  367.74534961],
    #               [581.30689824, 459.89045529],
    #               [485.43875799, 359.36852182],
    #               [143.85033592, 159.25541353],
    #               [ 69.38964446, 333.30727981]])

    # p_dash = np.array([[991.77145991, 174.14755182],
    #                    [937.78745861, 309.10755509],
    #                    [889.38800916, 373.32990147],
    #                    [931.2721481,  466.4057658 ],
    #                    [827.02718006, 364.02231504],
    #                    [477.99268884, 184.38589689],
    #                    [415.63185975, 354.71472861]])



    ## Bonus Second Mosaic

    # p = np.array([[981.3509696,  213.88050722],
    #               [938.19538436, 361.84251373],
    #               [828.25139341, 416.30075224],
    #               [703.92220738, 363.8975416 ],
    #               [477.86914188, 235.45829983],
    #               [270.31132718, 473.84153255],
    #               [268.25629931, 521.10717352]])

    # p_dash = np.array([[1009.45587414,  357.50700454],
    #                    [ 970.36401112,  505.49762881],
    #                    [ 860.53449122,  562.27390605],
    #                    [ 734.88207438,  509.22066339],
    #                    [ 508.70772407,  378.91445333],
    #                    [ 295.56399477,  616.25790736],
    #                    [ 292.77171884,  663.72659817]])

#%%