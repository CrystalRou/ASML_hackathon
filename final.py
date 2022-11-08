import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tkinter as tk
import tkinter.messagebox as msg
from tkinter import StringVar

def get_result(num, choice):
    
    def loadImage(num):
        sem = cv2.imread(os.path.join('test', 'sem_' + str(num) + '.png'))
        gds = cv2.imread(os.path.join('test', 'gds_' + str(num) + '.png'))
    #     plt.imshow(sem)
    #     plt.imshow(gds)
        return sem, gds
    
    def wafer_defect_detect_function(sem, gds):
        sem_ori = sem
        sem_deno = cv2.fastNlMeansDenoisingColored(sem,None,10,10,7,21) # 去雜訊 
        ret,sem = cv2.threshold(sem_deno,105,255,cv2.THRESH_BINARY)
        sample_image = cv2.GaussianBlur(sem_deno,(13,13),0)
        blur = cv2.add(sample_image,sem)

      # k-means
        twoDimage = blur.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
        K = 3
        attempts=10

        ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((blur.shape))

        buffer = result_image.copy()

      # 輪廓檢測＋去掉小範圍面積
        gray = cv2.cvtColor(buffer,cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,0,100,cv2.THRESH_BINARY)

      #找邊界
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        thresh1 = cv2.drawContours(thresh1, contours, -1, (255,0,0),2)

        mask = buffer.copy()

        define_area = 200
        c_max = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # 处理掉小的轮廓区域，这个区域的大小自己定义。
            if(area < define_area):
                c_min = []
                c_min.append(cnt)
                mask = cv2.drawContours(mask, c_min, -1, (255, 255, 255), thickness=-1)
                continue

        gds = gds
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 5)
        bit_and1 = cv2.bitwise_and(gds,mask)

        gds_inv = ~gds
        sem_inv = ~sem
        g_s_merge = cv2.add(gds_inv,sem_inv)
        kernel = np.ones((2,2), np.uint8)
        g_s_merge = cv2.dilate(g_s_merge, kernel, iterations = 2)

        kernel = np.ones((2,2), np.uint8)
        bit_and1 = cv2.erode(bit_and1, kernel, iterations = 3)
        gray_zone = cv2.bitwise_and(~bit_and1,g_s_merge)
        gray_zone = cv2.morphologyEx(gray_zone, cv2.MORPH_CLOSE, kernel) #先膨脹後侵蝕，去除小暗點

      # 選取灰色瑕疵區域
        lower = np.array([170,170,170])  # 轉換成 NumPy 陣列
        upper = np.array([185,185,185]) # 轉換成 NumPy 陣列
        mask_g = cv2.inRange(gray_zone, lower, upper)  # 使用 inRange
        gray_zone_output = cv2.bitwise_and(gray_zone, gray_zone, mask = mask_g )  # 套用影像遮罩
        kernel = np.ones((7,7), np.uint8)
        gray_zone_output = cv2.morphologyEx(gray_zone_output, cv2.MORPH_OPEN, kernel)

        gray = cv2.cvtColor(gray_zone_output,cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,0,100,cv2.THRESH_BINARY)

      #找邊界
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        thresh1 = cv2.drawContours(thresh1, contours, -1, (255,0,0),2)

        mask = gray_zone_output.copy()

        define_area = 85
        c_max = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            #print(area)
            # 处理掉小的轮廓区域，这个区域的大小自己定义。
            if(area < define_area):
                c_min = []
                c_min.append(cnt)
                mask = cv2.drawContours(mask, c_min, -1, (0, 0, 0), thickness=-1)
                continue

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 12)

        gray_zone_edge = cv2.Canny(mask,100,200)
        gray_zone_edge = cv2.dilate(gray_zone_edge, kernel, iterations = 3)
        gray_zone_edge = cv2.cvtColor(gray_zone_edge, cv2.COLOR_GRAY2BGR)
        gray_zone_edge[:,:,0] = 0

        final_edge = cv2.Canny(mask, 100, 200)
        contours, hierarchy = cv2.findContours(final_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(final_edge, contours, -1, (255, 0, 0), 3)
        final_edge_nonbox = cv2.cvtColor(final_edge, cv2.COLOR_GRAY2BGR)

        final_edge_box = gray_zone_edge.copy()
        img_black = np.zeros((1024,1024,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        img_black[0:1024, 0:1024] = [0,0,0]
        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            cv2.rectangle(img_black,(x,y),(x+w,y+h),(255,0,0),2)  
    #     plt.imshow(img_black)
        return img_black
    
    def inv_detect_wafer_defect(img1,img2):
        img2_f=cv2.fastNlMeansDenoisingColored(img2,None,10,10,7,21)
        ret,img2_thres = cv2.threshold(img2_f,105,255,cv2.THRESH_BINARY)
    #     plt.subplot(121),plt.imshow(img2_f)
        sample_image = cv2.GaussianBlur(img2_f,(13,13),0)
        pink = cv2.add(sample_image,img2_thres)
    #     plt.subplot(121),plt.imshow(pink)
        img = cv2.cvtColor(pink,cv2.COLOR_BGR2RGB)
    #     plt.imshow(img)
        twoDimage = img.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        attempts=10
        ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))
    #     plt.axis('off')
    #     plt.imshow(result_image)
        #green
        green = np.zeros((1024,1024,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        green[0:1024, 0:1024] = [0,255,0]  # 將中間 200x200 的每個項目內容，改為 [0,0,255]
        lower_l = np.array([75,75,75])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper_l = np.array([85,85,85]) # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        mask_g_l = cv2.inRange(result_image, lower_l, upper_l)           # 使用 inRange
        output_l = cv2.bitwise_and(result_image, result_image, mask = mask_g_l )  # 套用影像遮罩
        output_l = ~output_l
    #     plt.subplot(121),plt.imshow(output_l)
        ret,img2_buffer = cv2.threshold(output_l,200,255,cv2.THRESH_BINARY)
    #     green = cv2.imread(r'.\green.png')
        output_l = cv2.add(img2_buffer,green)
        output_l = ~output_l
    #     plt.subplot(122),plt.imshow(output_l)
        #red
        red = np.zeros((1024,1024,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        red[0:1024, 0:1024] = [0,0,255]  # 將中間 200x200 的每個項目內容，改為 [0,0,255]
        lower_h = np.array([40,40,40])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper_h = np.array([50,50,50]) # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        mask_g_h = cv2.inRange(result_image, lower_h, upper_h)           # 使用 inRange
        output_h = cv2.bitwise_and(result_image, result_image, mask = mask_g_h )  # 套用影像遮罩
        output_h = ~output_h
    #     plt.subplot(121),plt.imshow(output_h)
        ret,img2_buffer = cv2.threshold(output_h,220,255,cv2.THRESH_BINARY)
    #     plt.subplot(122),plt.imshow(img2)
        # red = cv2.imread(r'.\red.png')
        output_h = cv2.add(img2_buffer,red)
        output_h = ~output_h
    #     plt.subplot(122),plt.imshow(output_h)
        lower_w = np.array([255,255,255])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper_w = np.array([255,255,255]) # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        mask_g_w = cv2.inRange(result_image, lower_w, upper_w)           # 使用 inRange
        output_w = cv2.bitwise_and(result_image, result_image, mask = mask_g_w )  # 套用影像遮罩
    #     plt.subplot(121),plt.imshow(output_w)
        three_channel = cv2.add(output_l, output_h)
        three_channel = cv2.add(three_channel, output_w)
    #     plt.imshow(three_channel)
        total = cv2.bitwise_and(three_channel,~img1)
    #     plt.axis('off')
    #     plt.imshow(total)
        lower = np.array([255,255,255])
        upper = np.array([255,255,255])
        mask_g = cv2.inRange(total,lower,upper)
        output=cv2.bitwise_and(total,total,mask=mask_g)
    #     plt.subplot(121),plt.imshow(output)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(output, kernel, iterations=7)
        kernel = np.ones((2,2),np.uint8)
        dilation_w = cv2.dilate(erosion, kernel, iterations=6)
    #     plt.subplot(122),plt.imshow(dilation_w)
        #buffer_img
        img_b = np.zeros((1024,1024,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        img_b[12:1012, 12:1012] = [255,255,255]  # 將中間 200x200 的每個項目內容，改為 [0,0,255]
    #     plt.subplot(121),plt.imshow(img_b)
        new = cv2.bitwise_and(img_b,dilation_w)
    #     plt.subplot(121),plt.imshow(new)
        kernel = np.ones((3,3),np.uint8)
        new = cv2.dilate(new, kernel, iterations=10)
    #     plt.subplot(121),plt.imshow(new)
        #find contour
        img_black = np.zeros((1024,1024,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        img_black[0:1024, 0:1024] = [0,0,0]   
        edges2 = cv2.Canny(new,100,200)
        contours, hierarchy = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            # get rectangle bounding contour
            [x,y,w,h] = cv2.boundingRect(contour)
            cv2.rectangle(img_black,(x,y),(x+w,y+h),(255,0,0),2)
    #     plt.subplot(121),plt.imshow(img_black)
        return img_black
    
    def image_show_three(img1,img2,img3):
        buffer = np.zeros((1024,30,3), dtype='uint8')   # 快速產生 500x500，每個項目為 [0,0,0] 的三維陣列
        buffer[0:1024, 0:1024] = [255,255,255]  # 將中間 200x200 的每個項目內容，改為 [0,0,255]

        h1, w1 = img1.shape[:2]
        h_buf, w_buf = buffer.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]

        img_3 = np.zeros((max(h1, h2, h_buf, h3), w1 + w_buf + w2 + w_buf + w3,3), dtype=np.uint8)
        img_3[:,:] = (255,255,255)

        img_3[:h1, :w1,:3] = img1
        img_3[:h_buf, w1:w1+w_buf,:3] = buffer
        img_3[:h2, w1+w_buf:w1+w_buf+w2,:3] = img2
        img_3[:h_buf, w1+w_buf+w2:w1+w_buf+w2+w_buf,:3] = buffer
        img_3[:h_buf, w1+w_buf+w2+w_buf:w1+w_buf+w2+w_buf+w3,:3] = img3

        cv2.namedWindow('Wafer Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Wafer Image", 1636, 512)
        cv2.imshow('Wafer Image', img_3)
        
        if choice == 1:
            cv2.waitKey(0)
        else:
            cv2.waitKey(10000)
        cv2.destroyAllWindows()
    
    td, gt = loadImage(num)
    pos = wafer_defect_detect_function(td, gt)
    neg = inv_detect_wafer_defect(gt,td)
    bounding_box = cv2.add(pos, neg)
    tmp = cv2.add(bounding_box, td)
    result = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    image_show_three(gt,td,result)
#     return result


def define_img_testing():
#     num = 2
    num = str(img_num.get())
#     print(img_num.get())
    get_result(num,1)

def rotate_img_testing():
    total_pic = int(len(os.listdir('test'))/2)
    for i in range(total_pic):
#         print(i)
        get_result(str(i+1),2)

win = tk.Tk()
win.title("GROUP5_DEMO")

win.geometry('540x40')
win.resizable(0,0)
# 将俩个标签分别布置在第一行、第二行
tk.Label(win, text="Input the wafer image number : ",font = ('Arial', 16)).grid(row=0, column=0)

# 创建输入框控件
img_num = tk.StringVar()
e1 = tk.Entry(win, textvariable=img_num)
e1.grid(row=0, column=1, padx=10, pady=5)

# 使用 grid()的函数来布局，并控制按钮的显示位置
tk.Button(win, text="Confirm", width=8, command=define_img_testing, font = ('Arial', 12)).grid(row=0, column=2, sticky="e")
# tk.Button(win, text="Wafer image rotate in turn", width=20, command=rotate_img_testing,font = ('Arial', 14)).grid(row=1, column=0, columnspan=3, sticky="ew")
win.mainloop()
