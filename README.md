# 2022 新竹 $\times$ 梅竹黑客松

ASML 艾司摩爾 第五組 好好組隊 final.py為最終黑客松競賽程式碼。

<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3-yellow" height=22.5></a>
<a href="https://opencv.org/opencv-4-6-0/"><img src="https://img.shields.io/badge/OpenCV-4.6.0-orange" height=22.5></a>

## Overview

![](https://i.imgur.com/QkBngG9.png)

## Preprocessing

微影原圖雜訊較多，因此在做偵測前，必須先對圖片做 NLM,  Gaussion blur, Binarization 等去雜訊方法

## Part 1 - Detect

利用微影原圖與設計圖做影像處理後採疊圖方式，偵測出瑕疵區域

- Case 1: 取出微影**凹陷**部分
- Case 2: 取出微影**凸出**部分

## Part 2 - Mark

合併 case1 與 case2 各自產生的瑕疵部分，框成 Bounding Box 疊在微影圖原圖上，標出瑕疵部位

## Skill

![](https://i.imgur.com/d44Y85q.png)
