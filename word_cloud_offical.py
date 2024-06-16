import os
import cv2
import argparse
import traceback

import numpy as np

from wordcloud import WordCloud

def random_state(seed=None):
    """Generate a random state based on a seed to ensure reproducibility."""
    return np.random.RandomState(seed)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('word', type=str, help='The word to be displayed')
    parser.add_argument('--width', type=int, default=400, help='The width of the image')
    parser.add_argument('--height', type=int, default=800, help='The height of the image')
    parser.add_argument('--nums', type=int, default=100, help='The number of words to be displayed')
    parser.add_argument('--fonts', type=str, default='fonts', help='The root directory of the fonts')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    word, width, height, word_nums, fonts_root = args.word, args.width, args.height, args.nums ,args.fonts

    final_img = np.zeros((height, width, 3), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(word_nums):
        font_name = np.random.choice(os.listdir(fonts_root))
        font_path = os.path.join(fonts_root, font_name)

        try:
            # 生成词云
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color='black',
                repeat=False,
                margin=max(4, int(0.03*min(width, height))),
                mask=mask,
                font_path=font_path,  # 字体路径
                max_font_size=int(0.1*max(width, height)),
                min_font_size=max(4, int(0.04*min(width, height))),
                max_words=3,  # 设置生成词的数量
                scale=1,
                random_state=random_state(),
                prefer_horizontal=width/(width+height),  # 设置水平和垂直单词的比例
            ).generate(word)
            # 词云图片转为opencv格式
            img = wordcloud.to_array()

            _, font_size, _, _, _ = wordcloud.layout_[0]

            # 转换为灰度图像
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            # 构造字体大小五分之一的膨胀核，对字体图应用闭运算
            kernel_size = max(1, font_size // 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
            closed_mask = np.array(closed_img != 0, dtype=np.uint8) * 255

            mask = mask + closed_mask

            final_img = final_img + img
        except IndexError:
            pass
        except Exception:
            traceback.print_exc()
            break

        cv2.imshow("Word Cloud", cv2.hconcat((final_img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
        # cv2.imshow("Word Cloud mask", mask)

        cv2.waitKey(50)

    cv2.imwrite("wordcloud.jpg", final_img)
