# sd-webui-cloth-segmentation
此存储库用于从人类肖像进行布料解析。\
这里的衣服被解析为3类：上半身，下半身和全身且都是以蒙版形式展示。

该库是利用 [cloth-segmentation](https://github.com/levindabhi/cloth-segmentation/tree/main)开源库来实现的SD功能

### 安装方法 Install
- 拷贝项目的git地址：https://github.com/Kevin-coder80/sd-webui-cloth-segmentation.git 到 stable diffusion 中的 “扩展 -> 从网址安装“，将git地址复制到第一个输入框中进行安装
- 下载 [cloth_segm_u2net_latest.pth](https://huggingface.co/spaces/sidharthism/fashion-eye-try-on/resolve/main/cloth_segmentation/checkpoints/cloth_segm_u2net_latest.pth) 模型到 “stable-diffusion-webui.repositories.clothSegmentation.trained_checkpoint”目录中
- 重启sd服务，在“附加功能”中会有出现“Cloth segmentation"的功能区域，说明安装成功

### 使用方法 How to use?
该功能分为三部分，分别是：Upper body(上半身)，Lower body(下半身)和Full body(全身)。\
在使用时，可以分别选择其中一个后会进行图片的生成，该插件会自动根据衣服进行处理，最后生成相应衣服的蒙版。
