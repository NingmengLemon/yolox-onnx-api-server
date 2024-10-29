# yolox-onnx-api-server

为 yolox 的 onnx 模型提供 API 服务。

## 依赖

```
pip install -r requirements.txt
```

已验证的版本
```
Flask 3.0.3
numpy 1.26.3
onnxruntime 1.19.2
opencv-python 4.9.0.80
```

## 用法

### 最简用法

使用以下命令在 9656 端口上启动 API 服务器。

```
python server.py -m yolox.onnx -l labels.txt
```

### 参数解释

```
usage: yolox-onnx-api-server [-h] -m MODEL -l LABELS [-o OUTPUT_DIR] [-s SCORE_THR] [-i INPUT_SHAPE] [-p PORT]
                             [-k KEY] [-r RATE_LIMIT]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        指定ONNX模型文件。
  -l LABELS, --labels LABELS
                        分类标签文件。
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        可视化图片输出目录。
  -s SCORE_THR, --score_thr SCORE_THR
                        全局置信度阈值。
  -i INPUT_SHAPE, --input_shape INPUT_SHAPE
                        指定推理的输入形状。
  -p PORT, --port PORT  HTTP服务器监听端口。
  -k KEY, --key KEY     API密钥。
  -r RATE_LIMIT, --rate_limit RATE_LIMIT
                        每秒允许的最大请求数
```

## 调用

API 路径: `/predict`

### 查询参数

 - `vis`: 是否返回 base64 编码的可视化结果 (`0` 或 `1`)
 - `key`: API 密钥

### 图片传入

支持以下方法传入图片。

#### 直接POST

设置 `Content-Type` 为 `image` 开头，图片内容直接作为 POST 内容提交。

#### 文件上传

设置表单中名为 `image` 的文件为要上传的图片。

`Content-Type` 通常是 `multipart/form-data`。

#### JSON中的BASE64

设置 `Content-Type` 为 `application/json`

请求体
```
{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAUA... (省略的Base64字符串)"
}
```

## 用例

### API 调用

参考 [example.php](./example.php) 文件。

### MiraiEz 图片过滤

> 支持 mirai-api-http 的 PHP 机器人框架。
> 方便、快速、高效地使用 PHP 编写你自己的 Bot。

[群图片过滤插件](https://github.com/nkxingxh/miraiez-plugins/blob/main/top.nkxingxh.miraiez.yolox.ImageFilter.php)
