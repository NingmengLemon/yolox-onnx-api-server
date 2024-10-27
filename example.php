<?php
// 设置Flask应用的URL
$flaskUrl = 'http://192.168.6.3:9656/predict?vis=1'; // 请替换为你的Flask服务器地址

// 假设$image是你已经有的图像数据，例如从文件或其他来源获取的二进制数据
$image = 'D:\YOLOX\assets\5.png'; // 替换为实际图像的路径
$imageData = file_get_contents($image); // 从文件读取图像数据

// 使用cURL发送POST请求到Flask应用
$ch = curl_init($flaskUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, $imageData);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: image/jpeg', // 根据上传的图像类型调整
    'Content-Length: ' . strlen($imageData),
]);

// 执行请求并获取响应
$response = curl_exec($ch);
if ($response === false) {
    echo 'Curl error: ' . curl_error($ch);
} else {
    // 解析 json
    $response = json_decode($response, true);
    // 保存可视化结果
    file_put_contents('output.jpg', base64_decode($response['vis']));
    // 输出剩余结果
    unset($response['vis']);
    header('Content-Type: application/json');
    echo json_encode($response, JSON_PRETTY_PRINT);
}

curl_close($ch);
?>
