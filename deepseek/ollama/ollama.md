## Ollama 安装部署与服务发布

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

[手动安装说明](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### macOS

[下载](https://ollama.com/download/Ollama-darwin.zip)

### Windows 预览版

[下载](https://ollama.com/download/OllamaSetup.exe)

---

## 快速入门

要运行并与 [DeepSeek-R1-7B](https://ollama.com/library/deepseek-r1) 进行对话：

```bash
ollama run deepseek-r1
```

---

## 模型库

Ollama 支持在 [ollama.com/library](https://ollama.com/library) 上提供的一系列模型。

以下是一些可以下载的示例模型：

| 模型               | 参数  | 大小  | 下载命令                       |
| ------------------ | ---------- | ----- | -------------------------------- |
| Gemma 3            | 1B         | 815MB | `ollama run gemma3:1b`           |
| Gemma 3            | 4B         | 3.3GB | `ollama run gemma3`              |
| Gemma 3            | 12B        | 8.1GB | `ollama run gemma3:12b`          |
| Gemma 3            | 27B        | 17GB  | `ollama run gemma3:27b`          |
| QwQ                | 32B        | 20GB  | `ollama run qwq`                 |
| DeepSeek-R1        | 7B         | 4.7GB | `ollama run deepseek-r1`         |
| DeepSeek-R1        | 671B       | 404GB | `ollama run deepseek-r1:671b`    |
| Llama 4            | 109B       | 67GB  | `ollama run llama4:scout`        |
| Llama 4            | 400B       | 245GB | `ollama run llama4:maverick`     |
| Llama 3.3          | 70B        | 43GB  | `ollama run llama3.3`            |
| Llama 3.2          | 3B         | 2.0GB | `ollama run llama3.2`            |
| Llama 3.2          | 1B         | 1.3GB | `ollama run llama3.2:1b`         |
| Llama 3.2 Vision   | 11B        | 7.9GB | `ollama run llama3.2-vision`     |
| Llama 3.2 Vision   | 90B        | 55GB  | `ollama run llama3.2-vision:90b` |
| Phi 4              | 14B        | 9.1GB | `ollama run phi4`                |
| Phi 4 Mini         | 3.8B       | 2.5GB | `ollama run phi4-mini`           |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`             |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`           |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`         |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`         |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`           |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored`   |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`               |
| Granite-3.3         | 8B         | 4.9GB | `ollama run granite3.3`          |

---

### 命令行工具

#### 创建模型

`ollama create` 用于从 Modelfile 创建模型。

```bash
ollama create mymodel -f ./Modelfile
```

#### 拉取模型

```bash
ollama pull deepseek-r1
```

> 此命令还可用于更新本地模型。仅会拉取差异部分。

#### 删除模型

```bash
ollama rm deepseek-r1
```

#### 复制模型

```bash
ollama cp deepseek-r1 my-model
```

#### 多行输入

对于多行输入，可以使用 `"""` 包裹文本：

```bash
>>> """Hello,
... world!
... """
```
这将输出一个包含“Hello, world!”消息的简单程序。

#### 多模态模型

```bash
ollama run llava "这张图片中有什么？ /Users/jmorgan/Desktop/smile.png"
```
图像中显示的是一个黄色的笑脸，可能是图片的中心焦点。

#### 以参数传递提示

```bash
$ ollama run deepseek-r1 "总结此文件: $(cat README.md)"
```
Ollama 是一个轻量级、可扩展的框架，用于在本地计算机上构建和运行语言模型。

---

### REST API

Ollama 提供 REST API 来运行和管理模型。

#### 生成响应

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1",
  "prompt":"为什么天空是蓝色的？"
}'
```

#### 与模型对话

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1",
  "messages": [
    { "role": "user", "content": "为什么天空是蓝色的？" }
  ]
}'
```

有关所有端点（Endpoint）的详细信息，请参阅 [API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)。

---

### Docker 支持

Ollama 官方提供了 Docker 镜像 `ollama/ollama`，可以在 Docker Hub 上找到。

#### 使用 CPU 运行

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### 使用 Nvidia GPU 运行

要使用 Nvidia GPU，首先需要安装 NVIDIA Container Toolkit：

```bash
# 配置仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

# 安装 NVIDIA Container Toolkit 包
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 使用 Nvidia 驱动
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

启动容器：

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### 使用 AMD GPU 运行

要使用 AMD GPU 运行 Ollama，可以使用 `rocm` 标签，并运行以下命令：

```bash
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

### 本地运行模型

现在，你可以运行一个模型：

```bash
docker exec -it ollama ollama run llama3
```

---

请根据以上内容进行 Ollama 的安装和配置，使用 CLI 工具和 Docker 镜像来管理和运行各种模型。如需更多信息，请访问 [Ollama GitHub 仓库](https://github.com/ollama/ollama)。
