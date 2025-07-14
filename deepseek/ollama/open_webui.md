# 在 Ubuntu 22.04 上使用 Docker 安装 Open WebUI

## 前提条件：安装 Docker

在安装 Open WebUI 之前，请确保已安装 Docker。以下是安装 Docker 的步骤：

```bash
# 更新软件包索引
sudo apt-get update

# 安装依赖包
sudo apt-get install -y ca-certificates curl

# 安装 Docker
sudo apt-get install -y docker.io

# 验证 Docker 是否安装成功
sudo docker run hello-world
```

## 前提条件：安装 NVIDIA Container Toolkit

```bash
# 添加软件源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新软件包列表
sudo apt-get update

# 安装 NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 使用 NVIDIA 运行时
sudo nvidia-ctk runtime configure --runtime=docker

# 重启 Docker 服务
sudo systemctl restart docker
```


## 安装 Open WebUI

### 默认配置（Ollama 在同一台计算机上）

如果 Ollama 服务运行在同一台计算机上，使用以下命令：

```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### Ollama 在另一台服务器上

要连接到另一台服务器上的 Ollama，请将 `OLLAMA_BASE_URL` 环境变量设置为该服务器的 URL：

```bash
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=https://example.com \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### [重点关注]使用 Nvidia GPU 支持运行 Open WebUI

要使用 Nvidia GPU 支持运行 Open WebUI，请使用以下命令：

```bash
docker run -d -p 3000:8080 \
  --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:cuda
```

### 仅使用 API

如果您仅使用 OpenAI API，可以使用以下命令（需要提供您的 OpenAI API 密钥）：

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_KEY=your_secret_key \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### [重点关注]安装捆绑 Ollama 的 Open WebUI（一体式安装）

这种方法使用一个容器镜像，该镜像捆绑了 Open WebUI 和 Ollama，允许通过单个命令进行设置。根据您的硬件配置选择适当的命令：

#### 使用 GPU

```bash
docker run -d -p 3000:8080 \
  --gpus=all \
  -v ollama:/root/.ollama \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:ollama
```

#### 仅使用 CPU

```bash
docker run -d -p 3000:8080 \
  -v ollama:/root/.ollama \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:ollama
```

## 访问 Open WebUI

安装完成后，您可以通过 [http://localhost:3000](http://localhost:3000) 访问 Open WebUI。

## 常用 Docker 命令

### 查看运行中的容器
```bash
docker ps
```

### 停止容器
```bash
docker stop open-webui
```

### 启动容器
```bash
docker start open-webui
```

### 重启容器
```bash
docker restart open-webui
```

### 查看容器日志
```bash
docker logs open-webui
```

### 删除容器
```bash
docker rm open-webui
```

### 删除数据卷
```bash
docker volume rm open-webui
docker volume rm ollama  # 如果使用了捆绑安装
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `-d` | 在后台运行容器（守护进程模式） |
| `-p 3000:8080` | 将容器的 8080 端口映射到主机的 3000 端口 |
| `--add-host=host.docker.internal:host-gateway` | 允许容器访问主机服务 |
| `-v open-webui:/app/backend/data` | 创建名为 open-webui 的卷并挂载到容器 |
| `-v ollama:/root/.ollama` | 创建名为 ollama 的卷用于存储模型 |
| `--name open-webui` | 指定容器名称 |
| `--restart always` | 设置容器总是自动重启 |
| `--gpus all` | 允许容器使用所有 GPU |
| `-e OLLAMA_BASE_URL=...` | 设置 Ollama 服务地址 |
| `-e OPENAI_API_KEY=...` | 设置 OpenAI API 密钥 |

## 故障排除

1. **端口冲突**：如果 3000 端口已被占用，请更改 `-p` 参数（例如 `-p 3001:8080`）
2. **GPU 支持问题**：确保已正确安装 NVIDIA 驱动和容器工具包
3. **权限问题**：如果遇到权限错误，在命令前添加 `sudo`
4. **容器启动失败**：使用 `docker logs open-webui` 查看错误日志
