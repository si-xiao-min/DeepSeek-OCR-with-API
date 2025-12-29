# 磁盘空间问题排查与解决指南

本文档详细说明如何分析、排查和解决磁盘空间不足的问题。

---

## 📋 目录

1. [快速诊断](#快速诊断)
2. [问题分析步骤](#问题分析步骤)
3. [常见原因与解决方案](#常见原因与解决方案)
4. [预防措施](#预防措施)
5. [实战案例](#实战案例)

---

## 🔍 快速诊断

当系统提示磁盘空间不足时，首先运行以下命令快速了解情况：

```bash
# 1. 查看整体磁盘使用情况
df -h

# 2. 查看当前目录下最大的文件和目录（Top 20）
du -sh * 2>/dev/null | sort -rh | head -20

# 3. 查看系统根目录占用情况
du -sh /* 2>/dev/null | sort -rh | head -20
```

**预期输出示例：**
```
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   28G  2.6G  92% /
```

如果 `Use%` 超过 **85%**，就需要立即清理了！

---

## 📊 问题分析步骤

### Step 1: 确认磁盘使用情况

```bash
# 查看所有挂载点的使用情况
df -h

# 查看inode使用情况（有时候磁盘空间够但inode用完）
df -i
```

### Step 2: 定位大文件和大目录

```bash
# 查看当前目录下最大的文件和目录
du -sh * | sort -rh | head -20

# 查看指定目录下的占用
du -sh /path/to/directory/* | sort -rh | head -20

# 递归查找大于100MB的文件
find /path -type f -size +100M -exec ls -lh {} \; 2>/dev/null
```

### Step 3: 检查常见缓存目录

```bash
# Python pip 缓存
du -sh ~/.cache/pip

# Conda 缓存
du -sh ~/miniconda3/pkgs  # 或 ~/anaconda3/pkgs

# HuggingFace 模型缓存
du -sh ~/.cache/huggingface

# Docker 占用
docker system df

# 日志文件
du -sh /var/log/*
ls -lh /var/log/*.log
```

### Step 4: 查找大日志文件

```bash
# 查找大于10MB的日志文件
find /var/log -type f -size +10M -exec ls -lh {} \;

# 查找当前目录下的大日志
find . -name "*.log" -type f -size +10M -exec ls -lh {} \;
```

### Step 5: 检查临时文件

```bash
# 系统临时文件
du -sh /tmp/*

# 用户临时文件
du -sh /tmp/user/*
```

---

## 🐛 常见原因与解决方案

### 1. Python 包缓存（pip）

**症状**: `~/.cache/pip` 目录占用数GB空间

**原因**:
- pip 下载的包会被缓存，避免重复下载
- 长期使用后会积累大量缓存

**解决方案**:
```bash
# 清理所有 pip 缓存
pip cache purge

# 查看缓存占用
pip cache info

# 仅列出缓存的包（不删除）
pip cache list
```

**预期释放空间**: 1GB - 5GB（取决于使用频率）

---

### 2. Conda 环境缓存

**症状**: `~/miniconda3/pkgs` 或 `~/anaconda3/pkgs` 占用大量空间

**原因**:
- conda 下载的包 tarball
- 已卸载包的残留
- 索引缓存

**解决方案**:
```bash
# 清理所有缓存（包括索引缓存、包缓存、临时文件）
conda clean --all -y

# 仅清理索引缓存
conda clean --index-cache

# 仅清理未使用的包
conda clean --packages

# 仅清理tarball
conda clean --tarballs
```

**预期释放空间**: 200MB - 2GB

---

### 3. 日志文件过大

**症状**:
- `/var/log/syslog` 或 `/var/log/messages` 数GB大小
- 应用日志文件持续增长

**解决方案**:

```bash
# 清空日志文件（保留文件）
sudo truncate -s 0 /var/log/syslog
sudo truncate -s 0 /var/log/messages

# 或者使用 logrotate 自动管理日志
sudo logrotate -f /etc/logrotate.conf
```

**配置日志轮转** (`/etc/logrotate.d/custom-app`):
```
/path/to/your/app.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    maxsize 100M
}
```

**预期释放空间**: 取决于日志大小

---

### 4. Docker 占用

**症状**:
- Docker 镜像、容器、卷占用大量空间
- `docker system df` 显示高占用

**解决方案**:
```bash
# 查看Docker占用详情
docker system df

# 清理未使用的镜像、容器、网络、构建缓存
docker system prune -a

# 清理未使用的卷
docker volume prune

# 清理构建缓存
docker builder prune

# 一键清理所有（谨慎使用！）
docker system prune -a --volumes
```

**预期释放空间**: 1GB - 10GB+

---

### 5. HuggingFace 模型缓存

**症状**: `~/.cache/huggingface` 占用数GB空间

**原因**:
- 下载的预训练模型
- 数据集缓存
- Pipeline 缓存

**解决方案**:
```bash
# 查看HuggingFace缓存占用
du -sh ~/.cache/huggingface/hub
du -sh ~/.cache/huggingface/datasets

# 删除特定模型缓存
rm -rf ~/.cache/huggingface/hub/models--model-name

# 删除数据集缓存
rm -rf ~/.cache/huggingface/datasets/*

# 清理所有缓存（谨慎！）
rm -rf ~/.cache/huggingface/
```

**预期释放空间**: 1GB - 50GB（取决于缓存的模型数量）

---

### 6. 临时文件

**症状**: `/tmp` 目录占用大量空间

**解决方案**:
```bash
# 清理超过7天的临时文件
sudo find /tmp -type f -atime +7 -delete

# 清理当前用户的临时文件
rm -rf /tmp/user/*
```

---

### 7. 系统快照和备份

**症状**: Linux 系统快照占用磁盘空间

**检查**:
```bash
# 查看LV快照
sudo lvdisplay

# 查看Timeshift快照
sudo timeshift --list
```

**解决方案**:
```bash
# 删除旧快照（根据具体工具）
sudo timeshift --delete --snapshot '2024-01-01_12-00-00'

# 保留最近3个快照
sudo timeshift --delete-all-but-snapshot 3
```

---

### 8. 包管理器缓存

**APT (Debian/Ubuntu)**:
```bash
# 清理已下载的包文件
sudo apt clean

# 删除无法再下载的过时包
sudo apt autoclean

# 删除为满足依赖而安装的、现在不再需要的包
sudo apt autoremove
```

**YUM/DNF (CentOS/Fedora)**:
```bash
# 清理缓存
sudo dnf clean all

# 删除不再需要的依赖包
sudo dnf autoremove
```

---

## 🛡️ 预防措施

### 1. 定期清理脚本

创建清理脚本 `/usr/local/bin/cleanup.sh`:

```bash
#!/bin/bash
# 磁盘清理脚本

echo "开始清理磁盘空间..."

# 清理 pip 缓存
pip cache purge

# 清理 conda 缓存
conda clean --all -y

# 清理系统日志（保留最近7天）
sudo journalctl --vacuum-time=7d

# 清理 APT 缓存
sudo apt clean && sudo apt autoclean && sudo apt autoremove -y

# 清理临时文件（超过7天）
sudo find /tmp -type f -atime +7 -delete 2>/dev/null

echo "清理完成！"
df -h
```

**设置定时任务**（每周日凌晨2点执行）:
```bash
# 编辑 crontab
crontab -e

# 添加以下行
0 2 * * 0 /usr/local/bin/cleanup.sh >> /var/log/cleanup.log 2>&1
```

### 2. 监控告警

**磁盘监控脚本** `/usr/local/bin/disk-monitor.sh`:

```bash
#!/bin/bash
# 磁盘监控告警脚本

THRESHOLD=85
USAGE=$(df / | grep / | awk '{print $5}' | sed 's/%//g')

if [ $USAGE -gt $THRESHOLD ]; then
    echo "警告: 磁盘使用率达到 ${USAGE}%，超过阈值 ${THRESHOLD}%"
    # 可以发送邮件或通知
    # mail -s "磁盘空间告警" admin@example.com <<< "磁盘空间不足"
fi
```

### 3. 日志轮转配置

确保关键应用配置了日志轮转：

```bash
# 为应用创建日志轮转配置
sudo nano /etc/logrotate.d/myapp
```

内容：
```
/var/log/myapp/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 www-data www-data
    sharedscripts
    postrotate
        systemctl reload myapp >/dev/null 2>&1 || true
    endscript
}
```

### 4. Docker 定期清理

```bash
# 添加到 crontab
0 3 * * * docker system prune -f --volumes >> /var/log/docker-cleanup.log 2>&1
```

### 5. 配置磁盘使用限制

**为用户配置磁盘配额**（可选）:
```bash
# 启用配额
sudo quotacheck -cum /
sudo quotaon /

# 为用户设置配额（例如：最大50GB）
sudo setquota username 50G 55G 0 0 /
```

---

## 📚 实战案例

### 案例1: DeepSeek-OCR 服务磁盘清理

**问题现象**:
- 磁盘使用率 92% (28G/30G)
- API 服务运行缓慢

**排查过程**:
```bash
# 1. 查看整体情况
df -h
# overlay    30G   28G  2.6G  92% /

# 2. 定位大目录
du -sh /* | sort -rh | head -20
# 32G  /usr
# 13G  /hy-tmp
# 7.6G /root

# 3. 深入分析
du -sh /root/.cache/*
# 3.9G  /root/.cache/JetBrains
# 3.1G  /root/.cache/pip

du -sh /usr/local/*
# 16G   /usr/local/miniconda3
```

**解决方案**:
```bash
# 清理 pip 缓存（释放 3.1GB）
pip cache purge

# 清理 conda 缓存（释放 240MB）
conda clean --all -y
```

**最终结果**:
- 磁盘使用率: 92% → 80%
- 可用空间: 2.6G → 6.1G
- 释放空间: ~4GB

---

### 案例2: 日志文件占满磁盘

**问题现象**:
- 服务器无法写入新文件
- 应用报错 "No space left on device"

**排查过程**:
```bash
# 查找大日志文件
find /var/log -type f -size +100M -exec ls -lh {} \;

# 发现问题
-rw-r--r-- 1 root root  8.5G Jan 15 10:30 /var/log/syslog
-rw-r--r-- 1 root root  3.2G Jan 15 10:30 /var/log/messages
```

**解决方案**:
```bash
# 清空日志文件
sudo truncate -s 0 /var/log/syslog
sudo truncate -s 0 /var/log/messages

# 配置日志轮转防止再次发生
sudo nano /etc/logrotate.d/syslog-custom
```

---

## 📝 快速参考命令速查表

| 任务 | 命令 |
|------|------|
| 查看磁盘使用 | `df -h` |
| 查看目录大小 | `du -sh /path` |
| 查找大文件 | `find /path -size +100M` |
| 清理 pip 缓存 | `pip cache purge` |
| 清理 conda 缓存 | `conda clean --all -y` |
| 清理 Docker | `docker system prune -a` |
| 清理 APT 缓存 | `apt clean && apt autoremove` |
| 清理日志 | `journalctl --vacuum-time=7d` |
| 清空文件 | `truncate -s 0 /path/to/file` |
| 查看 inode 使用 | `df -i` |

---

## ⚠️ 注意事项

1. **删除前确认**: 删除文件前务必确认文件内容，避免误删重要数据
2. **停用服务**: 清理应用日志前最好先停用服务
3. **备份重要数据**: 清理前备份重要配置和数据
4. **权限问题**: 清理系统文件可能需要 sudo 权限
5. **正在使用的文件**: 即使删除正在使用的文件，空间也可能不会立即释放（进程重启后才释放）
6. **HuggingFace 缓存**: 删除前确认是否需要重新下载模型

---

## 🔗 相关资源

- [Linux du 命令详解](https://linux.die.net/man/1/du)
- [Linux df 命令详解](https://linux.die.net/man/1/df)
- [Logrotate 官方文档](https://linux.die.net/man/8/logrotate)
- [Docker 清理最佳实践](https://docs.docker.com/config/pruning/)

---

**文档版本**: 1.0
**最后更新**: 2025-12-29
**维护者**: DeepSeek-OCR Team
