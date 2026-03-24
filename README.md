# 图像增强算法集成 Web 服务

这是一个 **FastAPI 单体式图像增强平台**：

- **前后端不分离**：FastAPI 直接托管 `static/index.html`
- **逻辑解耦**：前端下拉框不写死，动态调用 `/api/algorithms`
- **插件式扩展**：新增算法只需新增一个 wrapper Python 文件，并打上装饰器
- **硬件约束感知**：若算法声明 `requires_gpu=True`，但当前机器无 GPU，则 API 直接拦截并返回可读错误
- **兼容工程隔离**：当开源仓库依赖与主环境冲突时，推荐通过 `subprocess + conda run` 在独立环境中执行推理脚本

---

## 1. 项目分层原则

### 1.1 `third_party/` 放原始开源代码

这里放你从 GitHub clone 下来的原始仓库，尽量 **不改或少改**。

示例：

```bash
git clone https://github.com/xxx/DarkIR.git third_party/darkir