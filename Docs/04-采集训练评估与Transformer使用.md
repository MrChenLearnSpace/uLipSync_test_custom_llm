# 原 uLipSync 数据采集、Transformer 训练与 Unity 应用

本流程用原 uLipSync 的实时分析结果作为第一版监督标签，训练一个“PCM 音频窗口 -> A/I/U/E/O 概率”的 Transformer。它用于快速打通自研模型链路；原算法的标签本身不是人工真值，后续仍应按 `01-AI模型替换与数据集方案.md` 扩充人工校验与视频标注数据。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `Assets/Runtime/AiLipSyncTrainingDataRecorder.cs` | 在 Unity 中采集训练数据 |
| `Assets/Runtime/AiLipSyncTransformer.cs` | 运行 ONNX Transformer；未启用 Sentis 时回退固定输出 |
| `Tools/ai_lipsync/train.py` | PyTorch 训练 |
| `Tools/ai_lipsync/evaluate.py` | 独立评估与混淆矩阵 |
| `Tools/ai_lipsync/export_onnx.py` | 导出 Unity 使用的 ONNX |
| `Tools/ai_lipsync/model.py` | Transformer 网络定义 |

## 1. Unity 中采集数据

1. 使用原来的 `uLipSync`，不要在同一个输入对象同时启用 `AiLipSync` 或 `AiLipSyncTransformer`。
2. 在挂有 `uLipSync` 的 GameObject 添加 `AiLipSyncTrainingDataRecorder`。
3. 将该对象的 `uLipSync` 拖到 `Legacy Lip Sync`；若组件在同一对象，`Reset`/运行时会自动寻找。
4. 保持 `Model Input Sample Rate = 16000`、`Model Input Sample Count = 1024`、标签顺序为 `A/I/U/E/O`。这与默认 uLipSync Profile 的 16 kHz、1024 样本窗口时长一致。
5. 播放干净的单人语音，或启动麦克风；在组件菜单（右键标题）使用 `Start Recording`，结束时使用 `Stop Recording`。
6. Console 会打印数据目录。默认位置为 `Application.persistentDataPath/uLipSyncTrainingData/<时间戳>`，不同平台的实际 persistentDataPath 不同；通过 `Last Output Directory` 字段可复制当前路径。

建议每次录制只使用一种语言、一个说话人或一个噪声条件，以便后续按会话划分训练集、验证集和测试集。不要在同一说话人的片段间随机切分后宣称模型泛化。

## 2. 采集文件格式

每个录制会话生成：

```text
<session>/
  metadata.json
  inputs.f32
  labels.f32
  manifest.jsonl
```

- `inputs.f32`：little-endian `float32`，每行连续 1024 个值；单声道 PCM，旧到新，16 kHz，通常范围为 `[-1, 1]`。
- `labels.f32`：little-endian `float32`，每行连续 5 个归一化概率；默认顺序固定为 `[A, I, U, E, O]`。
- `manifest.jsonl`：每行记录对应行号、DSP 时间、原算法主音素和音量。
- `metadata.json`：唯一可信的形状、采样率、标签顺序说明。训练脚本会拒绝文件大小和元数据不一致的会话。

采集器通过 `uLipSync.TryCopyLastAnalyzedAudio` 取得原 `LipSyncJob` 使用的同一循环缓冲窗口，再按固定模型形状重采样。不要在开始采集后任意更改 Profile 的采样率或样本数；如果必须改，使用独立会话并将录制器输入形状与 Profile 对齐。

## 3. 安装训练环境

在项目根目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Tools\ai_lipsync\requirements.txt
```

Python 建议使用 3.10 或更新版本。PyTorch 的 GPU 版本需按你的 CUDA 环境从 PyTorch 官方安装说明选择；直接按上述 requirements 安装通常会得到 CPU 版本或平台默认版本。

## 4. 训练

将一个或合并后的训练会话目录传入训练脚本：

```powershell
py Tools\ai_lipsync\train.py "C:\路径\到\训练会话" --output artifacts\lipsync_transformer.pt --epochs 30
```

训练脚本使用 soft-label KL 散度，输出模型最好的验证集 checkpoint 和同名 JSON 摘要。它的随机验证集切分只适合快速验证；正式评估必须使用另一个说话人或另一个录制会话。

## 5. 评估

对从未参与训练的会话运行：

```powershell
py Tools\ai_lipsync\evaluate.py "C:\路径\到\测试会话" artifacts\lipsync_transformer.pt --output artifacts\evaluation.json
```

输出包含 KL 散度、平均 L1、top-1 一致率和混淆矩阵。由于标签来自原 uLipSync，指标表示“复现原算法”的程度，不直接等于真实视觉自然度；仍需在实际角色上做回放评审。

## 6. 导出 ONNX

```powershell
py Tools\ai_lipsync\export_onnx.py artifacts\lipsync_transformer.pt --output artifacts\lipsync_transformer.onnx
```

固定 ONNX 契约：

| 张量 | 名称 | 类型与形状 | 含义 |
| --- | --- | --- | --- |
| 输入 | `audio_pcm` | `float32 [1, 1024]` | 16 kHz、单声道、旧到新的 PCM 窗口 |
| 输出 | `viseme_logits` | `float32 [1, 5]` | 未归一化 logits，顺序 `A/I/U/E/O` |

导出后不要只在 Python 中测试。先用同一段 Unity 音频在原算法和 AI 模型间来回切换，检查延迟、静音、闭嘴和每个嘴型的映射。

## 7. Unity 应用 ONNX

`AiLipSyncTransformer` 在没有额外依赖时仍可编译，并回退使用 `AiLipSync.fixedModelOutput`。要真正运行 ONNX：

1. 通过 Unity Package Manager 添加与 Unity 版本兼容的 Sentis 包。
2. 在 Player Settings > Scripting Define Symbols 添加 `ULIPSYNC_SENTIS`。
3. 等 Unity 编译完成后，在对象添加 `AiLipSyncTransformer`，指定 Sentis 的 `Model Asset`，并按平台选择 CPU/GPU backend。
4. 确保 `Model Input Sample Rate/Count` 为训练时的 `16000/1024`，`Model Output Labels` 顺序为训练 checkpoint 的 `A/I/U/E/O`。
5. 将 `AiLipSyncTransformer > On Lip Sync Update` 绑定到原有 `uLipSyncBlendShape.OnLipSyncUpdate`、`uLipSyncAnimator.OnLipSyncUpdate` 或 VRM 驱动器。
6. 禁用同一音频输入对象上的原 `uLipSync`，避免两个组件同时写表情。

`AiLipSyncTransformer` 在 C# 中对 ONNX 输出 logits 做 softmax，再交给 `AiLipSync` 转为原项目的 `LipSyncInfo`；因此原有 BlendShape、Animator、VRM 表现层不需要为 Transformer 改接口。

## 注意事项

- 采集器默认跳过静音帧，第一版模型不含 `sil` 类；运行时静音由 `silenceVolumeThreshold` 控制。若要训练语音活动检测，请关闭 `Record Only When Voice`，添加 `sil` 标签，并同步改训练和模型输出维度。
- 新模型的输入时长、采样率、归一化和标签顺序必须与训练数据完全一致。最常见的部署问题是训练 `1024@16k`、运行时却误用 `1600@16k`。
- 目前录制标签来自 MFCC 原算法，所以 Transformer 的第一目标是蒸馏/复现原算法。要超过原算法，应引入人工修正的 viseme 边界和真实表情/视频监督数据。
