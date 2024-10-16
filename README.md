# 化工大模型问答

![design](./asset/design.svg)

### 待办

- [ ] Training
- [ ] Web UI
- [ ] history

### 文件说明

- `utils`用于存放工具调用函数
- `ChemAgent`包含agent类
- `prompt`存放提示词
  - `example`是样例集合
  - `system`存放状态改变的提示词
  - `tools`存放用于工具的提示词，检索，web划分为工具
- `config`存放基本配置
