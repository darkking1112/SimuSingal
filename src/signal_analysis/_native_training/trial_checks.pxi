cdef extern from "trial_limits.h":
    enum:
        SA_BUILD_TRIAL
        SA_TRIAL_MAX_TRAIN_SAMPLES
        SA_TRIAL_MAX_IQ_SAMPLES
        SA_TRIAL_MAX_MODEL_PARAMETERS
        SA_TRIAL_MAX_EPOCHS


cdef inline Py_ssize_t _checked_epochs(object epochs) except -1:
    if type(epochs) is not int or epochs < 1:
        raise ValueError("训练轮数必须为正整数")
    if SA_BUILD_TRIAL and epochs > SA_TRIAL_MAX_EPOCHS:
        raise ValueError(
            f"TRIAL_EPOCH_LIMIT: 试用版最多训练 {SA_TRIAL_MAX_EPOCHS} 轮，当前请求 {epochs} 轮")
    if epochs > 10000:
        raise ValueError("训练轮数不能超过 10000")
    return epochs


cdef inline void _guard_train_count(object count) except *:
    if type(count) is not int or count < 1:
        raise ValueError("训练集不能为空")
    if SA_BUILD_TRIAL and count > SA_TRIAL_MAX_TRAIN_SAMPLES:
        raise ValueError(
            f"TRIAL_DATA_LIMIT: 试用版最多使用 {SA_TRIAL_MAX_TRAIN_SAMPLES} 条训练样本，"
            f"当前 {count} 条")


cdef inline void _guard_iq_data(object train_x, object val_x) except *:
    if train_x.ndim != 3 or val_x.ndim != 3:
        raise ValueError("IQ 输入必须是 (M, 2, N)")
    if train_x.shape[1] != 2 or val_x.shape[1] != 2:
        raise ValueError("IQ 输入必须包含 I、Q 两个通道")
    if train_x.shape[2] != val_x.shape[2]:
        raise ValueError("训练集与验证集窗口长度不一致")
    if val_x.shape[0] < 1:
        raise ValueError("验证集不能为空")
    _guard_train_count(int(train_x.shape[0]))
    if SA_BUILD_TRIAL and train_x.shape[2] > SA_TRIAL_MAX_IQ_SAMPLES:
        raise ValueError(
            f"TRIAL_IQ_LENGTH_LIMIT: 试用版 IQ 窗口最多 {SA_TRIAL_MAX_IQ_SAMPLES} 点，"
            f"当前 {train_x.shape[2]} 点")


cdef inline object _guard_model(object model):
    cdef object total = 0
    cdef object parameter
    try:
        for parameter in model.parameters():
            total += parameter.numel()
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        raise ValueError("模型参数尚未初始化，无法检查规模") from exc
    if SA_BUILD_TRIAL and total > SA_TRIAL_MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"TRIAL_MODEL_LIMIT: 试用版模型最多 {SA_TRIAL_MAX_MODEL_PARAMETERS} 个参数，"
            f"当前 {total} 个")
    return total
