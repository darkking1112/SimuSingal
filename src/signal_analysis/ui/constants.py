"""界面常量：模式/导出选项、目标与来源文案、导入清单列定义与播放参数。"""

MODE_CHOICES = [("am", "AM 调幅"), ("fm", "FM 调频"), ("ssb", "SSB 单边带"),
                ("ask2", "2ASK 二进制幅移键控"), ("qpsk", "QPSK 四相相移键控"),
                ("qam16", "16QAM 正交幅度调制"), ("qam64", "64QAM 正交幅度调制"),
                ("fh_rc", "跳频 · 遥控链路 (FH-2FSK)"), ("fh_video", "跳频 · 图传链路 (FH-OFDM)")]
MODE_SHORT = {"am": "AM", "fm": "FM", "ssb": "SSB", "ask2": "2ASK", "qpsk": "QPSK",
              "qam16": "16QAM", "qam64": "64QAM", "fh_rc": "FH遥控", "fh_video": "FH图传"}
EXPORT_FORMATS = [("不导出（仅内部资产 .npy）", ""), ("NPY 格式 (.npy)", "npy"),
                  ("CSV 两列 I,Q (.csv)", "csv"), ("交织 IQ · int16 (.bin)", "iq16"),
                  ("交织 IQ · float32 (.bin)", "iq32"),
                  ("SigMF 双文件 (.sigmf-meta + .sigmf-data)", "sigmf")]
#: 目标粒度与来源分类的界面文案（与 storage 枚举一一对应）。
_SCOPE_LABELS = {"whole_record": "整条记录", "session": "会话", "hop": "单跳",
                 "segment": "片段"}
_SOURCE_KIND_LABELS = {"imported": "导入", "generated": "生成", "derived": "衍生",
                       "legacy": "历史", "manual": "手工", "algorithm": "算法",
                       "external": "外部"}
#: 目标参考参数版本的来源枚举与资产来源分类不同名（generator/import），单独映射。
_VERSION_SOURCE_LABELS = {"generator": "生成器", "import": "导入", "manual": "手工",
                          "external": "外部", "algorithm": "算法"}
#: 导入页文件清单：过滤器、可收集的后缀与列序（2026-10 改版，以文件清单为中心）。
IMPORT_FILE_FILTER = "数据 (*.npy *.csv *.bin *.raw *.iq *.sigmf-meta *.sigmf-data)"
IMPORT_SUFFIXES = (".npy", ".csv", ".bin", ".raw", ".iq", ".sigmf-meta", ".sigmf-data")
IMPORT_FORMAT_LABELS = {"npy": "NPY", "csv": "CSV", "binary": "IQ 二进制",
                        "sigmf": "SigMF", "unknown": "—"}
#: 调制下拉：A09 六类 + AM（字典外规范名）+ 未知；可自由输入其他类名（保留原名）。
IMPORT_MODULATION_CHOICES = ("未知", "FM", "SSB", "2ASK", "QPSK", "16QAM", "64QAM", "AM")
IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_RATE, IMPORT_COL_DTYPE = 0, 1, 2, 3
IMPORT_COL_ENDIAN, IMPORT_COL_POINTS, IMPORT_COL_RF_CENTER = 4, 5, 6
IMPORT_COL_MOD, IMPORT_COL_NOTE, IMPORT_COL_STATUS = 7, 8, 9
# 滚动瀑布图：时间窗内最多保留的帧数与单次刷新最多计算的帧数。
PLAY_MAX_ROWS = 360
PLAY_MAX_ROWS_PER_TICK = 64
# 波形图单次刷新最多绘制的样点数（超出则等间隔抽取）。
PLAY_WAVE_POINTS = 2000
