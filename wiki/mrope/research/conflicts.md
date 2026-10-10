# 冲突记录

- 时间编号的步长｜位置：视频时间编号每加一对应多少帧｜两处说法：论文正文 template.tex L370「the temporal ID increments for each frame」（每帧加一）；官方仓库 rope2d.py L155「temporal_patch_size ... it's 2 frames」与 L362 示例「[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]」（每个时间片、即每两帧加一）｜页面采用：论文正文口径，写「逐帧加一」
