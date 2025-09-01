数据可在 https://github.com/jinbochen0823/RCG2ECG 中获取
数据格式：
- 共 91 个mat文件
- 每个mat中是一个名为data的struct
- struct中有7个字段：
    - RCG：35505*50 double，预处理后的雷达数据，每一列是一个体素点的时序数据
    - ECG：35505*1 double，心电图数据
    - posXYZ：50*3 double，体素点位置
    - id：int，受试者的id
    - gender：'boy'/'girl'
    - age：int，受试者年龄
    - physistatus：4 different physiological status， 'NB'(normal-breath), 'IB'(irregular-breath), 'PE'(post-exercise) and 'SP'(sleep)