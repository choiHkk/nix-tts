import easydict


configs = easydict.EasyDict(
    # train
    
    
    # data
    n_symbols=190,
    n_speakers=1,
    n_mel_channels=80,
    max_len=1024,
    hop_length=256, 
    
    
    # model
    n_flows=2, 
    p_dropout=0.5, 
    hidden_channels=192,
    segment_size=8192, 
    inter_channels=32,
    gin_channels=256,
    kernel_size=5,
    resblock_kernel_sizes=[3,7,11],
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
    dilations=[1,2,4],
    upsample_rates=[8,8,2,2],
    upsample_initial_channel=256,
    upsample_kernel_sizes=[16,16,4,4],
    temperature=0.0005,
    resblock='ds',
)
